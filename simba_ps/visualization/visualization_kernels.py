import math

from numba import cuda

import simba_ps.physics_kernels
from simba_ps.physics_kernels import mass_from_type
from simba_ps.utils.util_kernels import grid_from_pos, grid_point_outside_field


@cuda.jit(device=True, inline=True)
def compute_color_gradient(gradient_colors, color_count, color_idx, v, mag_min, mag_max):
    if v > mag_max:
        v = mag_max
    elif v < mag_min:
        v = mag_min

    v = (v - mag_min) / (mag_max - mag_min)

    c_idx0 = int((color_count - 1) * v)
    if c_idx0 < 0:
        c_idx0 = 0

    if c_idx0 >= color_count - 1:
        c_idx0 = color_count - 1
        c_idx1 = c_idx0
    else:
        c_idx1 = c_idx0 + 1
    size = 1.0 / (color_count - 1)
    v = (v - c_idx0 * size) / size

    c0 = gradient_colors[c_idx0, color_idx]
    c1 = gradient_colors[c_idx1, color_idx]
    return ((1.0 - v) * c0 + v * c1)


@cuda.jit
def set_colors_from_vector_magnitude_kernel(p_types, vectors, gradient_colors, camera_colors, mag_min, mag_max):
    particle_idx, color_idx = cuda.grid(2)
    if particle_idx >= p_types.shape[0] or color_idx >= 3:
        return
    p_type_cache = cuda.shared.array(32, 'int16')
    vector_cache = cuda.shared.array((32, 2), 'float32')

    if cuda.threadIdx.y < 2:
        vector_cache[cuda.threadIdx.x, cuda.threadIdx.y] = vectors[particle_idx, cuda.threadIdx.y]
    else:
        p_type_cache[cuda.threadIdx.x] = p_types[particle_idx]
    cuda.syncthreads()
    p_type = p_type_cache[cuda.threadIdx.x]
    if simba_ps.physics_kernels.mass_from_type(p_type) <= 0.0:
        return
    v = math.sqrt(vector_cache[cuda.threadIdx.x, 0] ** 2 + vector_cache[cuda.threadIdx.x, 1] ** 2)
    color_count = gradient_colors.shape[0]

    camera_colors[particle_idx, color_idx] = compute_color_gradient(gradient_colors, color_count, color_idx,
                                                                    v, mag_min, mag_max)


@cuda.jit
def set_color_from_p_type_kernel(p_types, colors, target_p_type, target_color):
    particle_idx, color_idx = cuda.grid(2)
    if particle_idx >= p_types.shape[0]:
        return
    if p_types[particle_idx] == target_p_type:
        colors[particle_idx, color_idx] = target_color[color_idx]


@cuda.jit
def render_particles_smooth(p_types, positions, view, scale, r,
                            grid_size,
                            densities):
    particle_idx, pixel_idx = cuda.grid(2)
    if particle_idx >= positions.shape[0]:
        return
    transformed_pos = cuda.shared.array((32, 2), 'float64')
    p_types_cache = cuda.shared.array(32, 'int16')

    if cuda.threadIdx.y < 2:
        p = positions[particle_idx, cuda.threadIdx.y]
        screen_geom = float(densities.shape[cuda.threadIdx.y])
        view_point = view[cuda.threadIdx.y]
        transformed_pos[cuda.threadIdx.x, cuda.threadIdx.y] = 0.5 * screen_geom + scale * (p - view_point)
    elif cuda.threadIdx.y == 2:
        p_types_cache[cuda.threadIdx.x] = p_types[particle_idx]

    cuda.syncthreads()

    j = int(pixel_idx / grid_size)
    i = int(pixel_idx - j * grid_size)
    if i >= grid_size or j >= grid_size:
        return

    p_type = p_types_cache[cuda.threadIdx.x]
    draw_particle_smooth(p_type,
                         i, j,
                         transformed_pos[cuda.threadIdx.x, 0], transformed_pos[cuda.threadIdx.x, 1],
                         grid_size,
                         r[p_type],
                         densities)


@cuda.jit(device=True, inline=True)
def draw_particle_smooth(p_type, i, j, pos_x, pos_y, grid_size, r, densities):
    pixel_x, pixel_y = grid_from_pos(pos_x, pos_y, i, j,
                                     grid_size, grid_size)
    if not (0 <= pixel_x < densities.shape[0]):
        return
    if not (0 <= pixel_y < densities.shape[1]):
        return
    dx = (pixel_x - pos_x)
    dy = (pixel_y - pos_y)
    pixel_r = math.sqrt(dx ** 2 + dy ** 2)
    if pixel_r > r:
        return
    if mass_from_type(p_type) > 0:
        cuda.atomic.add(densities, (pixel_x, pixel_y, p_type), (r - pixel_r) / r)
        return
    densities[pixel_x, pixel_y, p_type] = 1.0


@cuda.jit
def render_densities(densities, gradient_colors, out_buffer, mag_min, mag_max):
    pixel_idx, density_idx, color_idx = cuda.grid(3)
    width = out_buffer.shape[0]
    x = pixel_idx % width
    y = pixel_idx // width
    if grid_point_outside_field(x, y, out_buffer):
        return
    d = densities[x, y, density_idx]
    if d == 0:
        return
    color_gradient_for_current_density = gradient_colors[density_idx]
    color_count = color_gradient_for_current_density.shape[0]

    c = compute_color_gradient(color_gradient_for_current_density, color_count, color_idx, d, mag_min, mag_max)
    cuda.atomic.max(out_buffer, (x, y, color_idx), c * 255)


@cuda.jit
def copy_buffer(src, dst):
    i, j, k = cuda.grid(3)
    if i >= src.shape[0] or j >= src.shape[1] or k >= src.shape[2]:
        return
    dst[i, j, k] = src[i, j, k]


@cuda.jit
def render_particles_scaled_colored(positions, view, scale, r2,
                                    grid_size,
                                    out_buffer,
                                    colors):
    particle_idx, pixel_idx = cuda.grid(2)
    if particle_idx >= positions.shape[0]:
        return
    transformed_pos = cuda.shared.array((32, 2), 'float64')
    color = cuda.shared.array((32, 3), 'int16')
    if cuda.threadIdx.y < 2:
        p = positions[particle_idx, cuda.threadIdx.y]
        screen_geom = float(out_buffer.shape[cuda.threadIdx.y])
        view_point = view[cuda.threadIdx.y]
        transformed_pos[cuda.threadIdx.x, cuda.threadIdx.y] = 0.5 * screen_geom + scale * (p - view_point)
    elif 2 <= cuda.threadIdx.y <= 4:
        color[cuda.threadIdx.x, cuda.threadIdx.y - 2] = colors[particle_idx, cuda.threadIdx.y - 2] * 255
    cuda.syncthreads()

    j = int(pixel_idx / grid_size)
    i = int(pixel_idx - j * grid_size)
    if i >= grid_size or j >= grid_size:
        return

    draw_particle(i, j,
                  transformed_pos[cuda.threadIdx.x, 0], transformed_pos[cuda.threadIdx.x, 1],
                  grid_size,
                  r2,
                  out_buffer,
                  color[cuda.threadIdx.x, 0],
                  color[cuda.threadIdx.x, 1],
                  color[cuda.threadIdx.x, 2])


@cuda.jit(device=True, inline=True)
def draw_particle(i, j, pos_x, pos_y, grid_size, r2, out_buffer, r, g, b):
    if r < 0 or g < 0 or b < 0:
        return
    pixel_x, pixel_y = grid_from_pos(pos_x, pos_y, i, j,
                                     grid_size, grid_size)
    if not (0 <= pixel_x < out_buffer.shape[0]):
        return
    if not (0 <= pixel_y < out_buffer.shape[1]):
        return
    dx = (pixel_x - pos_x)
    dy = (pixel_y - pos_y)
    pixel_r2 = dx ** 2 + dy ** 2
    if pixel_r2 <= r2:
        out_buffer[pixel_x, pixel_y, 0] = r  # * (r2 - pixel_r2)/r2
        out_buffer[pixel_x, pixel_y, 1] = g  # * (r2 - pixel_r2)/r2
        out_buffer[pixel_x, pixel_y, 2] = b  # * (r2 - pixel_r2)/r2


@cuda.jit
def fade_3d_field(field, value):
    x, y, z = cuda.grid(3)
    if x >= field.shape[0] or y >= field.shape[1] or z >= field.shape[2]:
        return
    field[x, y, z] = field[x, y, z] * value


@cuda.jit
def set_colors_in_rect_kernel(colors, image, p_types, positions, cx, cy, p_types_to_set):
    idx, c = cuda.grid(2)

    if idx >= positions.shape[0] or c >= 3:
        return
    x = int(positions[idx, 0] - cx)
    y = int(positions[idx, 1] - cy)
    if simba_ps.physics_kernels.grid_point_outside_field(y, x, image):
        return

    for i in range(p_types_to_set.shape[0]):
        if p_types_to_set[i] < 0 or p_types[idx] == p_types_to_set[i]:
            colors[idx, c] = image[y, x, c]
            return
