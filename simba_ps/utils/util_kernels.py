import math

import numba.cuda
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def set_vector_for_p_type(p_types, vectors, target_p_type, target_vector):
    idx, c = cuda.grid(2)
    if idx >= p_types.shape[0]:
        return
    p_type_cache = cuda.shared.array((256), 'int')
    if c == 0:
        p_type_cache[cuda.threadIdx.x] = p_types[idx]
    cuda.syncthreads()
    self_p_type = p_type_cache[cuda.threadIdx.x]
    if self_p_type != target_p_type:
        return
    vectors[idx, c] = target_vector[c]


@cuda.jit
def place_particles_from_img(img,
                             target_p_type, target_color, target_density, target_velocity,
                             p_types, positions, velocities,
                             rng_states,
                             particle_count):
    x, y, c = cuda.grid(3)
    if x >= img.shape[0] or y >= img.shape[1]:
        return
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    pixel_color = cuda.shared.array((16, 16, 3), 'uint8')
    target_color_shared = cuda.shared.array((16, 16, 3), 'uint8')
    pixel_color[tx, ty, c] = img[x, y, c]
    target_color_shared[tx, ty, c] = target_color[c]
    cuda.syncthreads()

    if c > 0:
        return

    if pixel_color[tx, ty, 0] != target_color_shared[tx, ty, 0] or \
            pixel_color[tx, ty, 1] != target_color_shared[tx, ty, 1] or \
            pixel_color[tx, ty, 2] != target_color_shared[tx, ty, 2]:
        return

    tid = x + y * img.shape[0]
    p = xoroshiro128p_uniform_float32(rng_states, tid)
    if p > target_density:
        return

    cnt = cuda.atomic.add(particle_count, 0, 1)
    if cnt >= p_types.shape[0]:
        print("Max particles reached", cnt)
        return

    positions[cnt, 0] = x
    positions[cnt, 1] = y

    a = 2 * 3.1415926 * (xoroshiro128p_uniform_float32(rng_states, tid))
    velocities[cnt, 0] = math.cos(a) * target_velocity
    velocities[cnt, 1] = math.sin(a) * target_velocity

    p_types[cnt] = target_p_type


def get_bpg(tpb, *args):
    blocks_per_grid = tuple([math.ceil(arg / tpb[idx]) for idx, arg in enumerate(args)])
    return blocks_per_grid


@cuda.jit(device=True, inline=True)
def grid_from_pos(x, y, i, j, width, height):
    integer_x = int(x)
    integer_y = int(y)
    idx_i = i - int(width / 2)
    idx_j = j - int(height / 2)
    grid_x = integer_x + idx_i
    grid_y = integer_y + idx_j
    return grid_x, grid_y


@cuda.jit
def apply_round(variable):
    idx = cuda.grid(1)
    variable[idx, 0] = round(variable[idx, 0], 10)
    variable[idx, 1] = round(variable[idx, 1], 10)


@cuda.jit
def copy3d(dst, src):
    x, y, z = cuda.grid(3)
    if x >= dst.shape[0] or y >= dst.shape[1] or z >= dst.shape[2]:
        return
    dst[x, y, z] = src[x, y, z]


@numba.cuda.jit(device=True, inline=True)
def grid_point_outside_field(grid_x, grid_y, field):
    return grid_x < 0 or grid_y < 0 or grid_x >= field.shape[0] or grid_y >= field.shape[1]


@cuda.jit
def select_particle_kernel(positions, x, y, out_idx):
    idx = cuda.grid(1)
    if idx >= positions.shape[0]:
        return
    if abs(positions[idx, 0] - x) < 1.0 and abs(positions[idx, 1] - y) < 1.0:
        out_idx[0] = idx
