import math

from numba import cuda

from simba_ps.physics_kernels import MAX_P_TYPES, mass_from_type


@cuda.jit
def clear_thermostate_data(avg_v, temperatures):
    rect_idx = cuda.grid(1)
    avg_v[rect_idx, 0] = 0
    avg_v[rect_idx, 1] = 0
    avg_v[rect_idx, 2] = 0
    temperatures[rect_idx] = 0


@cuda.jit(device=True)
def rect_from_pos(positions, rects):
    idx, rect_idx = cuda.grid(2)
    if idx >= positions.shape[0] or rect_idx >= rects.shape[0]:
        return -1, -1, (-1.0, -1.0, -1.0, -1.0)
    left = rects[rect_idx, 0]
    top = rects[rect_idx, 1]
    right = left + rects[rect_idx, 2]
    bottom = top + rects[rect_idx, 3]

    x = positions[idx, 0]
    y = positions[idx, 1]

    if not (left <= x < right):
        return -1, -1, (-1.0, -1.0, -1.0, -1.0)

    if not (top <= y < bottom):
        return -1, -1, (-1.0, -1.0, -1.0, -1.0)

    return idx, rect_idx, (top, bottom, left, right)


@cuda.jit
def set_temperature_in_rect(p_types, positions, velocities, rects, temperatures,  avg_v, target_temperatures):
    idx, rect_idx, rect = rect_from_pos(positions, rects)
    if idx < 0:
        return
    target_t = target_temperatures[rect_idx]

    if target_t < 0.0:
        return
    current_t = temperatures[rect_idx] * 1e-6

    cnt = avg_v[rect_idx, 2]

    v_x, v_y = velocities[idx, 0], velocities[idx, 1]

    if current_t <= 0.0:
        return
    scale = math.sqrt(target_t / current_t)

    velocities[idx, 0] = v_x*scale
    velocities[idx, 1] = v_y*scale



@cuda.jit
def compute_temperature_in_rects(p_types, positions, velocities, rects, avg_v, temperatures):
    idx, rect_idx, rect = rect_from_pos(positions, rects)
    if idx < 0:
        return

    particle_type = p_types[idx]
    if particle_type == 0 or particle_type >= MAX_P_TYPES:
        return

    m = mass_from_type(particle_type)
    cnt = avg_v[rect_idx, 2]
    avg_v_vec_x, avg_v_vec_y = avg_v[rect_idx, 0] / cnt, avg_v[rect_idx, 1]/cnt
    v_x, v_y = velocities[idx, 0], velocities[idx, 1]

    dv_x = v_x - avg_v_vec_x
    dv_y = v_y - avg_v_vec_y
    dv2 = dv_x**2 + dv_y**2
    cuda.atomic.add(temperatures, rect_idx, 0.5 * m * dv2 / cnt)


@cuda.jit
def det_compute_temperature_in_rects(p_types, positions, velocities, rects, avg_v, temperatures):
    idx, rect_idx, rect = rect_from_pos(positions, rects)
    if idx < 0:
        return

    particle_type = p_types[idx]
    if particle_type == 0 or particle_type >= MAX_P_TYPES:
        return

    m = mass_from_type(particle_type)
    cnt = avg_v[rect_idx, 2]

    avg_v_x, avg_v_y = avg_v[rect_idx, 0]*1.0e-6 / cnt, avg_v[rect_idx, 1]*1.0e-6 / cnt
    v_x, v_y = velocities[idx, 0], velocities[idx, 1]

    dv_x = v_x - avg_v_x
    dv_y = v_y - avg_v_y

    cuda.atomic.add(temperatures, rect_idx, 1e6 * 0.5 * m * (dv_x**2 + dv_y**2) / cnt)

@cuda.jit
def compute_avg_v_in_rects(p_types, positions, velocities, rects, avg_v):
    idx, rect_idx, rect = rect_from_pos(positions, rects)
    if idx < 0:
        return
    particle_type = p_types[idx]
    if particle_type == 0 or particle_type >= MAX_P_TYPES:
        return
    # print("Particle ", idx, rect_idx, rect[0], rect[1], rect[2], rect[3])
    cuda.atomic.add(avg_v, (rect_idx, 0), velocities[idx, 0]*1e6)
    cuda.atomic.add(avg_v, (rect_idx, 1), velocities[idx, 1]*1e6)
    cuda.atomic.add(avg_v, (rect_idx, 2), 1)