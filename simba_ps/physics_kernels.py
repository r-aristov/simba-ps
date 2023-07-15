import numba.cuda
from numba import cuda, float32

from simba_ps.utils.util_kernels import grid_point_outside_field

ACTION_RANGE = 5
ACTION_AREA = ACTION_RANGE * ACTION_RANGE
DIM = int(1024 / ACTION_AREA)

MAX_P_TYPES = 3

_MASSES = (0.0, 1.0, 1.0)

_SIGMA = ((1.0, 1.0, 1.0),
          (1.0, 1.0, 1.0),
          (1.0, 1.0, 1.0))

_EPS = ((1.0, 1.0, 1.0),
        (1.0, 1.0, 0.1),
        (1.0, 0.1, 0.1))


@cuda.jit(device=True, inline=True)
def mass_from_type(p_type):
    return _MASSES[p_type]


@cuda.jit(device=True, inline=True)
def potential_params_from_type(p_type0: int, p_type1: int):
    return _SIGMA[p_type0][p_type1], _EPS[p_type0][p_type1]


@cuda.jit
def update_position(dt, positions, velocities, accelerations, p_types):
    idx, c_idx = cuda.grid(2)
    if idx >= positions.shape[0]:
        return

    p_type_cache = cuda.shared.array(16, 'int16')
    if c_idx == 0:
        p_type_cache[cuda.threadIdx.x] = p_types[idx]
    cuda.syncthreads()
    p_type = p_type_cache[cuda.threadIdx.x]

    if p_type < 0 or p_type >= MAX_P_TYPES or mass_from_type(p_type) <= 0:
        return

    v = velocities[idx, c_idx]
    a = accelerations[idx, c_idx]
    d_pos = v * dt + (0.5 * dt ** 2) * a
    positions[idx, c_idx] = positions[idx, c_idx] + d_pos


@cuda.jit
def clear_forces(forces):
    idx, c_idx = cuda.grid(2)
    if idx >= forces.shape[0]:
        return
    forces[idx, c_idx] = 0.0


@cuda.jit
def update_v(dt, p_types, velocities, accelerations, forces):
    idx, c_idx = cuda.grid(2)
    if idx >= velocities.shape[0]:
        return
    p_type_cache = cuda.shared.array(16, 'int16')
    if c_idx == 0:
        p_type_cache[cuda.threadIdx.x] = p_types[idx]
    cuda.syncthreads()
    p_type = p_type_cache[cuda.threadIdx.x]

    if p_type < 0:
        return

    m = mass_from_type(p_type)

    if m == 0:
        return

    v = velocities[idx, c_idx]
    a = accelerations[idx, c_idx]

    new_a = forces[idx, c_idx] / m
    new_v = v + dt * 0.5 * (a + new_a)

    accelerations[idx, c_idx] = new_a
    velocities[idx, c_idx] = new_v


@cuda.jit(inline=True)
def clamp(value, min_limit, max_limit):
    if value > max_limit:
        value = max_limit
    if value < min_limit:
        value = min_limit


@cuda.jit
def clear_3d_field(field, value):
    x, y, z = cuda.grid(3)
    if x >= field.shape[0] or y >= field.shape[1] or z >= field.shape[2]:
        return
    field[x, y, z] = value


@cuda.jit
def clear_2d_field(field, value):
    x, y = cuda.grid(2)
    if x >= field.shape[0] or y >= field.shape[1]:
        return
    field[x, y] = value


@cuda.jit(device=True, inline=True)
def lennard_jones_force_from_r2(r2, sigma, eps):
    inv_r2 = 1.0 / r2
    sigma_over_r2 = sigma * sigma * inv_r2
    sigma_over_r6 = sigma_over_r2 * sigma_over_r2 * sigma_over_r2
    sigma_over_r12 = sigma_over_r6 * sigma_over_r6
    mag = 24 * eps * inv_r2 * (2 * sigma_over_r12 - sigma_over_r6)  # - a
    potential = 4 * eps * (sigma_over_r12 - sigma_over_r6)  # - a*r - b
    return mag, potential

@cuda.jit
def compute_object_field_exch(positions, p_types, field):
    idx = cuda.grid(1)

    if idx >= positions.shape[0]:
        return
    p_type = p_types[idx]
    if p_type < 0 or p_type >= MAX_P_TYPES:
        return
    grid_x = int(positions[idx, 0])
    grid_y = int(positions[idx, 1])

    if grid_point_outside_field(grid_x, grid_y, field):
        return

    field_idx = idx  # self field idx
    for idx_field_z in range(field.shape[2]):
        field_idx = numba.cuda.atomic.exch(field, (grid_x, grid_y, idx_field_z), field_idx)
        if field_idx < 0:
            return
    print("Cohesion lost ", idx)


@numba.cuda.jit(device=True, inline=True)
def idx_to_grid_2d(index, size):
    x = index % size
    y = index // size
    return x - int(0.5 * size), y - int(0.5 * size)


@numba.cuda.jit(device=True, inline=True)
def grid_2d_to_idx(x, y, size):
    x += int(0.5 * size)
    y += int(0.5 * size)
    index = y * size + x
    return index


@cuda.jit
def compute_forces(positions,
                   p_types,
                   field,
                   external_positional_forces,

                   forces):
    idx, field_idx = cuda.grid(2)
    if idx >= positions.shape[0]:
        return

    self_pos = cuda.shared.array((DIM, 2), 'float64')
    self_p_type_cache = cuda.shared.array(DIM, 'int32')
    total_force = cuda.shared.array((DIM, ACTION_AREA, 2), 'float64')

    if cuda.threadIdx.y < 2:
        self_pos[cuda.threadIdx.x, cuda.threadIdx.y] = positions[idx, field_idx]
    elif cuda.threadIdx.y == 2:
        self_p_type = p_types[idx]
        self_p_type_cache[cuda.threadIdx.x] = self_p_type

    total_force[cuda.threadIdx.x, cuda.threadIdx.y, 0] = 0.0
    total_force[cuda.threadIdx.x, cuda.threadIdx.y, 1] = 0.0
    cuda.syncthreads()

    grid_offset = idx_to_grid_2d(field_idx, ACTION_RANGE)
    grid_pos = (int(self_pos[cuda.threadIdx.x, 0]) + grid_offset[0],
                int(self_pos[cuda.threadIdx.x, 1]) + grid_offset[1])
    grid_x = grid_pos[0]
    grid_y = grid_pos[1]

    self_p_type = self_p_type_cache[cuda.threadIdx.x]
    if mass_from_type(self_p_type) <= 0:
        return

    if not grid_point_outside_field(grid_x, grid_y, field):
        compute_forces_from_field_point(field, grid_x, grid_y,
                                        idx, self_p_type, self_pos,
                                        p_types, positions,
                                        external_positional_forces, total_force)
        
    reduce_forces(idx, total_force, forces)


@cuda.jit(inline=True, device=True)
def reduce_forces(idx, total_force, forces):
    cuda.syncthreads()
    a_2 = int(ACTION_AREA / 2)
    if cuda.threadIdx.y < a_2:
        f_x = total_force[cuda.threadIdx.x, cuda.threadIdx.y, 0] + total_force[
            cuda.threadIdx.x, a_2 + cuda.threadIdx.y, 0]
        cuda.atomic.add(forces, (idx, 0), f_x)
    elif cuda.threadIdx.y == ACTION_AREA - 1:  # for odd areas
        cuda.atomic.add(forces, (idx, 0), total_force[cuda.threadIdx.x, cuda.threadIdx.y, 0])
        cuda.atomic.add(forces, (idx, 1), total_force[cuda.threadIdx.x, cuda.threadIdx.y, 1])
    else:
        f_y = total_force[cuda.threadIdx.x, cuda.threadIdx.y, 1] + total_force[
            cuda.threadIdx.x, cuda.threadIdx.y - a_2, 1]
        cuda.atomic.add(forces, (idx, 1), f_y)


@cuda.jit(inline=True, device=True)
def compute_forces_from_field_point(field, grid_x, grid_y, idx, self_p_type, self_pos, p_types, positions,
                                    external_positional_forces, total_force):
    z_limit = field.shape[2]
    other_idx = -1
    for other_z_idx in range(z_limit):
        other_idx = field[grid_x, grid_y, other_z_idx]
        if other_idx < 0:
            break
        if other_idx == idx:
            f_x = external_positional_forces[grid_x, grid_y, 0]
            f_y = external_positional_forces[grid_x, grid_y, 1]

            total_force[cuda.threadIdx.x, cuda.threadIdx.y, 0] += f_x
            total_force[cuda.threadIdx.x, cuda.threadIdx.y, 1] += f_y
            # interacted = True
            continue

        other_p_type = p_types[other_idx]

        r_vec_x = self_pos[cuda.threadIdx.x, 0] - positions[other_idx, 0]
        r_vec_y = self_pos[cuda.threadIdx.x, 1] - positions[other_idx, 1]

        r2 = float32(r_vec_x * r_vec_x) + float32(r_vec_y * r_vec_y)

        sigma, eps = potential_params_from_type(self_p_type, other_p_type)

        f, p = lennard_jones_force_from_r2(r2, sigma, eps)

        f_x = float32(r_vec_x * f)
        f_y = float32(r_vec_y * f)

        total_force[cuda.threadIdx.x, cuda.threadIdx.y, 0] += f_x
        total_force[cuda.threadIdx.x, cuda.threadIdx.y, 1] += f_y


@cuda.jit
def limit_component_value(p_types, parameter, limit):
    idx, c_idx = cuda.grid(2)
    if idx >= parameter.shape[0]:
        return
    p_type = p_types[idx]
    if p_type < 0 or p_type >= MAX_P_TYPES:
        return
    p = parameter[idx, c_idx]
    if p > limit:
        parameter[idx, c_idx] = limit
        return
    if p < -limit:
        parameter[idx, c_idx] = -limit
