import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

import simba_ps.physics_kernels
from simba_ps.physics_kernels import update_position, update_v, \
    clear_3d_field, \
    clear_forces, \
    limit_component_value, \
    compute_object_field_exch, \
    compute_forces
from simba_ps.utils.util_kernels import get_bpg, place_particles_from_img


class Simba:

    @staticmethod
    def set_symmetric_potential_param(param, value, type_a, type_b=None):
        param_matrix = np.array(vars(simba_ps.physics_kernels)[param])
        old_size = param_matrix.shape[0]
        if type_b is None:
            type_b = type_a
        if max(type_a, type_b) >= old_size:
            new_size = max(type_a, type_b)
            new_param_matrix = np.ones((new_size, new_size))
            new_param_matrix[:old_size, :old_size] = param_matrix
            param_matrix = new_param_matrix
            vars(simba_ps.physics_kernels)['MAX_P_TYPES'] = new_size

            old_masses = np.array(vars(simba_ps.physics_kernels)['_MASSES'])
            if old_masses.shape[0] < new_size:
                masses = np.ones(new_size)
                masses[:old_size] = old_masses
                vars(simba_ps.physics_kernels)['_MASSES'] = tuple(masses)

        param_matrix[type_a, type_b] = value
        param_matrix[type_b, type_a] = value
        vars(simba_ps.physics_kernels)[param] = tuple(map(tuple, param_matrix))

    @staticmethod
    def get_mass(p_type):
        return vars(simba_ps.physics_kernels)['_MASSES'][p_type]
    @staticmethod
    def set_mass(mass, p_type):
        masses = np.array(vars(simba_ps.physics_kernels)['_MASSES'])
        masses[p_type] = mass
        vars(simba_ps.physics_kernels)['_MASSES'] = tuple(masses)

    @staticmethod
    def set_eps(eps, type_a, type_b=None):
        Simba.set_symmetric_potential_param('_EPS', eps, type_a, type_b)

    @staticmethod
    def set_r_min(r_min, type_a, type_b=None):
        sigma = r_min / (2 ** (1.0 / 6))
        Simba.set_symmetric_potential_param('_SIGMA', sigma, type_a, type_b)

    @staticmethod
    def set_potential_params(masses, r_min, eps):
        vars(simba_ps.physics_kernels)['_SIGMA'] = r_min
        vars(simba_ps.physics_kernels)['_EPS'] = eps
        vars(simba_ps.physics_kernels)['_MASSES'] = masses
        vars(simba_ps.physics_kernels)['MAX_P_TYPES'] = len(masses)

    @staticmethod
    def set_default_potential_parameters(p_types_count=1):
        if p_types_count < 1:
            p_types_count = 1
        masses = tuple(np.ones(p_types_count))
        r_min = tuple(map(tuple, np.ones((p_types_count, p_types_count))))
        eps = tuple(map(tuple, np.ones((p_types_count, p_types_count))))
        Simba.set_potential_params(masses, r_min, eps)

    @classmethod
    def from_npz_file(cls, filename: str):
        state = np.load(filename)

        width = state['width']
        height = state['height']

        host_positions = state['positions'].copy()
        host_velocities = state['velocities'].copy()
        host_accelerations = state['accelerations'].copy()
        host_external_forces = state['external_forces'].copy()
        host_types = state['particle_types'].copy()

        sim = cls(width,
                  height,
                  host_positions,
                  host_velocities,
                  host_accelerations,
                  host_external_forces,
                  host_types)

        masses = state['masses']
        r_min = state['r_min']
        eps = state['eps']
        sim.set_potential_params(masses, r_min, eps)

        return sim

    @classmethod
    def from_image(cls, img_data, particle_params=None):
        MAX_PARTICLES = 1024 * 256

        if type(img_data) is not np.ndarray:
            img_data = np.array(img_data)

        img_data = np.transpose(img_data, (1, 0, 2))
        empty_color = (255, 255, 255)

        pixels = img_data.reshape(-1, 3)
        unique_colors, color_counts = np.unique(pixels, axis=0, return_counts=True)
        color_count_dict = {tuple(color): count for color, count in zip(unique_colors, color_counts)}

        if particle_params is None:
            particle_params = {(0, 0, 0): {'m': 0}}

        if empty_color in particle_params:
            print(
                f"Color {empty_color} is reserved for empty space, {color_count_dict.setdefault(empty_color, 0)} pixels ignored")
            del color_count_dict[empty_color]

        tmp_dict = dict()
        for color, count in color_count_dict.items():
            if color not in particle_params and color != empty_color:
                if count > 0:
                    print(f"Color {color} is not in particle_params dict, {count} pixels ignored")
                continue
            tmp_dict[color] = count
        color_count_dict = tmp_dict

        total_particle_count = np.sum(list(color_count_dict.values()))
        total_particle_count = min(MAX_PARTICLES, total_particle_count)

        sim = cls.create_empty(img_data.shape[0], img_data.shape[1], total_particle_count)

        particle_count = np.zeros(1, dtype=int)

        rng_states = create_xoroshiro128p_states(img_data.shape[0] * img_data.shape[1], seed=1)
        threads_per_block = (16, 16, 3)
        blocks_per_grid = get_bpg(threads_per_block, img_data.shape[0], img_data.shape[1], 3)

        p_type = 0

        potential_params_list = []
        for color, params in particle_params.items():
            count = color_count_dict.setdefault(color, 0)
            if count == 0:
                continue

            v0 = params.setdefault('v', 0.0)
            d = params.setdefault('d', 1.0)
            m = params.setdefault('m', 1.0)
            r = params.setdefault('r', 1.0)
            eps = params.setdefault('eps', 1.0)

            potential_params_list.append((m, r, eps))
            place_particles_from_img[blocks_per_grid, threads_per_block](cuda.to_device(np.ascontiguousarray(img_data)),
                                                                         p_type, cuda.to_device(color), d, v0,
                                                                         sim.p_types, sim.positions, sim.velocities,
                                                                         rng_states,
                                                                         particle_count)
            p_type += 1

        sim.particle_count = min(MAX_PARTICLES, particle_count[0])
        sim.positions = sim.positions[:sim.particle_count, :]
        sim.velocities = sim.velocities[:sim.particle_count, :]
        sim.accelerations = sim.accelerations[:sim.particle_count, :]
        sim.forces = sim.forces[:sim.particle_count, :]

        print(f"Placed {sim.particle_count} particles")

        sim.set_default_potential_parameters(p_type)
        for i, potential_params in enumerate(potential_params_list):
            sim.set_mass(potential_params[0], i)
            sim.set_r_min(potential_params[1], i)
            sim.set_eps(potential_params[2], i)

        sim.p_types_count = np.max(p_type)
        return sim

    @classmethod
    def create_empty(cls, width, height, particle_count):
        host_positions = np.zeros((particle_count, 2))

        phi = np.random.random(particle_count) * np.pi * 2
        host_velocities = np.stack([np.cos(phi), np.sin(phi)], axis=1)
        host_accelerations = np.zeros_like(host_velocities)
        host_external_forces = np.zeros((width, height, 2))
        host_types = -1 * np.ones((particle_count), dtype=int)

        sim = cls(width,
                  height,
                  host_positions,
                  host_velocities,
                  host_accelerations,
                  host_external_forces,
                  host_types)
        # sim.state_dir = os.path.dirname(filename)
        return sim

    def __init__(self, width, height,
                 host_positions,
                 host_velocities,
                 host_accelerations,
                 host_external_forces,
                 host_types):

        Simba.set_default_potential_parameters()
        self.deterministic = False

        self.width = width
        self.height = height
        self._time_step = 1.0 / 50.0
        self.time = 0.0

        self.particle_count = host_positions.shape[0]
        self.positions = cuda.to_device(host_positions.astype(np.float64))

        self.velocities = cuda.to_device(host_velocities.astype(np.float32))
        self.accelerations = cuda.to_device(host_accelerations)
        self.forces = cuda.device_array((self.particle_count, 2),
                                        dtype='double')

        self.external_forces = cuda.to_device(host_external_forces.astype(np.float32))

        self.p_types = cuda.to_device(host_types)
        self.p_types_count = np.max(host_types)+1
        # I prefer regarding hash grid as an "index field" enforcing interaction locality
        self.idx_field = cuda.to_device(np.zeros((self.width, self.height, 16), dtype=int))

    @property
    def time_step(self):
        if self._time_step <= 0:
            return self.compute_time_step()
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        self._time_step = value

    def compute_time_step(self):
        raise NotImplemented()

    def update_position(self, dt):
        threads_per_block = (16, 2)
        blocks_per_grid = simba_ps.utils.util_kernels.get_bpg(threads_per_block, self.particle_count, 2)
        update_position[blocks_per_grid, threads_per_block](dt, self.positions,
                                                            self.velocities,
                                                            self.accelerations,
                                                            self.p_types)
        cuda.synchronize()

    def compute_idx_field(self):
        compute_object_field_exch[256, get_bpg((256,), self.particle_count)](self.positions, self.p_types,
                                                                             self.idx_field)
        cuda.synchronize()

    def compute_forces(self):
        self.compute_idx_field()
        threads_per_block = (simba_ps.physics_kernels.DIM, simba_ps.physics_kernels.ACTION_AREA)
        blocks_per_grid = get_bpg(threads_per_block, self.particle_count, simba_ps.physics_kernels.ACTION_AREA)
        compute_forces[blocks_per_grid, threads_per_block](self.positions, self.p_types, self.idx_field,
                                                           self.external_forces,
                                                           self.forces)
        cuda.synchronize()

        if self.deterministic:
            threads_per_block = (16,)
            blocks_per_grid = simba_ps.utils.util_kernels.get_bpg(threads_per_block, self.particle_count)
            simba_ps.utils.util_kernels.apply_round[blocks_per_grid, threads_per_block](self.forces)
            cuda.synchronize()

    def update_acceleration_and_velocities(self, dt):
        threads_per_block = (16, 2)
        blocks_per_grid = simba_ps.utils.util_kernels.get_bpg(threads_per_block, self.particle_count, 2)
        update_v[blocks_per_grid, threads_per_block](dt, self.p_types, self.velocities,
                                                     self.accelerations, self.forces)
        cuda.synchronize()

    def clear_idx_field(self, value=-1):
        threads_per_block = (8, 8, 16)
        blocks_per_grid = get_bpg(threads_per_block, self.idx_field.shape[0], self.idx_field.shape[1],
                                  self.idx_field.shape[2])
        clear_3d_field[blocks_per_grid, threads_per_block](self.idx_field, value)

    def clear_forces(self):
        threads_per_block = (32, 2)
        blocks_per_grid = get_bpg(threads_per_block, self.particle_count, 2)
        clear_forces[blocks_per_grid, threads_per_block](self.forces)

    def clear_accumulated_data(self):
        self.clear_forces()
        self.clear_idx_field()
        cuda.synchronize()

    def verelet(self, dt=0.0):
        self.on_verelet_start(dt)

        self.clear_accumulated_data()

        self.update_position(dt)
        self.on_position_updated(dt)

        self.compute_forces()
        self.on_forces_computed(dt)

        self.update_acceleration_and_velocities(dt)

        self.on_verelet_step_complete(dt)
        self.time += dt

    def save_single_state(self, filename):

        np.savez_compressed(filename,
                            width=self.width,
                            height=self.height,
                            positions=self.positions.copy_to_host(),
                            velocities=self.velocities.copy_to_host(),
                            accelerations=self.accelerations.copy_to_host(),
                            external_forces=self.external_forces.copy_to_host(),
                            particle_types=self.p_types.copy_to_host(),

                            masses=np.array(vars(simba_ps.physics_kernels)['_MASSES']),
                            r_min=vars(simba_ps.physics_kernels)['_SIGMA'],
                            eps=vars(simba_ps.physics_kernels)['_EPS'])

    def on_position_updated(self, dt):
        pass

    def on_forces_computed(self, dt):
        pass

    def on_verelet_step_complete(self, dt):
        pass

    def on_computation_finished(self):
        pass

    def component_limit(self, parameter, limit):
        threads_per_block = (16, parameter.shape[1])
        blocks_per_grid = simba_ps.utils.util_kernels.get_bpg(threads_per_block, parameter.shape[0], parameter.shape[1])
        limit_component_value[blocks_per_grid, threads_per_block](
            self.p_types,
            parameter, limit)

        if self.deterministic:
            threads_per_block = (16,)
            blocks_per_grid = simba_ps.utils.util_kernels.get_bpg(threads_per_block, self.particle_count)
            simba_ps.utils.util_kernels.apply_round[blocks_per_grid, threads_per_block](parameter)

    def copy(self):
        pass

    def on_verelet_start(self, dt):
        pass
