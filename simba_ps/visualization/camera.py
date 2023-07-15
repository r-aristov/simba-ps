import math

import numba
import numpy as np

from numba import cuda
import simba_ps
from simba_ps.utils.util_kernels import get_bpg
from simba_ps import Simba
from simba_ps.visualization.visualization_kernels import set_color_from_p_type_kernel, set_colors_from_vector_magnitude_kernel, \
    render_particles_scaled_colored, fade_3d_field, \
    render_particles_smooth, render_densities, copy_buffer


class Camera:
    def __init__(self, sim: Simba, width=256.0, height=256.0, scale=1.0):

        self.particle_size = -np.ones(np.max(sim.p_types) + 1, dtype=float)
        self.particle_grid_size = None
        self.r = None

        self.sim = sim
        self.colors = cuda.to_device(np.ones((self.sim.particle_count, 3)))

        self.colors_from_types({i: np.random.random(3) for i in range(sim.p_types_count)})

        self.transformed_coords = None
        self.to_render = None
        self.object_count = cuda.to_device(np.array([0]))

        self.scale = scale
        self._width = int(width)
        self._height = int(height)
        self._init_buffers()

        self.view_point = np.array([0.0, 0.0])

    def colors_from_types(self, type_dict: dict):
        threads_per_block = (256, 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.sim.particle_count,
                                  3)

        for p_type, color in type_dict.items():
            set_color_from_p_type_kernel[blocks_per_grid, threads_per_block](self.sim.p_types,
                                                                             self.colors,
                                                                             p_type,
                                                                             cuda.to_device(color))
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = int(value)
        self._init_buffers()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = int(value)
        self._init_buffers()

    def _init_buffers(self):
        self.screen_buffer = cuda.device_array((self.width, self.height, 3), dtype="uint8")
        self.accumulating_buffer = cuda.device_array((self.width, self.height, 3), dtype="uint32")
        self.densities = cuda.device_array((self.width, self.height, self.sim.p_types_count), dtype="float")

        self.screen_buffer[:] = 0


    def set_colors_from_velocity(self, gradient_colors, v_min, v_max):
        threads_per_block = (32, 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.sim.particle_count,
                                  3)
        set_colors_from_vector_magnitude_kernel[blocks_per_grid, threads_per_block](self.sim.p_types,
                                                                                    self.sim.velocities,
                                                                                    gradient_colors, self.colors,
                                                                                    v_min, v_max)

    def color_render(self, particle_size=1.1, background_retention=0.8):
        threads_per_block = (16, 16, 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.width,
                                  self.height,
                                  3)

        fade_3d_field[blocks_per_grid, threads_per_block](self.screen_buffer, background_retention)

        particle_size_scaled = particle_size * self.scale
        particle_grid_size = np.ceil(particle_size_scaled)
        if particle_grid_size % 2 == 0:
            particle_grid_size += 1
        self.render_particles_colored(self.sim.particle_count, self.sim.positions, particle_size_scaled,
                                      particle_grid_size, self.colors)
        numba.cuda.synchronize()

    def render_particles_colored(self, particle_count, positions, particle_size_scaled, grid_size, colors):
        threads_per_block = (32, 32)
        blocks_per_grid = get_bpg(threads_per_block,
                                  particle_count,
                                  grid_size * grid_size)

        r2 = math.ceil((particle_size_scaled - 1) * (particle_size_scaled - 1) * 0.25)
        render_particles_scaled_colored[blocks_per_grid, threads_per_block](positions,
                                                                            self.view_point,
                                                                            self.scale,
                                                                            r2,
                                                                            grid_size,
                                                                            self.screen_buffer,
                                                                            colors)

    def render_particles_smooth(self, particle_count, p_types, positions, r, grid_size):
        threads_per_block = (32, 32)
        blocks_per_grid = get_bpg(threads_per_block,
                                  particle_count,
                                  grid_size * grid_size)

        render_particles_smooth[blocks_per_grid, threads_per_block](p_types,
                                                                    positions,
                                                                    self.view_point,
                                                                    self.scale,
                                                                    r,
                                                                    grid_size,
                                                                    self.densities)

    def compute_particle_grid_params_scaled(self):
        particle_size_scaled = self.particle_size * self.scale
        particle_grid_size = np.max(np.ceil(particle_size_scaled))
        if particle_grid_size % 2 == 0:
            particle_grid_size += 1
        r = np.sqrt(np.ceil((particle_size_scaled - 1) * (particle_size_scaled - 1) * 0.25))
        self.r = cuda.to_device(r)
        self.particle_grid_size = particle_grid_size

    def smooth_render(self, colors, mag_min, mag_max, falloff_radius=2.5):
        threads_per_block = (16, 16, 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.width, self.height, 3)
        simba_ps.physics_kernels.clear_3d_field[blocks_per_grid, threads_per_block](self.accumulating_buffer, 0)

        blocks_per_grid = get_bpg(threads_per_block,
                                  self.width, self.height, self.densities.shape[2])

        simba_ps.physics_kernels.clear_3d_field[blocks_per_grid, threads_per_block](self.densities, 0)
        numba.cuda.synchronize()

        if self.r is None:
            new_sizes = np.ones(np.max(self.sim.p_types) + 1, dtype=float) * falloff_radius
            if self.particle_size is None:
                self.particle_size = new_sizes
            else:
                mask = self.particle_size < 0
                self.particle_size = ~mask*self.particle_size + mask * new_sizes

            self.compute_particle_grid_params_scaled()

        self.render_particles_smooth(self.sim.particle_count, self.sim.p_types, self.sim.positions,
                                     self.r,
                                     self.particle_grid_size)

        threads_per_block = (16, self.densities.shape[2], 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.width * self.height,
                                  self.densities.shape[2],
                                  3)
        render_densities[blocks_per_grid, threads_per_block](self.densities, colors, self.accumulating_buffer, mag_min,
                                                             mag_max)

        threads_per_block = (16, 16, 3)
        blocks_per_grid = get_bpg(threads_per_block,
                                  self.width, self.height, 3)
        copy_buffer[blocks_per_grid, threads_per_block](self.accumulating_buffer, self.screen_buffer)
        numba.cuda.synchronize()
