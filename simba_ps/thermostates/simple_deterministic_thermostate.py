import numpy as np
from numba import cuda

from simba_ps import Simba
from simba_ps.thermostates.sdt_kernels import clear_thermostate_data, compute_avg_v_in_rects, det_compute_temperature_in_rects, \
    set_temperature_in_rect
from simba_ps.utils.util_kernels import get_bpg


class SDThermostate:
    def __init__(self, sim: Simba):
        self.sim: Simba = sim

        rects = 1
        self.temperatures = cuda.to_device(np.zeros(rects, dtype='int64'))
        self.temperature_rects = np.zeros((rects, 4))
        self.temperature_rects[0, :] = np.array([0, 0, self.sim.width, self.sim.height])
        self.temperature_rects = cuda.to_device(self.temperature_rects)
        self.target_temperatures = cuda.to_device(np.array([-1.0], dtype='float64'))

        self.avg_v = cuda.to_device(np.zeros((rects, 3), dtype='int64'))

    def compute_temperature_in_rects(self):
        rects_count = self.temperature_rects.shape[0]

        threads_per_block = (min(rects_count, 32),)
        blocks_per_grid = get_bpg(threads_per_block, rects_count)
        clear_thermostate_data[blocks_per_grid, threads_per_block](self.avg_v,
                                                                   self.temperatures)

        threads_per_block = (32, min(rects_count, 32))
        blocks_per_grid = get_bpg(threads_per_block, self.sim.particle_count, rects_count)
        compute_avg_v_in_rects[blocks_per_grid, threads_per_block](self.sim.p_types,
                                                                   self.sim.positions,
                                                                   self.sim.velocities,
                                                                   self.temperature_rects,

                                                                   self.avg_v)

        threads_per_block = (32, min(rects_count, 32))
        blocks_per_grid = get_bpg(threads_per_block, self.sim.particle_count, rects_count)
        det_compute_temperature_in_rects[blocks_per_grid, threads_per_block](self.sim.p_types,
                                                                             self.sim.positions,
                                                                             self.sim.velocities,
                                                                             self.temperature_rects,
                                                                             self.avg_v,

                                                                             self.temperatures)

    def set_temperature_in_rect(self, target_temperatures):
        rects_count = self.temperature_rects.shape[0]
        threads_per_block = (32, min(rects_count, 32))
        blocks_per_grid = get_bpg(threads_per_block, self.sim.particle_count, rects_count)
        set_temperature_in_rect[blocks_per_grid, threads_per_block](self.sim.p_types,
                                                                    self.sim.positions,
                                                                    self.sim.velocities,
                                                                    self.temperature_rects,
                                                                    self.temperatures,
                                                                    self.avg_v,
                                                                    target_temperatures)

    def define_temperature_rects(self, rects):
        self.temperature_rects = cuda.to_device(rects)
        self.temperatures = cuda.to_device(np.zeros(len(rects), dtype='int64'))
        targets = -np.ones(len(rects))
        self.target_temperatures = cuda.to_device(targets)
        self.avg_v = cuda.to_device(np.zeros((len(rects), 3), dtype='int64'))
