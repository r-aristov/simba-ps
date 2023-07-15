import numpy as np
from PIL import Image
from numba import cuda

from simba_ps import Simba

from simba_ps.thermostates.simple_deterministic_thermostate import SDThermostate
from simba_ps.utils.pygame_app import PyGameSimApp

from simba_ps.visualization.camera import Camera
from simba_ps.utils.helpers import load_img_to_gpu, set_colors_in_rect, select_particle


class Chem201App(PyGameSimApp):
    def __init__(self):
        super().__init__()
        self.therm = None
        self.target_temperature = 0.01
        self.current_temperature = self.target_temperature
        self.cooling = False
        self.initial_kick_finished = False

    def caption(self, sim, cam):
        return f"Simba - particles: {sim.particle_count}," \
               f" sim time: {sim.time:1.2f}," \
               f"fps: {self.fps_timer.get_fps():1.2f}, temp: {self.current_temperature:1.2f}, "

    def on_start(self, sim, cam):
        set_colors_in_rect(cam.colors,
                           load_img_to_gpu("simba_examples/inputs/texture_blue.jpg"),
                           sim.p_types, sim.positions,
                           147, 67, [1])

        set_colors_in_rect(cam.colors, load_img_to_gpu("simba_examples/inputs/texture_red.jpg"),
                           sim.p_types, sim.positions,
                           147, 67,
                           [2])

        set_colors_in_rect(cam.colors, load_img_to_gpu("simba_examples/inputs/texture_yellow.jpg"),
                           sim.p_types, sim.positions,
                           147, 67,
                           [3])

        set_colors_in_rect(cam.colors, load_img_to_gpu("simba_examples/inputs/texture_green.jpg"),
                           sim.p_types, sim.positions,
                           147, 67,
                           [4])

        self.therm = SDThermostate(sim)
        self.therm.define_temperature_rects([
            (0, 0, sim.width, sim.height)
        ])

    def on_computation_finished(self, frame, sim, cam):
        if sim.time >= 0.75 and not self.initial_kick_finished:
            initial_kick_finished = True
            idx = select_particle(sim.positions, 232, 158)
            if idx > 0:
                sim.velocities[idx, 0] = 10.0

        self.therm.compute_temperature_in_rects()
        self.current_temperature = self.therm.temperatures[0] * 1e-6

        if not self.cooling and self.current_temperature > 450:
            self.cooling = True
            self.target_temperature = self.current_temperature

        if self.cooling and self.target_temperature > 50:
            self.target_temperature -= 10.0 * sim.time_step

        if sim.time < 1.00 or self.cooling:
            self.therm.set_temperature_in_rect(np.array([self.target_temperature]))


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'Chemistry 201' example",
                                         default_substeps=100,
                                         default_scale=5,
                                         default_sim_len=600,
                                         default_render_mode='color')


    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/chem_1.bmp")

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (237, 28, 36): {'v': 0.1, 'eps': 1.0},
        (255, 127, 39): {'v': 0.1, 'eps': 1.0},
        (255, 242, 0): {'v': 0.1, 'eps': 1.0},
        (34, 177, 76): {'v': 0.1, 'eps': 1.0},
    })

    # ------- chem param -------
    sim.set_eps(500, 1, 2)
    sim.set_eps(1.0, 1)
    sim.set_eps(0.5, 2)
    sim.set_r_min(0.7, 1, 2)
    sim.set_r_min(1.5, 2)
    sim.set_r_min(3.0, 1)

    sim.set_eps(500, 3, 4)
    sim.set_eps(1.0, 3)
    sim.set_eps(0.5, 4)
    sim.set_r_min(0.7, 3, 4)
    sim.set_r_min(1.5, 4)
    sim.set_r_min(3.0, 3)

    sim.set_r_min(0.7, 1, 3)
    sim.set_r_min(2, 1, 4)

    sim.set_eps(12000, 1, 3)
    # ------- -------

    sim.time_step = 1.0 / 50

    camera = Camera(sim, sim.width * args.scale, sim.height * args.scale)
    camera.scale = args.scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))
    camera.colors_from_types({0: (1.0, 1.0, 1.0)})

    Chem201App().run(sim, camera, seconds=args.sim_len, substeps=args.substeps,
                     render_options=vars(args).copy())


if __name__ == '__main__':
    main()
