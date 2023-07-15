import numpy as np
from PIL import Image
from numba import cuda
from simba_ps.thermostates.simple_deterministic_thermostate import SDThermostate

from simba_ps import Simba
from simba_ps.utils.pygame_app import PyGameSimApp, CustomRange
from simba_ps.visualization.camera import Camera


class DistillationApp(PyGameSimApp):
    def __init__(self, therm_params, max_t=60.0, heating_speed=1.5):
        super().__init__()
        self.therm = None
        self.heater_temp = 0.1
        self.temperatures = cuda.to_device(np.array([self.heater_temp, 40.0, 20.0, 5.0, 5.0]))
        self.max_temp = max_t
        self.heating_speed = heating_speed
        self.therm_params = therm_params

    def caption(self, sim, cam):
        return f"Simba - particles: {sim.particle_count}," \
               f" sim time: {sim.time:1.2f}," \
               f"fps: {self.fps_timer.get_fps():1.2f}, heater t: {self.heater_temp:1.2f}, "

    def on_start(self, sim, cam):
        self.therm = SDThermostate(sim)
        self.therm.define_temperature_rects(self.therm_params)

    def on_computation_finished(self, frame, sim, cam):
        self.temperatures[0] = self.heater_temp
        self.therm.compute_temperature_in_rects()
        self.therm.set_temperature_in_rect(self.temperatures)
        if self.heater_temp < self.max_temp and 10 < sim.time < 700:
            self.heater_temp += self.heating_speed * sim.time_step
            self.heater_temp = min(self.heater_temp, self.max_temp)


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'Distillation' example", default_substeps=20,
                                         default_sim_len=600,
                                         default_scale=2,
                                         default_render_mode='smooth', default_v_max=8, default_falloff_radius=5)

    parser.add_argument('--alembic_size', type=str, default='big', choices=['big', 'small'],
                        help="Size of simulated chemical equipment, either 'big' (~43k particles) or 'small' (13k "
                             "particles)")
    parser.add_argument('--max_t', type=float, default=60.0, choices=CustomRange(2, 100),
                        metavar='[2.0 - 100.0]', help="Maximum heater temperature (2.0 to 100.0)")
    parser.add_argument('--heating_speed', type=float, default=1.5, choices=CustomRange(1, 100),
                        metavar='[1.0 - 10.0]', help="Heating speed (degrees per sec) (1.0 to 10.0)")

    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/distillation_small.bmp") if args.alembic_size == 'small' \
        else Image.open("simba_examples/inputs/distillation_big.bmp")

    therm_params = [

        (28, 195, 82, 5),
        (127, 114, 106, 43),
        (163, 160, 77, 60),
        (173, 224, 64, 24),
        (180, 336, 97, 9)

    ] if args.alembic_size == 'small' else [

        (335, 244, 64, 14),
        (563, 118, 47, 47),
        (579, 163, 113, 62),
        (612, 240, 67, 20),
        (616, 349, 97, 9)

    ]

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (237, 28, 36): {'v': 0.01, 'eps': 5.0, 'm': 1.0, 'r': 0.95},
        (255, 127, 39): {'v': 0.01, 'eps': 70.0, 'm': 1.0, 'r': 0.95},
    })

    sim.set_eps(2.0, 1, 2)
    sim.set_r_min(1.0, 1, 2)

    sim.external_forces[:, :, 1] = 2.0
    sim.time_step = 1.0 / 50

    scale = args.scale
    camera = Camera(sim, sim.width * scale, sim.height * scale)
    camera.scale = scale
    camera.particle_size = -np.ones(np.max(sim.p_types) + 1, dtype=float)
    camera.particle_size[0] = 2.0

    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))

    render_options = vars(args).copy()

    DistillationApp(therm_params, max_t=args.max_t, heating_speed=args.heating_speed).run(sim, camera,
                                                                                          seconds=args.sim_len,
                                                                                          substeps=args.substeps,
                                                                                          render_options=render_options)


if __name__ == '__main__':
    main()
