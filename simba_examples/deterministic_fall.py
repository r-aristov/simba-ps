import os.path
import sys

import numpy as np
from PIL import Image
from numba import cuda
from simba_ps.thermostates.simple_deterministic_thermostate import SDThermostate

from simba_ps import Simba
from simba_ps.utils.pygame_app import PyGameSimApp
from simba_ps.visualization.camera import Camera
from simba_ps.utils.helpers import load_img_to_gpu, set_colors_in_rect

colors_fname = "simba_examples/outputs/final_colors.npy"
initial_state_fname = "simba_examples/outputs/initial_state.npz"
fingus_fname = "simba_examples/inputs/fingus.bmp"
winkle_fname = "simba_examples/inputs/winkle.bmp"

class DeterministicFallApp(PyGameSimApp):
    def __init__(self, first_pass=True):
        super().__init__()
        self.therm = None
        self.temperature = 30.0
        self.first_pass = first_pass
        self.save_state_at = 100

    def on_start(self, sim, cam):
        if self.first_pass:
            fingus = load_img_to_gpu(fingus_fname)
            winkle = load_img_to_gpu(winkle_fname)
            set_colors_in_rect(cam.colors, fingus, sim.p_types, sim.positions, 194, 145, [1])
            set_colors_in_rect(cam.colors, winkle, sim.p_types, sim.positions, 60, 194, [1])
        else:
            cam.colors = cuda.to_device(np.load(colors_fname))

        self.therm = SDThermostate(sim)
        self.therm.define_temperature_rects([
            (0, 0, sim.width, sim.height)
        ])

    def on_computation_finished(self, frame, sim, cam):
        self.therm.compute_temperature_in_rects()
        self.therm.set_temperature_in_rect(np.array([self.temperature]))
        if sim.time > 50.0 and self.temperature > 1:
            self.temperature -= 1.0 * sim.time_step

        if sim.time >= self.save_state_at and self.first_pass:
            fingus = load_img_to_gpu(fingus_fname)
            winkle = load_img_to_gpu(winkle_fname)
            set_colors_in_rect(cam.colors, fingus, sim.p_types, sim.positions, 276, 520, [1])
            set_colors_in_rect(cam.colors, winkle, sim.p_types, sim.positions, 93, 478, [1])

            np.save(colors_fname, cam.colors.copy_to_host())
            print(f"Final positions used to assign colors, saved to '{colors_fname}'")
            sys.exit()


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'Deterministic Fall' example",
                                         default_substeps=25,
                                         default_sim_len=590,
                                         default_scale=1.5)
    parser.add_argument('--first_pass', type=bool, default=False, help="Set to True to compute color mapping, "
                                                                       "set to False for fallin particles to form picture")

    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/det_fall.bmp")

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (255, 127, 39): {'v': 2.1, 'r': 1.09, 'eps': 5.0, 'm': 0.5},
    })

    first_pass = args.first_pass
    if not os.path.exists(initial_state_fname) or not os.path.exists(colors_fname):
        print(f"--first_pass=False is set, but '{initial_state_fname}' or '{colors_fname}' is missing, "
              f"running with first_pass=True, run simulation for 100 seconds to compute necessary data.")
        first_pass = True

    if first_pass:
        print(f"Initial state saved to '{initial_state_fname}'")
        sim.save_single_state(initial_state_fname)

    sim = Simba.from_npz_file(initial_state_fname)
    sim.external_forces[:, :, 1] = 5.0
    sim.time_step = 1.0 / 50

    sim.deterministic = True

    scale = args.scale
    camera = Camera(sim, sim.width * scale, sim.height * scale)
    camera.scale = scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))

    camera.colors_from_types({0: (1.0, 1.0, 1.0), 1: (0.48, 0.25, 0.78)})
    render_options = vars(args).copy()
    render_options['render_mode'] = 'colors'

    DeterministicFallApp(first_pass=first_pass).run(sim, camera, seconds=args.sim_len, substeps=args.substeps,
                               render_options=render_options)



if __name__ == '__main__':
    main()
