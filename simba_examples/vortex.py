import numpy as np
from PIL import Image
from numba import cuda
from simba_ps.utils.pygame_app import PyGameSimApp, CustomRange

from simba_ps import Simba
from simba_ps.visualization.camera import Camera


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'vortex' example",
                                         default_sim_len=120,
                                         default_substeps=20,
                                         default_v_max=25,
                                         default_scale=5.0)

    parser.add_argument('--density', type=float, default=1.0, choices=CustomRange(0.01, 1.0),
                        help="Particle density (0.01 to 1.0)")
    parser.add_argument('--eps', type=float, default=1.0, choices=CustomRange(0.5, 30.0),
                        help="Depth of Lennard-Jones potential well (0.5 to 30.0)")

    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/test_256_256_full.bmp")

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (237, 28, 36): {'v': 0.1, 'eps': args.eps, 'd': args.density},
    })

    sim.time_step = 1.0 / 50
    sim.external_forces[int(sim.width * 0.5):, :, 1] = 1.0
    sim.external_forces[:int(sim.width * 0.5), :, 1] = -1.0

    camera = Camera(sim, sim.width * args.scale, sim.height * args.scale)
    camera.scale = args.scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))
    camera.colors_from_types({0: (1.0, 1.0, 1.0)})
    PyGameSimApp().run(sim, camera, seconds=args.sim_len, substeps=args.substeps,
                       render_options=vars(args).copy())


if __name__ == '__main__':
    main()
