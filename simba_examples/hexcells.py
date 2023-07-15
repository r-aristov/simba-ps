import numpy as np
from PIL import Image
from numba import cuda
from simba_ps.utils.pygame_app import PyGameSimApp, CustomRange

from simba_ps import Simba
from simba_ps.visualization.camera import Camera


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'hexcells' example",
                                         default_substeps=12,
                                         default_scale=2,
                                         default_render_mode='smooth',
                                         default_v_max=7,
                                         default_falloff_radius=5)

    parser.add_argument('--eps', type=float, default=10.0, choices=CustomRange(0.5, 30.0),
                        help="Depth of Lennard-Jones potential well (0.5 to 30.0)")
    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/cells_960_540.bmp")
    eps = args.eps
    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (136, 0, 21): {'v': 0.01, 'eps': eps},
        (237, 28, 36): {'v': 0.1, 'eps': eps},
        (255, 127, 39): {'v': 1.0, 'eps': eps},
        (255, 242, 0): {'v': 2.0, 'eps': eps},
        (34, 177, 76): {'v': 3.0, 'eps': eps},
        (0, 162, 232): {'v': 4.0, 'eps': eps},
        (63, 72, 204): {'v': 5.0, 'eps': eps},
        (163, 73, 164): {'v': 7.0, 'eps': eps},
        (255, 174, 201): {'v': 9.0, 'eps': eps},
        (239, 228, 176): {'v': 10.0, 'eps': eps}
    })

    sim.time_step = 1.0 / 50
    sim.external_forces[:, :, 1] = 1.0

    camera = Camera(sim, sim.width * args.scale, sim.height * args.scale)
    camera.scale = args.scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))
    camera.colors_from_types({0: (1.0, 1.0, 1.0)})

    camera.particle_size = -np.ones(np.max(sim.p_types) + 1, dtype=float)
    camera.particle_size[0] = 2.0

    PyGameSimApp().run(sim, camera, seconds=args.sim_len, substeps=args.substeps,
                       render_options=vars(args))


if __name__ == '__main__':
    main()
