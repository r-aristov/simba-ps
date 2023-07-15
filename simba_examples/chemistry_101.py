import numpy as np
from PIL import Image
from numba import cuda
from simba_ps.utils.pygame_app import PyGameSimApp, CustomRange

from simba_ps import Simba
from simba_ps.visualization.camera import Camera
from simba_ps.utils.helpers import load_img_to_gpu, set_colors_in_rect


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'Chemistry 101' example",
                                         default_substeps=20,
                                         default_render_mode='color',
                                         default_scale=5)
    parser.add_argument('--eps_blue_red', type=float, default=25.0, choices=CustomRange(0.5, 200.0),
                        help="Depth of Lennard-Jones potential well between blue and red particles (0.5 to 200.0)")
    parser.add_argument('--eps_blue_orange', type=float, default=100.0, choices=CustomRange(0.5, 200.0),
                        help="Depth of Lennard-Jones potential well between blue and orange particles (0.5 to 200.0)")
    parser.add_argument('--eps_red_orange', type=float, default=1.0, choices=CustomRange(0.5, 200.0),
                        help="Depth of Lennard-Jones potential well between red and orange particles (0.5 to 200.0)")

    parser.add_argument('--r_blue_red', type=float, default=1.0, choices=CustomRange(0.5, 4.0),
                        help="Lennard-Jones potential radius of blue-red particles (0.5 to 4.0)")
    parser.add_argument('--r_blue_orange', type=float, default=0.8, choices=CustomRange(0.5, 4.0),
                        help="Lennard-Jones potential radius of blue and orange particles (0.5 to 4.0)")
    parser.add_argument('--r_red_orange', type=float, default=1.5, choices=CustomRange(0.5, 4.0),
                        help="Lennard-Jones potential radius of red and orange particles (0.5 to 4.0)")

    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/chem_0.bmp")

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (237, 28, 36): {'v': 0.5, 'eps': 1.0},
        (255, 127, 39): {'v': 0.5, 'eps': 1.0},
        (255, 242, 0): {'v': 0.5, 'eps': 1.0},
        (34, 177, 76): {'v': 0.5, 'eps': 1.0},
    })

    sim.set_eps(args.eps_blue_red, 1, 2)
    sim.set_r_min(args.r_blue_red, 1, 2)

    sim.set_r_min(args.r_blue_orange, 1, 3)
    sim.set_eps(args.eps_blue_orange, 1, 3)

    sim.set_r_min(args.r_red_orange, 3, 2)
    sim.set_eps(args.eps_red_orange, 2, 3)

    sim.set_r_min(1.5, 3)

    sim.time_step = 1.0 / 50
    sim.external_forces[:, int(sim.height * 0.5), 0] = 1.0

    camera = Camera(sim, sim.width * args.scale, sim.height * args.scale)
    camera.scale = args.scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))

    set_colors_in_rect(camera.colors,
                       load_img_to_gpu("simba_examples/inputs/texture_blue.jpg"),
                       sim.p_types, sim.positions,
                       80, 10, [1])

    set_colors_in_rect(camera.colors, load_img_to_gpu("simba_examples/inputs/texture_red.jpg"),
                       sim.p_types, sim.positions,
                       135, 66,
                       [2])

    set_colors_in_rect(camera.colors, load_img_to_gpu("simba_examples/inputs/texture_orange.jpg"),
                       sim.p_types, sim.positions,
                       180, 180,
                       [3])
    camera.colors_from_types({0: (1.0, 1.0, 1.0)})
    PyGameSimApp().run(sim, camera, seconds=args.sim_len, substeps=args.substeps,
                       render_options=vars(args).copy())


if __name__ == '__main__':
    main()
