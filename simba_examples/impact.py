import numpy as np
from PIL import Image
from numba import cuda

from simba_ps.utils.helpers import set_vector_by_p_type
from simba_ps.utils.pygame_app import PyGameSimApp, CustomRange
from simba_ps.utils.util_kernels import get_bpg, set_vector_for_p_type
from simba_ps import Simba
from simba_ps.visualization.camera import Camera


def main():
    parser = PyGameSimApp.get_arg_parser("Simba 'Impact' example",
                                         default_substeps=20,
                                         default_scale=2,
                                         default_render_mode='smooth',
                                         default_v_max=7,
                                         default_falloff_radius=5)

    parser.add_argument('--eps_bulk', type=float, default=5.0, choices=CustomRange(0.5, 30.0),
                        help="Depth of Lennard-Jones potential well for particle bulk (0.5 to 30.0)")
    parser.add_argument('--eps_proj', type=float, default=10.0, choices=CustomRange(0.5, 30.0),
                        help="Depth of Lennard-Jones potential well for particle projectiles (0.5 to 30.0)")

    parser.add_argument('--mass_bulk', type=float, default=1.0, choices=CustomRange(0.5, 10.0),
                        help="Mass of bulk particles (0.5 to 30.0)")
    parser.add_argument('--mass_proj', type=float, default=2.0, choices=CustomRange(0.5, 10.0),
                        help="Mass of projectile particles (0.5 to 30.0)")

    args = parser.parse_args()

    img = Image.open("simba_examples/inputs/impact_480_270.bmp")

    sim = Simba.from_image(img, {
        (0, 0, 0): {'m': 0.0},

        (237, 28, 36): {'v': 0.5, 'eps': args.eps_proj, 'm': args.mass_proj},
        (255, 127, 39): {'v': 0.01, 'eps': args.eps_bulk, 'm': args.mass_bulk},
    })

    set_vector_by_p_type(1, cuda.to_device(np.array([0.0, -30.0])), sim.particle_count, sim.p_types, sim.velocities)

    sim.time_step = 1.0 / 50

    camera = Camera(sim, sim.width * args.scale, sim.height * args.scale)
    camera.scale = args.scale
    camera.view_point = cuda.to_device(np.array([sim.width * 0.5, sim.height * 0.5]))
    camera.colors_from_types({0: (1.0, 1.0, 1.0)})

    render_option = vars(args).copy()

    if args.render_mode == 'smooth':

        white = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        red = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        blue = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
        ]
        colors = [white, red, blue]
        render_option['colors'] = cuda.to_device(np.array(colors))

    PyGameSimApp().run(sim, camera, seconds=args.sim_len, substeps=args.substeps, render_options=render_option)


if __name__ == '__main__':
    main()
