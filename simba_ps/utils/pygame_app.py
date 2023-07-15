import argparse
import sys

import numpy as np
import pygame
from numba import cuda
from pygame import QUIT

from simba_ps import Simba
from simba_ps.visualization.camera import Camera
from simba_ps.visualization.video_writer import VideoCapture


class PyGameSimApp:
    def __init__(self):
        self.screen = None
        self.video_capture = None
        self.fps_timer = pygame.time.Clock()

    @staticmethod
    def get_arg_parser(description="Simba particle simulation",
                       default_render_mode='velocity',
                       default_substeps=20,
                       default_v_max=10.0,
                       default_falloff_radius=3.5,
                       default_scale=2.0,
                       default_sim_len=60.0):

        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--render_mode', choices=['velocity', 'color', 'smooth'], default=default_render_mode,
                            help="Sets rendering mode to either color particles according to their velocity magnitude,"
                                 " to use static colors, rendering them as circles, or to use smooth (but slow) render"
                                 " computing particle density for each particle type")

        parser.add_argument('--output_video_file', default=None, type=str, metavar='output_file_name.mp4',
                            help="Output video file name. If provided *.mp4 extension video will be written,"
                                 " if no extension provided, output_video_file dir "
                                 "will be created and raw png frames will be written."
                                 "If not specified, no video will be written.")

        parser.add_argument('--output_video_frame_skip', default=1, type=int,
                            help="Write each output_video_frame_skip frame to video file")

        parser.add_argument('--substeps', type=int, default=default_substeps, choices=CustomRange(5, 40),
                            metavar='[5-40]', help="Sub step count for Verlet integration (5 to 40)")

        parser.add_argument('--sim_len', type=float, default=default_sim_len, choices=CustomRange(0.0, 3600.0),
                            help="Simulation length in seconds (0.0 to 3600.0)")

        parser.add_argument('--scale', type=float, default=default_scale, choices=CustomRange(1.0, 10.0),
                            help="Rendering scaling (1.0 to 10.0)")

        parser.add_argument('--value_max', type=float, default=default_v_max, choices=CustomRange(0.1, 1001.0),
                            help="Maximum value for gradient computation in velocity and smooth render modes")

        parser.add_argument('--falloff_radius', type=float, default=default_falloff_radius,
                            choices=CustomRange(1.0, 25.0),
                            help="Falloff radius for gradient computation in smooth render mode")
        return parser

    def finish(self, sim: Simba, cam: Camera):
        if self.video_capture is not None:
            self.video_capture.release_video()
        pygame.quit()
        sys.exit()

    def process_events(self, frame, sim, cam):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.finish(sim, cam)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.finish(sim, cam)

    def run(self, sim, cam, seconds=60.0, substeps=20, render_options=None):

        if render_options is None:
            render_options = dict()

        mode = render_options.setdefault('render_mode', 'velocity')

        v_min = 0.0
        v_max = 15.0
        gradient_colors = cuda.to_device(
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]))
        falloff_radius = 2.5

        if mode == 'smooth':
            v_min = 0.0
            v_max = 3.5
            white = [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
            red = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
            blue = [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 1.0],
            ]

            green = [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]

            colors = [white, red, blue, green]
            while len(colors) < sim.p_types_count:
                random_grad_color = np.random.random((3, 3))
                random_grad_color[0, :] = 0.0
                colors.append(random_grad_color)

            gradient_colors = cuda.to_device(np.array(colors))
            falloff_radius = render_options.setdefault('falloff_radius', falloff_radius)

        if mode == 'velocity' or mode == 'smooth':
            v_min = render_options.setdefault('value_min', v_min)
            v_max = render_options.setdefault('value_max', v_max)
            gradient_colors = render_options.setdefault('colors', gradient_colors)

        video_filename = None
        video_fps = 50
        video_frame_skip = 1

        if 'output_video_file' in render_options:
            video_filename = render_options['output_video_file']
            video_fps = render_options.setdefault('output_video_fps', video_fps)
            video_frame_skip = render_options.setdefault('output_video_frame_skip', video_frame_skip)

        frame = 0

        if self.screen is None:
            self.screen = pygame.display.set_mode((cam.width, cam.height))
            if video_filename is not None:
                self.video_capture = VideoCapture(video_filename, self.screen, video_fps)
            pygame.display.set_caption("Simba")

        self.on_start(sim, cam)

        while sim.time < seconds:

            self.process_events(frame, sim, cam)

            if mode == 'velocity':
                cam.set_colors_from_velocity(gradient_colors, v_min, v_max)

            if mode != 'smooth':
                cam.color_render()
            else:
                cam.smooth_render(gradient_colors, v_min, v_max, falloff_radius)

            buf = cam.screen_buffer.copy_to_host()
            pygame.pixelcopy.array_to_surface(self.screen, buf)

            if self.video_capture is not None and frame % video_frame_skip == 0:
                self.video_capture.capture_frame()

            frame += 1
            cap = self.caption(sim, cam)
            pygame.display.set_caption(cap)
            pygame.display.update()

            dt = sim.time_step / substeps
            for i in range(substeps):
                sim.verelet(dt)
                self.on_substep_finished(i, dt, sim, cam)
            self.on_computation_finished(frame, sim, cam)

            self.fps_timer.tick(100)

        print(f"Rendered {frame} frames, sim time: {sim.time:1.2f}, particles: {sim.particle_count}")
        self.finish(sim, cam)

    def on_start(self, sim, cam):
        pass

    def on_substep_finished(self, i, dt, sim, cam):
        pass

    def on_computation_finished(self, frame, sim, cam):
        pass

    def caption(self, sim, cam):
        return f"Simba - particles: {sim.particle_count}," \
               f" sim time: {sim.time:1.2f}," \
               f"fps: {self.fps_timer.get_fps():1.2f}"


class CustomRange:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __contains__(self, value):
        return self.lower <= value <= self.upper

    def __getitem__(self, index):
        if index == 0:
            return self
        else:
            raise IndexError()

    def __repr__(self):
        return f'[{self.lower}, {self.upper}]'
