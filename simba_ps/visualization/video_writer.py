import os
import threading

from PIL import Image
import pygame
import cv2
import numpy as np
from queue import Queue


class WriterThread(threading.Thread):
    def __init__(self, frame_queue, video, surface, filename=None):
        super().__init__()
        self.render_queue: Queue = frame_queue
        self.daemon = True
        self.video = video
        self.surface = surface
        self.save_raw = video is None
        self.filename = filename

    def run(self) -> None:
        frame = 0
        if self.save_raw and not os.path.exists(self.filename):
            os.mkdir(self.filename)

        while True:
            raw_str = self.render_queue.get(block=True)
            image = Image.frombuffer('RGBA', self.surface.get_size(), raw_str)
            if self.save_raw:
                image.save(os.path.join(self.filename, f"{frame:05d}.png"))
            else:
                self.video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            frame += 1


class VideoCapture:
    def __init__(self, filename, surface, fps=50):
        self.filename = filename
        self.fps = fps
        self.frame_queue = Queue(8)
        self.surface = surface
        self.save_raw = False
        self.video = None

        output_path = os.path.dirname(self.filename)
        if not os.path.exists(output_path):
            print(f"The path {output_path} does not exist, can not write '{filename}'")
            raise FileNotFoundError(filename)
        _, file_extension = os.path.splitext(self.filename)
        if file_extension == '':
            self.save_raw = True
            print(f"Writing raw image sequence to '{self.filename}'")
        else:
            print(f"Writing video to '{self.filename}'")
            size = self.surface.get_size()

            self.width = size[0]
            self.height = size[1]

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.video = cv2.VideoWriter(filename,
                                         fourcc,
                                         fps,
                                         (self.width, self.height))

        self.write_thread = WriterThread(self.frame_queue, self.video, self.surface, filename=filename)
        self.write_thread.start()

    def capture_frame(self):
        raw_str = pygame.image.tostring(self.surface, 'RGBA', False)
        self.frame_queue.put(raw_str)

    def release_video(self):
        if self.video is not None:
            self.video.release()
        self.video = None
