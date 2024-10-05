from collections import deque
import os

from .weight_manager import WeightManager
from .data_handler import DataHandler
from .util import Util
from .image_processing import ProcessedImage

import cv2
import tqdm
import numpy as np
import pandas as pd

class InferencePipeline:
    
    def __init__(self, model, framerate, window_width, scaling_factor, um_per_pixel, output_folder):
        self.model = model
        self.framerate = framerate
        self.scaling_factor = scaling_factor
        self.um_per_pixel = um_per_pixel
        self.output_folder = output_folder
        self.window_width = window_width

        self.process_queue = deque()


    def process_video(self, video_path, scatter=False, verbose=False):
        video_name = os.path.basename(video_path)
        output_video_name = os.path.join(self.output_folder, os.path.splitext(video_name)[0] + "-analysis" + ".avi")
        output_csv_name = os.path.join(self.output_folder, os.path.splitext(video_name)[0] + "-analysis" + ".csv")

        dh = DataHandler(deltatime=1/self.framerate, scatter=scatter)

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scaling_factor
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scaling_factor

        centerX = width

        frame = np.zeros((height, width, 3), np.uint8)
        out = Util.combine_images(frame, dh.plot.get_img())
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 15, (out.shape[1], out.shape[0]))
        
        for _ in tqdm.tqdm(range(num_frames)):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (width, height), cv2.INTER_NEAREST)

            img = ProcessedImage(frame, centerX, self.window_width, self.scaling_factor, self.um_per_pixel, self.model)
            dh.update_data(area=img.get_parameter('area'),
                           perimeter=img.get_parameter('perimeter'),
                           height=img.get_parameter('height'),
                           circularity=img.get_parameter('circularity'),
                           centerX=img.get_parameter('centerX'))

            if img.get_contour() is not None:
                cv2.drawContours(frame, [img.get_contour()], -1, (0, 255, 0), 1)

            if dh.prev_data['centerX']:
                centerX = max(dh.prev_data['centerX'], self.window_width)

            out = Util.combine_images(frame, dh.plot.get_img())
            video.write(out)

            pd.DataFrame(dh.data).to_csv(output_csv_name)
            
        cap.release()
        cv2.destroyAllWindows()
        video.release()

        return dh.data
