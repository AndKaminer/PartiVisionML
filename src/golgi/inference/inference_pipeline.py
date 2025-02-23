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
        self.progress = 0
        self.tracked_contours = {}

        self.process_queue = deque()


    def process_video(self, video_path, scatter=False, verbose=False, avi=True, csv=True, include_plots=True):
        self.tracked_contours = {} 

        video_name = os.path.basename(video_path)
        if self.output_folder:
            output_video_name = os.path.join(self.output_folder, os.path.splitext(video_name)[0] + "-analysis" + ".avi")
            output_csv_name = os.path.join(self.output_folder, os.path.splitext(video_name)[0] + "-analysis" + ".csv")
        else:
            output_video_name = None
            output_csv_name = None

        dh = DataHandler(deltatime=1/self.framerate, scatter=scatter)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scaling_factor
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scaling_factor
        centerX = width

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Cannot read frames from the video.")
        frame = cv2.resize(frame, (width, height), cv2.INTER_NEAREST)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

        if avi:
            if include_plots:
                dummy_out = Util.combine_images(frame, dh.plot.get_img())
                out_size = (dummy_out.shape[1], dummy_out.shape[0])
            else:
                out_size = (width, height)
            video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 15, out_size)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for cur_frame in tqdm.tqdm(range(num_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            self.progress = cur_frame / num_frames
            frame = cv2.resize(frame, (width, height), cv2.INTER_NEAREST)

            img = ProcessedImage(frame, centerX, self.window_width, self.scaling_factor, self.um_per_pixel, self.model)
            contour = img.get_contour()

            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                self.tracked_contours[cur_frame] = contour.tolist()

            dh.update_data(area=img.get_parameter('area'),
                        perimeter=img.get_parameter('perimeter'),
                        height=img.get_parameter('height'),
                        circularity=img.get_parameter('circularity'),
                        ypos=img.get_parameter('ypos'),
                        centerX=img.get_parameter('centerX'))

            if dh.prev_data['centerX']:
                centerX = max(dh.prev_data['centerX'], self.window_width)

            if avi:
                if include_plots:
                    out_frame = Util.combine_images(frame, dh.plot.get_img())
                else:
                    out_frame = frame
                video.write(out_frame)

        if csv:
            pd.DataFrame(dh.data).to_csv(output_csv_name)

        cap.release()
        cv2.destroyAllWindows()
        if avi:
            video.release()

        return dh.data


    def get_tracked_contours(self):
        return self.tracked_contours.copy()
