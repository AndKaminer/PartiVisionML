from .image_annotation import AnnotatedImage, ImageAnnotater, Drawing

import tempfile
import os
import time

import huggingface_hub
from ultralytics import YOLO
import numpy as np
import cv2
import boto3


class ImageAutoAnnotater(ImageAnnotater):
    def __init__(self, image, resize_constant, model):
        super().__init__(image, resize_constant)
        self.model = model

    def annotate(self):
        prepared_image = self._get_prepared_image(self.image)
        xmin, xmax, ymin, ymax = self._get_roi(prepared_image)

        cropped_image, mask = self.auto_annotate(prepared_image, xmin, xmax, ymin, ymax)
        return AnnotatedImage(cropped_image, mask)
        

    def auto_annotate(self, prepared_image, xmin, xmax, ymin, ymax):
        window = prepared_image[ymin:ymax, xmin:xmax, :]

        ctr = self.model.infer(window)

        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(window.shape, np.uint8)
        cv2.drawContours(mask, [ctr], -1, (255), 1)

        mask = Drawing(window, mask).draw()        

        return window, mask

    def _get_prepared_image(self, image):
        shape = image.shape
        image = cv2.resize(image, (shape[1] * self.resize_constant, shape[0] * self.resize_constant), cv2.INTER_NEAREST)
        return image
