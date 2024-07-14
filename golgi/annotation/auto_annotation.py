from .image_annotation import AnnotatedImage, ImageAnnotater, Drawing

import tempfile
import os

import huggingface_hub
from ultralytics import YOLO
import numpy as np
import cv2


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

        results = self.model(window, max_det=1, verbose=False)

        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        mask = []

        if not results[0].masks:
            return window, np.zeros(window.shape, np.uint8)

        for r in results:
            plot = r.plot(boxes=False)
            for x, y in r.masks.xy[0]:
                mask.append(int(x))
                mask.append(int(y))

        ctr = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)

        mask = np.zeros(window.shape, np.uint8)
        cv2.drawContours(mask, [ctr], -1, (255), 1)

        mask = Drawing(window, mask).draw()        

        return window, mask

    def _get_prepared_image(self, image):
        shape = image.shape
        image = cv2.resize(image, (shape[1] * self.resize_constant, shape[0] * self.resize_constant), cv2.INTER_NEAREST)
        return image


class AutoAnnotater:
    def __init__(self, resize_constant, model_repo_id, model_filename, key=None):
        if key is not None:
            huggingface_hub.login(token=key)
        else:
            huggingface_hub.login()

        self.resize_constant = resize_constant
        tempdir = tempfile.TemporaryDirectory()
        huggingface_hub.hf_hub_download(repo_id=model_repo_id,
                                        local_dir=tempdir.name,
                                        filename=model_filename)

        self.model = YOLO(os.path.join(tempdir.name, model_filename))
        del tempdir

    def annotate(self, image):
        return ImageAutoAnnotater(image, self.resize_constant, self.model).annotate()
