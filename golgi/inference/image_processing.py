from .data_handler import DataHandler
from .util import Util

import numpy as np
import cv2
import diplib as dip

class ProcessedImage:

    def __init__(self, frame, centerX, window_width, scaling_factor, um_per_pixel, model):
        
        self.frame = frame
        self.centerX = centerX
        self.window_width = window_width
        self.model = model
        self.scaling_factor = scaling_factor
        self.um_per_pixel = um_per_pixel

        self.data = {
                'area' : 0,
                'perimeter' : 0,
                'height' : 0,
                'circularity' : 0,
                'centerX' : None}
        
        self.contour = None
        self.contour = self.get_contour()
        self.process_contour(self.contour)


    def get_dip_measurement(self):
        binary_image = cv2.cvtColor(np.zeros_like(self.frame), cv2.COLOR_BGR2GRAY)

        cv2.drawContours(binary_image, [self.contour], -1, (1), thickness=cv2.FILLED)

        binary_image_dip = dip.Image(binary_image.astype(np.uint8))
        binary_image_dip.Convert('BIN')

        labeled_image = dip.Label(binary_image_dip)
        measurement = dip.MeasurementTool.Measure(labeled_image, features=['Perimeter', 'SolidArea', 'Roundness'])

        return measurement


    def get_contour(self):
        return self.contour


    def process_contour(self, contour):
        if contour is None:
            return

        M = cv2.moments(contour)
        m10, m00 = M["m10"], M["m00"]
        if m00 != 0:
            self.data['centerX'] = int(m10 // m00)
        
        _, _, _, h = cv2.boundingRect(contour)

        self.data['height'] = h * self.um_per_pixel / self.scaling_factor

        measurement = self.get_dip_measurement()

        self.data['perimeter'] = measurement['Perimeter'][1][0] * self.um_per_pixel / self.scaling_factor
        self.data['area'] = measurement['SolidArea'][1][0] * pow(self.um_per_pixel, 2) / pow(self.scaling_factor, 2)
        self.data['circularity'] = measurement['Roundness'][1][0] 


    def get_contour(self):
        if self.contour is not None:
            return self.contour

        width = self.frame.shape[1]

        new_window = Util.get_window(self.frame, self.centerX, self.window_width)

        if new_window is None:
            return None

        results = self.model(new_window, max_det=1, verbose=False)

        mask = []

        if not results[0].masks:
            return None

        for r in results:
            plot = r.plot(boxes=False)
            for x, y in r.masks.xy[0]:
                mask.append(int(x) + self.centerX - self.window_width)
                mask.append(int(y))
        ctr = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
        
        self.contour = ctr
        return ctr


    def get_parameter(self, parameter):
        return self.data[parameter]
