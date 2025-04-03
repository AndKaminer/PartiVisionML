import cv2
import numpy as np

import os

class Util:

    def get_dip_values(contour, frame, scaling_factor, um_per_pixel=1):
        binary_image = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2GRAY)

        cv2.drawContours(binary_image, [contour], -1, (1), thickness=cv2.FILLED)

        binary_image_dip = dip.Image(binary_image.astype(np.uint8))
        binary_image_dip.Convert('BIN')

        labeled_image = dip.Label(binary_image_dip)
        measurement = dip.MeasurementTool.Measure(labeled_image, features=['Perimeter', 'SolidArea', 'Roundness'])
        perimeter = measurement['Perimeter'][1][0]
        area = measurement['SolidArea'][1][0]
        roundness = measurement['Roundness'][1][0]

        print(perimeter, measurement['Perimeter'], area)

        p, a = perimeter / scaling_factor * um_per_pixel, area / pow(scaling_factor, 2) * pow(um_per_pixel, 2)

        #return p, a, math.pi * 4 * a / pow(p, 2)
        return p, a, roundness



    def get_perimeter(contour, scaling_factor, um_per_pixel=1):
        raw = cv2.arcLength(contour, True)
        return raw / scaling_factor * um_per_pixel

    def get_area(contour, scaling_factor, um_per_pixel=1):
        raw = cv2.contourArea(contour)
        return raw / pow(scaling_factor, 2) * pow(um_per_pixel, 2)

    def get_circularity(contour, frame, scaling_factor, um_per_pixel=1):
        # area = Util.get_area(contour, scaling_factor, um_per_pixel)
        # perimeter = Util.get_perimeter(contour, scaling_factor, um_per_pixel)
        # return math.pi * 4 * area / pow(perimeter, 2)
        binary_image = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2GRAY)

        cv2.drawContours(binary_image, [contour], -1, (1), thickness=cv2.FILLED)

        binary_image_dip = dip.Image(binary_image.astype(np.uint8))
        binary_image_dip.Convert('BIN')

        labeled_image = dip.Label(binary_image_dip)

        measurement = dip.MeasurementTool.Measure(labeled_image, features=['Roundness'])

        roundness = measurement['Roundness'][1][0]
        return roundness


    def get_z_height(contour, scaling_factor, um_per_pixel=1):
        x, y, w, h = cv2.boundingRect(contour)
        return h * um_per_pixel / scaling_factor

    def combine_images(img1, img2, pad_color=(0,0,0)):
        w1 = img1.shape[1]
        w2 = img2.shape[1]

        if w1 > w2:
            return cv2.vconcat([img1, Util.pad_image(w1, img2)])
        else:
            return cv2.vconcat([Util.pad_image(w2, img1), img2])

    def pad_image(new_width, frame, pad_color=(0, 0, 0)):
        width, height = frame.shape[1], frame.shape[0]

        if width > new_width:
            raise Exception("New width too small")

        width_to_add = new_width - width
        wta_perside = width_to_add // 2

        blank_image = np.zeros((height,wta_perside,3), np.uint8)

        if width_to_add % 2 == 0:
            img = cv2.hconcat([blank_image, frame, blank_image])
        else:
            blank_image_2 = np.zeros((height, wta_perside + 1, 3), np.uint8)
            img = cv2.hconcat([blank_image, frame, blank_image_2])

        return img

    def plot_to_opencv(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img

    def get_window(frame, centerX, window_width):
        new_left = max(0, centerX - window_width)
        new_right = min(frame.shape[1], centerX + window_width)

        if new_left == 0:
            new_right = min(frame.shape[1], window_width)
        elif new_right == frame.shape[1]:
            new_left == max(0, frame.shape[1] - window_width)

        new_img = frame[:, new_left:new_right]

        return new_img

    def list_pt(path):
        return [ f for f in os.listdir(path) if f.endswith(".pt") ]
