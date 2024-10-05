import json
import tempfile
import os

from .config import Configs
from .drawing import Drawing

import cv2
import roboflow
import numpy as np


class AnnotatedImage:
    def __init__(self, img, mask, filename="defaultfilename.png"):
        self.img = img
        self.mask = mask
        self.tempdir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.tempdir.name, filename)
        self.json_filename = self.filename + "-annotation.json"

        self.annotation_json = Configs.get_default_json()
        self.annotation_json["images"] = [self._get_image(img)]
        self.annotation_json["annotations"] = [self._get_annotation(mask)]

        cv2.imwrite(self.filename, img)
        with open(self.json_filename, "w") as file:
            file.write(json.dumps(self.annotation_json))
        
    def __del__(self):
        self.tempdir.cleanup()

    def _get_annotation(self, mask):
        annotation = Configs.get_default_annotation()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
            segmentation.append([int(coord) for pair in contour_pairs for coord in pair])

        area = cv2.contourArea(contours[0]) if contours else 0
        bbox = [int(x) for x in cv2.boundingRect(contours[0])] if contours else None

        annotation["segmentation"] = segmentation
        annotation["area"] = area
        annotation["bbox"] = bbox

        return annotation

    def _get_image(self, img):
        image = Configs.get_default_image()
        shape = img.shape

        image["width"] = shape[0]
        image["height"] = shape[1]
        image["file_name"] = self.filename

        return image

    def roboflow_upload(self, workspace, project, api_key=None):
        if api_key is None:
            api_key = input("Please input your roboflow api key here: ")
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)

        project.upload(image_path=self.filename,
                       annotation_path=self.json_filename,
                       batch_name=Configs.default_roboflow_batch)

    def show(self):
         contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         img = self.img.copy()
         cv2.drawContours(img, contours, -1, (0, 0, 0), 3)
         cv2.imshow("Annotation", img)
         while True:
            key = cv2.waitKey(1000)
            if key == 27 or cv2.getWindowProperty("Annotation", cv2.WND_PROP_VISIBLE) < 1:
                break
         cv2.destroyWindow("Annotation")

    def get_resized_image(self, window_width, offset):

        if self.annotation_json["annotations"]:
            seg = self.annotation_json["annotations"][0]["segmentation"][0]
            ctr = AnnotatedImage.coco_to_opencv(seg)
            M = cv2.moments(ctr)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = random.randint(40, self.img.shape[1] - 40)
            cY = self.img.shape[0] // 2

        newLeft = max(0, offset + cX - window_width // 2)

        imwidth = self.img.shape[1]
        newRight = min(imwidth, offset + cX + window_width // 2)

        if self.annotation_json["annotations"]:
            new_ctr = []
            for i in range(len(seg)):
                if i % 2 == 0:
                    #new_ctr.append(seg[i] - newLeft - offset) # not sure why this is broken now
                    new_ctr.append(seg[i] - newLeft)
                else:
                    new_ctr.append(seg[i])
        else:
            new_ctr = None

        new_ctr = AnnotatedImage.coco_to_opencv(new_ctr)

        new_img = self.img[:, newLeft:newRight]
        mask = np.zeros(new_img.shape, np.uint8)
        cv2.drawContours(mask, new_ctr, -1, (255), 1)

        return AnnotatedImage(new_img, mask)


    def coco_to_opencv(segmentation):
        if type(segmentation) == type(None):
            return None

        points = [ [segmentation[i], segmentation[i+1]] for i in range(0, len(segmentation), 2) ]
        ctr = np.array(points).reshape((-1,1,2)).astype(np.int32)
        return ctr

    def opencv_to_coco(contour):
        if type(contour) == type(None):
            return None

        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
            segmentation.append([int(coord) for pair in contour_pairs for coord in pair])

        return segmentation



    # how to do image resizing?
    # just mutate the object


class ImageAnnotater:
    def __init__(self, image, resize_constant):
        self.image = image
        self.resize_constant = resize_constant

    def annotate(self):
        prepared_image = self._get_prepared_image(self.image)
        xmin, xmax, ymin, ymax = self._get_roi(prepared_image)
        mask = self._get_initial_mask(prepared_image, xmin, xmax, ymin, ymax)

        d = Drawing(prepared_image, mask)
        final_mask = d.draw()

        return AnnotatedImage(prepared_image, final_mask)

    def _get_initial_mask(self, prepared_image, xmin, xmax, ymin, ymax):
        lower, higher = 0, 255
        mask_view = False
        contour_view = True
        mask = None
        
        window_name = "Choose Initial Mask"
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, (1920, 1080))
        cv2.imshow(window_name, prepared_image)

        while True:
            key = cv2.waitKey(1)
                    
            if key == 27:
                break
            elif key == ord('l'):
                lower += 1
            elif key == ord('k'):
                lower -= 1
            elif key == ord('h'):
                higher -= 1
            elif key == ord('j'):
                higher += 1
            elif key == ord('m'):
                mask_view = not mask_view
            elif key == ord('c'):
                contour_view = not contour_view
            elif key == ord('f'):
                print('finishing')
                break

            mask = self._get_mask(prepared_image, lower, higher, xmin, xmax, ymin, ymax)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            frame = prepared_image.copy()

            if contour_view:
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

            frame = cv2.putText(frame, f"{lower}-{higher}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if mask_view:
                cv2.imshow(window_name, mask)
            else:
                cv2.imshow(window_name, frame)

        cv2.destroyWindow(window_name)

        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel, iterations = 2)
        return mask

    def _get_roi(self, image):
        cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Select ROI", (1920, 1080))

        bbox = cv2.selectROI("Select ROI", image)
        cv2.destroyWindow("Select ROI")

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]

        return xmin, xmax, ymin, ymax

    def _get_mask(self, image, lower, upper, xmin, xmax, ymin, ymax):
        thresh = cv2.inRange(image, lower, upper)
        bbox_mask = thresh.copy()
        bbox_mask[0:, 0:] = 0
        cv2.rectangle(bbox_mask, (xmin, ymin), (xmax, ymax), 255, -1)
        thresh_final = cv2.bitwise_and(thresh, bbox_mask)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        thresh_final = cv2.morphologyEx(thresh_final,cv2.MORPH_OPEN,kernel, iterations = 2)

        return thresh_final
        
    def _get_prepared_image(self, image):
        shape = image.shape
        image = cv2.resize(image, (shape[1] * self.resize_constant, shape[0] * self.resize_constant), cv2.INTER_NEAREST)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return imgray
