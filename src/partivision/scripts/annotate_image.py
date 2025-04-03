from partivision.annotation import AnnotatedImage, ImageAnnotater

import argparse
import os

import easygui
import cv2


RESIZE_CONSTANT = 3

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=False)
    parser.add_argument('--rfkey', required=False)

    filename = parser.parse_args().filename
    api_key = parser.parse_args().rfkey

    if not filename:
        filename = easygui.fileopenbox()

    if not os.path.isfile(filename):
        raise FileNotFoundError("Invalid filename")

    if not cv2.haveImageReader(filename):
        raise FileNotFoundError("File is not an image that opencv can decode!")

    img = cv2.imread(filename)

    ann = ImageAnnotater(img, RESIZE_CONSTANT)

    ann_img = ann.annotate()

    ann_img = ann_img.get_resized_image(400, 50)

    ann_img.show()
    
    if api_key:
        ann_img.roboflow_upload(
            workspace="cv-time",
            project="final-dataset-idfeu",
            api_key=api_key)
    else:
        ann_img.roboflow_upload(
            workspace="cv-time",
            project="final-dataset-idfeu")
