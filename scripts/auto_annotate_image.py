from golgi.annotation import AnnotatedImage, AutoAnnotater

import argparse
import os

import easygui
import cv2

RESIZE_CONSTANT = 3

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=False)
    parser.add_argument('--endpoint', required=True)

    filename = parser.parse_args().filename
    endpoint_name = parser.parse_args().endpoint

    if not filename:
        filename = easygui.fileopenbox()

    if not os.path.isfile(filename):
        raise FileNotFoundError("Invalid filename")

    if not cv2.haveImageReader(filename):
        raise FileNotFoundError("File is not an image that opencv can decode!")

    img = cv2.imread(filename)

    auto_ann = AutoAnnotater(resize_constant=RESIZE_CONSTANT,
                             endpoint_name=endpoint_name,
                             access_key=os.environ["ACCESS_KEY"],
                             secret_key=os.environ["SECRET_KEY"])

    ann = auto_ann.annotate(img)

    ann.show()
