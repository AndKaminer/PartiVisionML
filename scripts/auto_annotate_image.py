from golgi.annotation import AnnotatedImage, AutoAnnotater

import argparse
import os

import easygui
import cv2

RESIZE_CONSTANT = 3

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=False)
    parser.add_argument('--key', required=False)

    filename = parser.parse_args().filename
    key = parser.parse_args().key

    if not filename:
        filename = easygui.fileopenbox()

    if not os.path.isfile(filename):
        raise FileNotFoundError("Invalid filename")

    if not cv2.haveImageReader(filename):
        raise FileNotFoundError("File is not an image that opencv can decode!")

    img = cv2.imread(filename)

    auto_ann = AutoAnnotater(resize_constant=RESIZE_CONSTANT,
                             model_repo_id="gt-sulchek-lab/cell-tracking",
                             model_filename="june20weights.pt",
                             key=key)

    ann = auto_ann.annotate(img)

    ann.show()
