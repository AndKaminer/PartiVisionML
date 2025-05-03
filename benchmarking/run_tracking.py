import cv2
import numpy as np

import argparse
import os

from utils import *


valid_options = {
        "tracking_type": ["MOG", "KNN"],
        "blur_type": ["gaussian", "median", None],
        "selection_type": ["largest", "cluster"],
        }


def main(video_path, tracking_type="MOG", blur_type=None, selection_type="largest", output_file="out.csv"):
    assert os.path.exists(video_path)
    assert tracking_type in valid_options["tracking_type"]
    assert blur_type in valid_options["blur_type"]
    assert selection_type in valid_options["selection_type"]

    cap = cv2.VideoCapture(video_path)

    if tracking_type == "MOG":
        fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=10, detectShadows=False)
    elif tracking_type == "KNN":
        fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    else:
        fgbg = None


    if blur_type == "gaussian":
        kernel_size = (3, 3)
        blur = lambda fgmask : cv2.GaussianBlur(fgmask, kernel_size, 0)
    elif blur_type == "median":
        kernel_size = 5
        blur = lambda fgmask : cv2.medianBlur(fgmask, kernel_size)
    else:
        blur = lambda fgmask : fgmask


    if selection_type == "largest":
        selector = largest_selector
    elif selection_type == "cluster":
        threshold_distance = 10.0
        selector = lambda blurred_mask : cluster_selector(blurred_mask, threshold_distance)
    else:
        raise Exception("Invalid selector type")

    contour_list = []

    while 1:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        blurred_mask = blur(fgmask)
        contour, final_frame = selector(blurred_mask)
        contour_list.append(contour)

        cv2.imshow("Frame", final_frame)
        if cv2.waitKey(30) == 27:
            break

    calculate_statistics(contour_list, output_file)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--tracking_type", default="MOG", help="Type of background subtraction to use. Can be either 'MOG' or 'KNN'.")
    parser.add_argument("--blur_type", default=None, help="Type of blur to apply. Optional argument. Can be either 'guassian' or 'median'")
    parser.add_argument("--selection_type", default="largest", help="Type of selection. See benchmarking-notes.md. Default option is 'largest'. Can be either 'largest' or 'cluster'.")
    parser.add_argument("--output_file", default="out.csv")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    tracking_type = args.tracking_type.upper()
    blur_type = args.blur_type.lower() if args.blur_type is not None else None
    selection_type = args.selection_type.lower()
    output_file = os.path.abspath(args.output_file)

    main(video_path=video_path,
         tracking_type=tracking_type,
         blur_type=blur_type,
         selection_type=selection_type,
         output_file=output_file)
