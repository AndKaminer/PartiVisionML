import cv2
import numpy as np
import pandas as pd


def calculate_contour_distance(contour1, contour2): 
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)

def agglomerative_cluster(contours, threshold_distance=40.0):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else: 
            break

    return current_contours

def merge_mask(fgmask, threshold_distance=40.0):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clustered_contours = agglomerative_cluster(list(contours), threshold_distance)

    mask = np.zeros(fgmask.shape[:2], np.uint8)
    cv2.drawContours(mask, contours, -1, (255), -1)
    return mask

def largest_selector(blurred_mask):
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, np.zeros(blurred_mask.shape[:2], np.uint8)

    winner = max(contours, key=cv2.contourArea)

    mask = np.zeros(blurred_mask.shape[:2], np.uint8)
    cv2.drawContours(mask, [winner], -1, (255), -1)

    return winner, mask

def cluster_selector(blurred_mask, threshold_distance):
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clustered_contours = agglomerative_cluster(list(contours), threshold_distance)
    if len(clustered_contours) == 0:
        return None, np.zeros(blurred_mask.shape[:2], np.uint8)

    winner = max(clustered_contours, key=cv2.contourArea)

    mask = np.zeros(blurred_mask.shape[:2], np.uint8)
    cv2.drawContours(mask, [winner], -1, (255), -1)

    return winner, mask

def calculate_statistics(contour_list, output_file):
    # For now, just doing area. Simple enough to add other metrics
    columns = ["area"]
    area_list = [ cv2.contourArea(contour) if type(contour) != type(None) else 0 for contour in contour_list ]
    df = pd.DataFrame(area_list, columns=columns)
    df.to_csv(output_file, index=True)
