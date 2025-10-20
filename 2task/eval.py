# -*- coding: utf-8 -*-
import numpy as np
import cv2

def predict_type_of_road_markings(image: np.ndarray) -> str:
    lower_white = np.array([245, 245, 245], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(image, lower_white, upper_white)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    if len(contours) < 2:
        return "fourth"

    lengths = [cv2.arcLength(c, closed=False) for c in contours]
    moments1 = cv2.moments(contours[0])
    moments2 = cv2.moments(contours[1])
    cx1 = int(moments1["m10"] / moments1["m00"]) if moments1["m00"] != 0 else 0
    cy1 = int(moments1["m01"] / moments1["m00"]) if moments1["m00"] != 0 else 0
    cx2 = int(moments2["m10"] / moments2["m00"]) if moments2["m00"] != 0 else 0
    cy2 = int(moments2["m01"] / moments2["m00"]) if moments2["m00"] != 0 else 0
    gap = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    avg_length = np.mean(lengths)

    if 60 < avg_length <= 120 and 40 < gap <= 100:
        predicted = "first"
    elif 35 < avg_length <= 60 and 15 < gap <= 40:
        predicted = "second"
    elif 20 < avg_length <= 35 and 10 < gap <= 15:
        predicted = "third"
    else:
        predicted = "fourth"

    return predicted
