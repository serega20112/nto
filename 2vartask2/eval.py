# -*- coding: utf-8 -*-
import numpy as np
import cv2


def predict_type_of_road_markings(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val = np.percentile(gray, 99)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "third"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    perimeters = [cv2.arcLength(c, closed=False) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]

    avg_p = np.mean(perimeters)
    area = sum(areas)

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        else:
            centers.append((0, 0))

    gap = 0
    if len(centers) >= 2:
        gap = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))

    # Упрощённые правила

    if avg_p > 100:
        return "first"

    if avg_p > 50:
        return "fourth"

    if avg_p > 42:
        if gap < 150 and area < 180:
            return "second"
        else:
            return "fourth"

    if avg_p > 32:
        if gap < 100:
            return "second"
        elif gap > 200 or area > 130:
            return "fourth"
        else:
            return "third"

    # avg_p <= 32
    if gap < 100:
        return "second"
    else:
        return "third"