import numpy as np
import cv2

def predict_type_of_road_markings(image: np.ndarray) -> str:
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
    avg_perimeter = np.mean(perimeters)
    total_area = sum(areas)

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

    h, w = image.shape[:2]
    ratio = w / h

    # РћРўР›РђР”РљРђ: РїРµС‡Р°С‚Р°РµРј РїСЂРёР·РЅР°РєРё
    print(f"DEBUG: w={w}, h={h}, ratio={ratio:.3f}, avg_p={avg_perimeter:.1f}, area={total_area:.1f}, gap={gap:.1f}")

    p1 = 68.0 * ratio
    p2 = 35.0 * ratio
    g1 = 190.0 * ratio

    if avg_perimeter > p1:
        return "first"
    elif avg_perimeter > p2:
        return "second" if gap < g1 else "fourth"
    else:
        compactness = total_area / (avg_perimeter + 1e-5)
        if total_area > 1200 * ratio or compactness > 15 * ratio:
            return "second"
        return "third"