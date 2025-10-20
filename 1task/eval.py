import numpy as np
import cv2


def predict_image_part_number(image: np.ndarray) -> list:
    """
    Определяет номера горизонтальных частей кадра,
    где находится граница между чёрным и белым цветами.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    part_height = h / 10

    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel)

    cols = np.array([np.argmax(sobel[:, c]) for c in range(0, w, max(1, w // 100))])
    row_min, row_max = cols.min(), cols.max()

    part_min = int(row_min // part_height) + 1
    part_max = int(row_max // part_height) + 1

    if part_min == part_max:
        return [part_min]
    return sorted([part_min, part_max])
