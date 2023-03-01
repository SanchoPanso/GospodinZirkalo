import numpy as np
import cv2
from typing import Tuple
import math

#find tennis ball
def find_id_3(img: np.ndarray) -> Tuple[int] or None:

    #range of color for yellow-green
    blue_min = np.array([30, 85, 165])
    blue_max = np.array([50, 255, 255])
    #blur
    img_blurred = cv2.medianBlur(img, 51)
    #to hsv
    imgHSV = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=2)
    #find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_area = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 4000:
            continue

        if not _contour_is_circle(cnt):
            continue
        #find biggest one
        if square > max_area:
            max_area = square
        else:
            continue

        necessary_object = (x, y, x + w, y + h)

    if len(necessary_object) == 0:
        return None

    return necessary_object

def _contour_is_circle(cnt) -> bool:
    perimeter = cv2.arcLength(cnt, True)
    square = cv2.contourArea(cnt)

    (xc, yc), radius = cv2.minEnclosingCircle(cnt)
    center = (int(xc), int(yc))
    radius = int(radius)

    circle_square = math.pi * (radius ** 2)
    circle_perimeter = 2 * math.pi * radius

    if abs(square - circle_square) / circle_square > 0.3:
        return False

    if abs(perimeter - circle_perimeter) / circle_perimeter > 0.3:
        return False

    return True
