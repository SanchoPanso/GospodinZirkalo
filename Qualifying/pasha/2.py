import numpy as np
import cv2
from typing import Tuple
import math

#find blue ball
def find_id_2(img: np.ndarray) -> Tuple[int] or None:

    #range of color for blue
    blue_min = np.array([110, 100, 100])
    blue_max = np.array([125, 215, 255])

    #to hsv
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_w = 0
    max_h = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 5000:
            continue

        if not _contour_is_circle(cnt):
            continue

        if w*h > max_w*max_h:
            max_w = w
            max_h = h
        else:
            continue

        necessary_object = (1, x, y, x + w, y + h)

        # # Debug
        cv2.drawContours(img, contours, i, (0, 255, 0), 5)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)
        print(x, y, w, h, perimeter, square)

    cv2.imshow("cnts", cv2.resize(img, (600, 600)))
    cv2.waitKey()

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

    if abs(square - circle_square) / circle_square > 0.1:
        return False

    if abs(perimeter - circle_perimeter) / circle_perimeter > 0.1:
        return False

    return True

image = cv2.imread("D:\pythonfiles\cv1\images_argus\(15).jpg")
print(find_id_2(image))
