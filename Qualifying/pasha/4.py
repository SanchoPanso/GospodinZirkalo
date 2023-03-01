import numpy as np
import cv2
from typing import Tuple
import math

#find green tube
def find_id_4(img: np.ndarray) -> Tuple[int] or None:

    #range of color for green
    blue_min = np.array([40, 50, 110])
    blue_max = np.array([70, 225, 255])
    #blur
    img_blurred = cv2.medianBlur(img, 51)
    #to hsv
    imgHSV = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                           kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                           kernel, iterations=4)
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

        necessary_object.append((4, int(x-0.25*w), int(y-0.25*h), int(x + w*1.25), int(y + h*1.25)))

    if len(necessary_object) == 0:
        return None
    #that object may have to 2 bounding rectangles, so have to unite them
    object = [4, necessary_object[0][1],necessary_object[0][3],necessary_object[0][2],necessary_object[0][4]]
    for i in necessary_object:
        if i[1] < object[1]:
            object[1] = i[1]
        if i[2] < object[2]:
            object[2] = i[2]
        if i[3] > object[3]:
            object[3] = i[3]
        if i[4] > object[4]:
            object[4] = i[4]
    return object[1:]

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

