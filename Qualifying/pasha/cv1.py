import numpy as np
import cv2
from typing import Tuple

#find red tray
def find_id_1(img: np.ndarray) -> Tuple[int] or None:

    #range of color for red
    red_min = np.array([0, 149, 85])
    red_max = np.array([6, 255, 237])

    #to hsv
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, red_min, red_max)
    #morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_area = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 2000:
            continue

        if square > max_area:
            max_area = square
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

