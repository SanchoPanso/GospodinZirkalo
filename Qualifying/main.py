import numpy as np
import math
import cv2
from typing import Tuple


def main():
    functions = {
        '11': find_id_11,
        # '13': find_id_13,
    }
    
    image_path = input()
    img = cv2.imread(image_path)
    
    for id in functions:
        func = functions[id]
        res = func(img)
        
        if res is None:
            continue
        
        x1, y1, x2, y2 = res
        print(f"{id}:{x1};{y1};{x2};{y2}")


###### TASK 11 ######
#####################

def find_id_11(img: np.ndarray) -> Tuple[int] or None:
    
    # img = cv2.GaussianBlur(img, (9, 9), 3)
    
    # Range of color
    # lower = np.array([25, 99, 130])
    # upper = np.array([36, 255, 255])
    
    lower = np.array([26, 97, 151])
    upper = np.array([36, 255, 255])
    
    # template
    tmplt_cnt = np.array(
        [
            [[0, 0]],
            [[10, 0]],
            [[5, 8.66]],
        ],
        dtype=np.int32,
    )
    
    # Find mask of blue color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower, upper)
    
    # Noise removing
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 
        iterations=2
    )
    
    # cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    #cv2.waitKey()
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # vis_img = img.copy()
    # cv2.drawContours(vis_img, cnts, -1, (255, 0, 0), 10)
    # cv2.imshow("vis", cv2.resize(vis_img, (600, 600)))
    # cv2.waitKey()
    
    yellow_triangles = []
    
    for i, cnt in enumerate(cnts):
        metric = cv2.matchShapes(cnt, tmplt_cnt, cv2.CONTOURS_MATCH_I2, 0)
        cnt_area = cv2.contourArea(cnt)
        
        if cnt_area < 700 or cnt_area > 5000:
            continue
        
        if metric < 2.3:
            yellow_triangles.append(cnt)
            # cv2.drawContours(img, cnts, i, (0, 255, 0), 5)
    
    # cv2.imshow("img", cv2.resize(img, (600, 600)))
    # cv2.waitKey()
    
    if len(yellow_triangles) == 0:
        return None
    
    x1, y1, x2, y2 = 0, 0, 0, 0    
    for i, cnt in enumerate(yellow_triangles):
        x, y, w, h = cv2.boundingRect(cnt)
        if i == 0:
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)
    
    return x1, y1, x2, y2

    

###### TASK 13 ######
#####################

def find_id_13(img: np.ndarray) -> Tuple[int] or None:
    
    # Range of circuitboard color
    circuitboard_lower = np.array([90, 75, 50])
    circuitboard_upper = np.array([120, 255, 255])
    
    # Find mask of blue color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, circuitboard_lower, circuitboard_upper)
    
    # Noise removing
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), 
        iterations=3
    )
    
    # cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    # cv2.waitKey()
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_objects = []
    
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Skip too small contours
        if square < 2000:
            continue
        
        # Skip blue circles
        if _contour_is_circle(cnt):
            continue
            
        # Skip bboxes with wrong spacial ratio
        if not 0.65 <= w / h <= 1.0:
            continue
        
        # Add to list
        necessary_objects.append((x, y, x + w, y + h))
        
        # # Debug
        # cv2.drawContours(img, contours, i, (0, 255, 0), 5)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)
        # # print(x, y, w, h, perimeter, square) 

    # cv2.imshow("cnts", cv2.resize(img, (600, 600)))
    # cv2.waitKey()
    
    if len(necessary_objects) == 0:
        return None
    
    return necessary_objects[0]


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

#####################


if __name__ == '__main__':
    main()

