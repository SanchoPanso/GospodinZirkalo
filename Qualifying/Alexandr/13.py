import cv2
import os
from pathlib import Path
import numpy as np
import math
from typing import Tuple


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
    
    cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    cv2.waitKey()
    
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
    
    # fix width, height
    x1, y1, x2, y2 = necessary_objects[0]
    w = x2 - x1
    h = y2 - y1
    x1 -= int(w * 0.3)
    x2 += int(w * 0.3)
    y1 -= int(h * 0.05)
    y2 += int(h * 0.05)
    
    return x1, y1, x2, y2


def run_searcing_func(src_path, dst_path):
    img = cv2.imread(src_path)
    result = find_id_13(img)
    if result is not None:
        x1, y1, x2, y2 = result
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        
    cv2.imshow(filename, cv2.resize(img, (600, 600)))
    cv2.waitKey()
    cv2.imwrite(dst_path, img)


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
    

if __name__ == '__main__':
    image_dir = os.path.join(str(Path(__file__).parent), 'images_argus')
    save_dir = os.path.join(str(Path(__file__).parent), 'res_id_13')
    image_files = os.listdir(image_dir)
    for filename in image_files: 
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(save_dir, filename)
        run_searcing_func(src_path, dst_path)


