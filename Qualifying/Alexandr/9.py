import cv2
import os
from pathlib import Path
import numpy as np
import math
from typing import Tuple


def get_tmplt_cnt():
    
     # Range of color
    red_k_lower = np.array([0, 135, 0])
    red_k_upper = np.array([20, 255, 255])
    
    tmplt_img = cv2.imread(os.path.join(str(Path(__file__).parent), 'red_k.jpg'))
    img_hsv = cv2.cvtColor(tmplt_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, red_k_lower, red_k_upper)
    tmplt_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    tmplt_cnt = tmplt_cnts[0]
    for cnt in tmplt_cnts:
        cur_area = cv2.contourArea(cnt)
        if cur_area > max_area:
            max_area = cur_area
            tmplt_cnt = cnt
    
    return tmplt_cnt


def find_id_9(img: np.ndarray) -> Tuple[int] or None:
    
    # Range of color
    red_k_lower = np.array([0, 120, 0])
    red_k_upper = np.array([20, 255, 255])
    
    # template
    tmplt_cnt = get_tmplt_cnt()
    
    # Find mask of blue color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, red_k_lower, red_k_upper)
    
    # Noise removing
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 
        iterations=1
    )
    
    # cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    # cv2.waitKey()
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_img = img.copy()
    cv2.drawContours(vis_img, cnts, -1, (255, 0, 0), 5)
    cv2.imshow("vis", cv2.resize(vis_img, (600, 600)))
    cv2.waitKey()
    
    for i, cnt in enumerate(cnts):
        metric = cv2.matchShapes(cnt, tmplt_cnt, cv2.CONTOURS_MATCH_I2, 0)
        
        if cv2.contourArea(cnt) < 100:
            continue
        
        if metric < 2:
            cv2.drawContours(img, cnts, i, (0, 255, 0), 5)
    
    cv2.imshow("img", cv2.resize(img, (600, 600)))
    cv2.waitKey()
    return None


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
    


def run_searcing_func(src_path, dst_path):
    img = cv2.imread(src_path)
    result = find_id_9(img)
    if result is not None:
        x1, y1, x2, y2 = result
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        
    # cv2.imshow('img', cv2.resize(img, (600, 600)))
    # cv2.waitKey()
    # cv2.imwrite(dst_path, img)
    

if __name__ == '__main__':
    image_dir = os.path.join(str(Path(__file__).parent), 'images_argus')
    save_dir = os.path.join(str(Path(__file__).parent), 'res_id_13')
    image_files = os.listdir(image_dir)
    for filename in image_files: 
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(save_dir, filename)
        run_searcing_func(src_path, dst_path)