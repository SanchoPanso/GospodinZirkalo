import cv2
import os
from pathlib import Path
import numpy as np
import math
from typing import Tuple


def get_tmplt_cnt():
    
    tmplt_cnt = np.array(
        [
            [[0, 0]],
            [[10, 0]],
            [[5, 8.66]],
        ],
        dtype=np.int32,
    )
    
    return tmplt_cnt


def find_id_11(img: np.ndarray) -> Tuple[int] or None:
    
    # img = cv2.GaussianBlur(img, (9, 9), 3)
    
    # Range of color
    # lower = np.array([25, 99, 130])
    # upper = np.array([36, 255, 255])
    
    lower = np.array([26, 97, 151])
    upper = np.array([36, 255, 255])
    
    # template
    tmplt_cnt = get_tmplt_cnt()
    
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
            print(cnt_area)
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
    
    # fix width, height
    w = x2 - x1
    h = y2 - y1
    x1 -= int(w * 0.25)
    x2 += int(w * 0.25)
    y1 -= int(h * 0.1)
    y2 += int(h * 0.1)
    
    
    return x1, y1, x2, y2
    

def run_searcing_func(src_path, dst_path):
    img = cv2.imread(src_path)
    result = find_id_11(img)
    if result is not None:
        x1, y1, x2, y2 = result
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
        
    cv2.imshow('img', cv2.resize(img, (600, 600)))
    cv2.waitKey(100)
    cv2.imwrite(dst_path, img)
    

if __name__ == '__main__':
    image_dir = os.path.join(str(Path(__file__).parent), 'images_argus')
    save_dir = os.path.join(str(Path(__file__).parent), 'res_id_11')
    image_files = os.listdir(image_dir)
    for filename in image_files: 
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(save_dir, filename)
        run_searcing_func(src_path, dst_path)