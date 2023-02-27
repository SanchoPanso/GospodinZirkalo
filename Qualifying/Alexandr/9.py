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
    
    tmplt_img = cv2.imread(os.path.join(str(Path(__file__).parent), 'red_p.jpg'))
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


def logarithmic_transform(img: np.ndarray, c: float):
    table = np.zeros((256,), dtype='uint8')
    for i in range(256):
        table[i] = c * np.log2(1 + i)
    return cv2.LUT(img, table)


def find_id_9(img: np.ndarray) -> Tuple[int] or None:
    
    
    blur = cv2.GaussianBlur(img, (11, 11), 3)
    sharped = np.abs(img.astype('int32') - blur.astype('int32'))
    sharped = logarithmic_transform(sharped.astype('uint8'), 20)
    # cv2.imshow("sharped", cv2.resize(sharped, (1000, 1000)))
    # cv2.waitKey()
    
    # # Range of color
    lower = [64, 62, 42]
    upper = [94, 168, 255]

    
    # Find mask of blue color
    img_hsv = cv2.cvtColor(sharped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
    
    # cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    # cv2.waitKey()
    
    # Noise removing
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), 
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), 
        iterations=2
    )
    
    # cv2.imshow("mask", cv2.resize(mask, (600, 600)))
    # cv2.waitKey()
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # vis_img = img.copy()
    # cv2.drawContours(vis_img, cnts, -1, (255, 0, 0), 5)
    # cv2.imshow("vis", cv2.resize(vis_img, (600, 600)))
    # cv2.waitKey()
    
    red_text_proposals = []
    
    for i, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) < 3500:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        if 2 <= w / h <= 4:
            red_text_proposals.append([x, y, w, h])
            #cv2.drawContours(img, cnts, i, (0, 255, 0), 5)
    
    # cv2.imshow("img", cv2.resize(img, (600, 600)))
    # cv2.waitKey()
    
    if len(red_text_proposals) == 0:
        return None
    
    # Sort in decending order by square
    red_text_proposals.sort(key=lambda x: -(x[2] * x[3]))
    obj_x1, obj_y1, obj_x2, obj_y2 = 0, 0, 0, 0
    
    for i, (x, y, w, h) in enumerate(red_text_proposals):
        if i == 0:
            obj_x1 = x
            obj_y1 = y
            obj_x2 = x + w
            obj_y2 = y + h
        else:
            obj_xc = (obj_x1 + obj_x2) // 2
            obj_yc = (obj_y1 + obj_y2) // 2
            obj_w = (obj_x2 - obj_x1) // 2
            obj_h = (obj_y2 - obj_y1) // 2
            
            xc = x + w // 2
            yc = y + h // 2
            
            if math.sqrt(((obj_xc - xc) ** 2 + (obj_yc - yc) ** 2)) > img.shape[0] * 0.15:
                continue
            
            obj_x1 = min(obj_x1, x)
            obj_y1 = min(obj_y1, y)
            obj_x2 = max(obj_x2, x + w)
            obj_y2 = max(obj_y2, y + h)
    
    # fix width, height
    w = obj_x2 - obj_x1
    h = obj_y2 - obj_y1
    obj_x1 -= int(w * 0.5)
    obj_x2 += int(w * 0.5)
    obj_y1 -= int(h * 0.5)
    obj_y2 += int(h * 0.5)
    
    return (obj_x1, obj_y1, obj_x2, obj_y2)
    

def run_searcing_func(src_path, dst_path):
    img = cv2.imread(src_path)
    result = find_id_9(img)
    if result is not None:
        x1, y1, x2, y2 = result
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        
    cv2.imshow('img', cv2.resize(img, (600, 600)))
    cv2.waitKey(10)
    cv2.imwrite(dst_path, img)
    

if __name__ == '__main__':
    image_dir = os.path.join(str(Path(__file__).parent), 'images_argus')
    save_dir = os.path.join(str(Path(__file__).parent), 'res_id_9')
    image_files = os.listdir(image_dir)
    for filename in image_files: 
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(save_dir, filename)
        run_searcing_func(src_path, dst_path)