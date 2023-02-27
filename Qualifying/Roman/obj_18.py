import cv2
import numpy as np

from pathlib import Path
import os


def find_red_box(img: np.ndarray):
    bd = cv2.barcode.BarcodeDetector()

    h = img.shape[0]
    w = img.shape[1]

    grid_size = 2
    step_x = int(w / grid_size)
    step_y = int(h / grid_size)

    points_list = []
    for i in range(grid_size * 4):
        for j in range(grid_size * 4):
            x_1 = int(step_x * j / 4)
            y_1 = int(step_y * i / 4)
            x_2 = x_1 + step_x
            y_2 = y_1 + step_y

            roi = img[y_1: y_2, x_1: x_2]

            if i == 2 and j == 1:
                pass

            retval, decoded_info, decoded_type, points = bd.detectAndDecode(roi)
            if decoded_info and decoded_info[0] != '15061027':
                points = points[0]
                points[:, 0] += x_1
                points[:, 1] += y_1
                points_list.append(points)
    area_list = []
    for pts in points_list:
        area = cv2.contourArea(pts)
        area_list.append(area)

    if len(area_list) == 0:
        return None

    points = points_list[area_list.index(max(area_list))]

    x1 = int(min(points[:, 0]))
    x2 = int(max(points[:, 0]))
    y1 = int(min(points[:, 1]))
    y2 = int(max(points[:, 1]))

    y2 += (y2 - y1)
    x2 += int((x2 - x1) * 9)
    x1 -= 150

    return (x1, y1, x2, y2)


if __name__ == '__main__':
    ROOT = Path(os.getcwd())
    imgs_path = r'D:\datasets\data'
    for img_name in os.listdir(imgs_path):
        # if img_name != '(7).jpg':
        #     continue
        image = cv2.imread(os.path.join(imgs_path, img_name))
        img_out = find_red_box(image)
        if img_out is not None:
            cv2.imwrite(os.path.join('results', '18', img_name), img_out)
