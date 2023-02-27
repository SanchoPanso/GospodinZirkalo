import cv2
import numpy as np

from pathlib import Path
import os


def find_id_19(img: np.ndarray):
    bd = cv2.barcode.BarcodeDetector()

    h = img.shape[0]
    w = img.shape[1]

    grid_size = 5
    step_x = int(w / grid_size)
    step_y = int(h / grid_size)

    points_list = []
    for i in range(grid_size * 2):
        for j in range(grid_size * 2):
            x_1 = int(step_x * j / 2)
            y_1 = int(step_y * i / 2)
            x_2 = x_1 + step_x
            y_2 = y_1 + step_y

            roi = img[y_1: y_2, x_1: x_2]

            pass
            retval, decoded_info, decoded_type, points = bd.detectAndDecode(roi)
            if retval and decoded_info[0] == '15061027':
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
    points = np.array([points[1], points[3]])

    h_ = points[1, 1] - points[0, 1]
    w_ = points[1, 0] - points[0, 0]

    points[0, 0] -= w_
    points[0, 1] -= h_
    points[1, 0] += w_
    points[1, 1] += h_

    points = points.astype(int)

    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]

    # roi = img[y1: y2, x1: x2]
    # if np.mean(roi) < 180:
    #     return None

    return (x1, y1, x2, y2)


# if __name__ == '__main__':
#     img_path = r'D:\datasets\data\(10).jpg'
#
#     image = cv2.imread(img_path)
#     img_out = find_red_box(image)
#     if img_out is not None:
#         cv2.imwrite('124.png', img_out)

    # imgs_path = r'D:\datasets\data'
    # for img_name in os.listdir(imgs_path):
    #     image = cv2.imread(os.path.join(imgs_path, img_name))
    #     img_out = find_red_box(image)
    #
    #     if img_out is not None:
    #         cv2.imwrite(os.path.join('results', '17', img_name), img_out)
