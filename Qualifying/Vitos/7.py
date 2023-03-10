import cv2
import numpy as np
from typing import Tuple


def find_id_7(img: np.ndarray) -> Tuple[int] or None:

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(img)
    corners = np.array(corners)

    if len(corners) < 1:
        return None

    if len(ids) < 2:
        all_x = [corners[0, 0, 0, 0], # первый - номер маркера, второй - бесполезный, третий - номер углы, четвертый - ось
                 corners[0, 0, 1, 0],
                 corners[0, 0, 2, 0],
                 corners[0, 0, 3, 0]]

        all_y = [corners[0, 0, 0, 1],
                 corners[0, 0, 1, 1],
                 corners[0, 0, 2, 1],
                 corners[0, 0, 3, 1]]

        all_x.sort()
        all_y.sort()

        x_min = int(all_x[0]) - int((int(all_x[-1])-int(all_x[0]))/10)
        x_max = int(all_x[-1]) + int((int(all_x[-1])-int(all_x[0]))/1)
        y_min = int(all_y[0]) - int((int(all_y[-1])-int(all_y[0]))/10)
        y_max = int(all_y[-1]) + int((int(all_y[-1])-int(all_y[0]))/10)

        cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 255, 0), 2)
        cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 255, 0), 2)
        cv2.line(img, (x_max, y_max), (x_max, y_min), (0, 255, 0), 2)
        cv2.line(img, (x_max, y_min), (x_min, y_min), (0, 255, 0), 2)

        returning_data = []
        returning_data.append((x_min, y_min, x_max, y_max))

        if len(returning_data) == 0:
            return None

        return returning_data[0]

    all_x = [corners[0, 0, 0, 0],#первый - номер маркера, второй - бесполезный, третий - номер углы, четвертый - ось
             corners[0, 0, 1, 0],
             corners[0, 0, 2, 0],
             corners[0, 0, 3, 0],
             corners[1, 0, 0, 0],
             corners[1, 0, 1, 0],
             corners[1, 0, 2, 0],
             corners[1, 0, 3, 0]]
    all_y = [corners[0, 0, 0, 1],
             corners[0, 0, 1, 1],
             corners[0, 0, 2, 1],
             corners[0, 0, 3, 1],
             corners[1, 0, 0, 1],
             corners[1, 0, 1, 1],
             corners[1, 0, 2, 1],
             corners[1, 0, 3, 1]]
    all_x.sort()
    all_y.sort()

    x_min = int(all_x[0]) - int((int(all_x[-1]) - int(all_x[0])) / 10)
    x_max = int(all_x[-1]) + int((int(all_x[-1]) - int(all_x[0])) / 10)
    y_min = int(all_y[0]) - int((int(all_y[-1]) - int(all_y[0])) / 10)
    y_max = int(all_y[-1]) + int((int(all_y[-1]) - int(all_y[0])) / 10)

    cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 255, 0), 2)
    cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 255, 0), 2)
    cv2.line(img, (x_max, y_max), (x_max, y_min), (0, 255, 0), 2)
    cv2.line(img, (x_max, y_min), (x_min, y_min), (0, 255, 0), 2)

    returning_data = []
    returning_data.append((x_min, y_min, x_max, y_max))

    if len(returning_data) == 0:
        return None

    return returning_data[0]


if __name__ == '__main__':

    src = cv2.imread("./photo/11.jpg")
    scale_percent = 20
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    last_transform = np.asarray(resized)
    get_bbox = []
    get_bbox = find_id_7(resized)
    print(get_bbox)
    cv2.imshow('img', resized)

    while True:
        k = cv2.waitKey(30)
        if k == 27:
            break
    cv2.destroyAllWindows()


