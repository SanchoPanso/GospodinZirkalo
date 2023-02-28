import sys
import cv2
import numpy as np
from typing import Tuple
import math



def find_id_10(img: np.ndarray) -> Tuple[int] or None:

    #gray_image = get_color_filtered_image(img = img, lower = (25, 0, 20), upper = (110, 50, 120))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0, 0)

    edges = cv2.Canny(gray, 100, 200)

    counter, ierarch = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(counter)):
        for j in range(0, len(counter)):

            mnts_a = cv2.moments(counter[i])
            mnts_b = cv2.moments(counter[j])

            if mnts_a['m00'] == 0:
                continue

            cmx_a = int(mnts_a['m10'] / mnts_a['m00'])
            cmy_a = int(mnts_a['m01'] / mnts_a['m00'])

            if mnts_b['m00'] == 0:
                continue

            cmx_b = int(mnts_b['m10'] / mnts_b['m00'])
            cmy_b = int(mnts_b['m01'] / mnts_b['m00'])

            dist = int(math.sqrt(math.pow((cmx_a - cmx_b), 2) + math.pow((cmy_a - cmy_b), 2)))

            print("D", dist)

            if dist < 100:
                cv2.polylines(img, counter[i], True, (255, 0, 255), 3)
                cv2.polylines(img, counter[j], True, (255, 0, 255), 3)


    cv2.imshow("detected circles", img)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":

    src = cv2.imread(cv2.samples.findFile("./photo/2.jpg"), cv2.IMREAD_COLOR)
    scale_percent = 20
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    last_transform = np.asarray(resized)
    find_id_10(last_transform)