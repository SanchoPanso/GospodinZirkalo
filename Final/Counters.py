import cv2
import numpy as np

def counter(input_img: np.ndarray):

    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 150, 1000, cv2.THRESH_BINARY)

    find_counters = dst.copy()

    couners_finded, herarhy = cv2.findContours(find_counters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);


    new_one = cv2.drawContours(input_img, couners_finded, -1, (255, 0, 0))

    return new_one


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_img = cv2.imread("Pasel.jpg")
    show_img = counter(start_img)

    while True:
        cv2.imshow('img', show_img)
        if cv2.waitKey(1) == 27:
            break

