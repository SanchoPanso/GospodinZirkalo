import numpy as np
import math
import cv2
from typing import Tuple


def main():
    
    # Conformity between obj id and finding functions
    functions = {
        '1': find_id_1,
        '2': find_id_2,
        '3': find_id_3,
        '4': find_id_4,
        '5': lambda x: None,
        '6': lambda x: None,
        '7': find_id_7,
        '8': lambda x: None,
        '9': find_id_9,
        '10': lambda x: None,
        '11': find_id_11,
        '12': lambda x: None,
        '13': find_id_13, 
        '14': lambda x: None,
        '15': lambda x: None,
        '16': lambda x: None,
        '17': lambda x: None,
        '18': find_id_18,
        '19': find_id_19,
        '20': lambda x: None,
    }
    
    # Read the image
    image_path = input()
    img = cv2.imread(image_path)
    
    for id in functions:
        img_copy = img.copy()   # Function can damage orig img, so we need a copy
        func = functions[id]
        res = func(img_copy)
        
        # If res is None, then obj is not here (skip)
        if res is None:
            continue
        
        # Otherwise print its coordinates
        x1, y1, x2, y2 = map(int, res)
        print(f"{id}:{x1};{y1};{x2};{y2}")
        

############ TASK 1 #############
#################################

#find red tray
def find_id_1(img: np.ndarray) -> Tuple[int] or None:

    #range of color for red
    red_min = np.array([0, 149, 85])
    red_max = np.array([6, 255, 237])

    #to hsv
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, red_min, red_max)
    #morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_area = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 2000:
            continue

        if square > max_area:
            max_area = square
        else:
            continue

        necessary_object = (x, y, x + w, y + h)

    if len(necessary_object) == 0:
        return None

    return necessary_object

############ TASK 2 #############
#################################

#find blue ball
def find_id_2(img: np.ndarray) -> Tuple[int] or None:

    #range of color for blue
    blue_min = np.array([110, 100, 100])
    blue_max = np.array([125, 215, 255])

    #to hsv
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_w = 0
    max_h = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 5000:
            continue

        if not _contour_is_circle(cnt):
            continue

        if w*h > max_w*max_h:
            max_w = w
            max_h = h
        else:
            continue

        necessary_object = (x, y, x + w, y + h)

    #     # # Debug
    #     cv2.drawContours(img, contours, i, (0, 255, 0), 5)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 5)
    #     # print(x, y, w, h, perimeter, square)

    # # cv2.imshow("cnts", cv2.resize(img, (600, 600)))
    # # cv2.waitKey()

    if len(necessary_object) == 0:
        return None

    return necessary_object

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

############ TASK 3 #############
#################################

#find tennis ball
def find_id_3(img: np.ndarray) -> Tuple[int] or None:

    #range of color for yellow-green
    blue_min = np.array([30, 85, 165])
    blue_max = np.array([50, 255, 255])
    #blur
    img_blurred = cv2.medianBlur(img, 51)
    #to hsv
    imgHSV = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=2)
    #find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_area = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 4000:
            continue

        if not _contour_is_circle(cnt):
            continue
        #find biggest one
        if square > max_area:
            max_area = square
        else:
            continue

        necessary_object = (x, y, x + w, y + h)

    if len(necessary_object) == 0:
        return None

    return necessary_object


############ TASK 4 #############
#################################

#find green tube
def find_id_4(img: np.ndarray) -> Tuple[int] or None:

    #range of color for green
    blue_min = np.array([40, 50, 110])
    blue_max = np.array([70, 225, 255])
    #blur
    img_blurred = cv2.medianBlur(img, 51)
    #to hsv
    imgHSV = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    #filter by color
    mask = cv2.inRange(imgHSV, blue_min, blue_max)
    #morphology
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                           kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                           kernel, iterations=4)
    #find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    necessary_object = []
    max_area = 0
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        square = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip too small contours
        if square < 4000:
            continue

        necessary_object.append((4, int(x-0.25*w), int(y-0.25*h), int(x + w*1.25), int(y + h*1.25)))

    if len(necessary_object) == 0:
        return None
    #that object may have to 2 bounding rectangles, so have to unite them
    object = [4, necessary_object[0][1],necessary_object[0][3],necessary_object[0][2],necessary_object[0][4]]
    for i in necessary_object:
        if i[1] < object[1]:
            object[1] = i[1]
        if i[2] < object[2]:
            object[2] = i[2]
        if i[3] > object[3]:
            object[3] = i[3]
        if i[4] > object[4]:
            object[4] = i[4]
    return object[1:]

def _contour_is_circle(cnt) -> bool:
    perimeter = cv2.arcLength(cnt, True)
    square = cv2.contourArea(cnt)

    (xc, yc), radius = cv2.minEnclosingCircle(cnt)
    center = (int(xc), int(yc))
    radius = int(radius)

    circle_square = math.pi * (radius ** 2)
    circle_perimeter = 2 * math.pi * radius

    if abs(square - circle_square) / circle_square > 0.3:
        return False

    if abs(perimeter - circle_perimeter) / circle_perimeter > 0.3:
        return False

    return True


############ TASK 5 #############
#################################

############ TASK 6 #############
#################################

############ TASK 7 #############
#################################


def find_id_7(img: np.ndarray) -> Tuple[int] or None:

    #find ArUco markers on surfaces
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(img)
    corners = np.array(corners)

    #if no markers - no fun
    if len(corners) < 1:
        return None

    #if only one marker
    if len(ids) < 2:
        #get corners data by axes
        all_x = [corners[0, 0, 0, 0], # первый - номер маркера, второй - бесполезный, третий - номер углы, четвертый - ось
                 corners[0, 0, 1, 0],
                 corners[0, 0, 2, 0],
                 corners[0, 0, 3, 0]]

        all_y = [corners[0, 0, 0, 1],
                 corners[0, 0, 1, 1],
                 corners[0, 0, 2, 1],
                 corners[0, 0, 3, 1]]

        #sort this data
        all_x.sort()
        all_y.sort()

        #set bbox axis data
        x_min = int(all_x[0]) - 5
        x_max = int(all_x[-1]) + 30
        y_min = int(all_y[0]) - 10
        y_max = int(all_y[-1]) + 10

        #draw lines by 4 points
        cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 255, 0), 2)
        cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 255, 0), 2)
        cv2.line(img, (x_max, y_max), (x_max, y_min), (0, 255, 0), 2)
        cv2.line(img, (x_max, y_min), (x_min, y_min), (0, 255, 0), 2)

        returning_data = []
        returning_data.append((x_min, y_min, x_max, y_max))

        if len(returning_data) == 0:
            return None

        return returning_data[0]

    #no need in another if-construction
    #get corners data by axes
    all_x = [corners[0, 0, 0, 0],
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
    # sort this data
    all_x.sort()
    all_y.sort()

    # set bbox axis data
    x_min = int(all_x[0]) - 5
    x_max = int(all_x[-1]) + 10
    y_min = int(all_y[0]) - 10
    y_max = int(all_y[-1]) + 10

    # draw lines by 4 points
    cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 255, 0), 2)
    cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 255, 0), 2)
    cv2.line(img, (x_max, y_max), (x_max, y_min), (0, 255, 0), 2)
    cv2.line(img, (x_max, y_min), (x_min, y_min), (0, 255, 0), 2)

    returning_data = []
    returning_data.append((x_min, y_min, x_max, y_max))

    if len(returning_data) == 0:
        return None

    return returning_data[0]


############ TASK 8 #############
#################################

############ TASK 9 #############
#################################

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
    
    if w / h >= 2:
        obj_x1 -= int(w * 0.5)
        obj_x2 += int(w * 0.5)
        obj_y1 -= int(h * 2)
        obj_y2 += int(h * 2)
    else:
        obj_x1 -= int(w * 0.25)
        obj_x2 += int(w * 0.25)
        obj_y1 -= int(h * 0.5)
        obj_y2 += int(h * 0.5)
    
    
    return obj_x1, obj_y1, obj_x2, obj_y2
    
############ TASK 10 ############
#################################

############ TASK 11 ############
#################################

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
    
    # fix width, height
    w = x2 - x1
    h = y2 - y1
    x1 -= int(w * 0.25)
    x2 += int(w * 0.25)
    y1 -= int(h * 0.1)
    y2 += int(h * 0.1)
    
    
    return x1, y1, x2, y2


############ TASK 12 ############
#################################

############ TASK 13 ############
#################################

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
    
    
        # fix width, height
    x1, y1, x2, y2 = necessary_objects[0]
    w = x2 - x1
    h = y2 - y1
    x1 -= int(w * 0.25)
    x2 += int(w * 0.25)
    y1 -= int(h * 0.1)
    y2 += int(h * 0.1)
    
    return x1, y1, x2, y2


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

############ TASK 14 ############
#################################

############ TASK 15 ############
#################################

############ TASK 16 ############
#################################

############ TASK 17 ############
#################################

############ TASK 18 ############
#################################

def find_id_18(img: np.ndarray):
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


############ TASK 19 ############
#################################

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

    return (x1, y1, x2, y2)

############ TASK 20 ############
#################################


if __name__ == '__main__':
    main()

