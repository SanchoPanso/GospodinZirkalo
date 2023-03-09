import cv2
import numpy as np


def match_puzzles(cnts1: np.ndarray, cnts2: np.ndarray) -> dict:
    conformity = {}
    
    for i, cnt1 in enumerate(cnts1):
        metrics = []
        
        for j, cnt2 in enumerate(cnts2):
            metric = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0)
            metrics.append(metric)
            
        metrics = np.array(metrics)
        min_idx = metrics.argmin()
        
        conformity[i] = min_idx
    
    return conformity


def get_contours(input_img: np.ndarray):

    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    couners_finded, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return couners_finded


def draw_conformity(img1: np.ndarray, img2: np.ndarray, 
                    conformity: dict, 
                    cnts1: np.ndarray, cnts2: np.ndarray):
    for i in conformity:
        j = conformity[i]
        cnt1 = cnts1[i]
        cnt2 = cnts2[j]
        
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        
        cv2.putText(img1, str(i), (x1 + w1 // 2, y1 + h1 // 2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img2, str(i), (x2  + w2 // 2, y2 + h2 // 2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


def main():
    img1 = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\image_2023-03-08_16-11-38 (3).png')
    img2 = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\image_2023-03-08_16-11-39 (1).png')
    
    cnts1 = get_contours(img1)
    cnts2 = get_contours(img2)
    
    conformity = match_puzzles(cnts1, cnts2)
    draw_conformity(img1, img2, conformity, cnts1, cnts2)
    
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    
    cv2.waitKey()
    

if __name__ == '__main__':
    main()
    
