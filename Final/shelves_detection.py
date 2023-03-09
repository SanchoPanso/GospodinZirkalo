import torch
import cv2
import argparse
from ultralytics import YOLO
from typing import Tuple

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='weights path')
    parser.add_argument('--cam', type=str, default='1', help='webcam id')
    
    return parser.parse_args()


def compute_color_for_labels(label: int) -> Tuple[int]:
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def main():
    cap = cv2.VideoCapture(1)
    args = parse_args()
    model = YOLO(args.weights)
    
    while True:
        ret, img = cap.read()
        results = model(img)
        
        boxes = results[0].boxes
        
        for box in boxes:
            xyxy = box.xyxy
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            
            x1, y1, x2, y2 = map(int, xyxy[0].tolist())
            color = compute_color_for_labels(cls_id)
            text = "{0} {1:.3f}".format(cls_id, conf)
            
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), color, -1)
            
            cv2.putText(img, text, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            
        
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()
