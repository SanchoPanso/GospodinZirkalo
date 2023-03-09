import torch
import cv2
import argparse
from ultralytics import YOLO
from typing import Tuple


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
conf_thresh = 0.4
classes = [
    'green_tube',
    'yellow_ball',
    'blue_ball',
    'three_color_ball',
    'wood_cube',
    'glass_cube',
    'uhi',
    'greenhouse',
    'duck',
    'vase',
    'bobina',
    'elect',
    'manupulator',
    'tank',
    'jump_ball',
]


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
    args = parse_args()
    cam_id = int(args.cam)
    
    cap = cv2.VideoCapture(cam_id)
    model = YOLO(args.weights)
    
    while True:
        # Read img
        ret, img = cap.read()
        
        # If img is not captured from webcam then skip it
        if ret is False:
            print('Image is not captured')
            continue
        
        # Neural network inference
        results = model(img, conf=conf_thresh)
        
        boxes = results[0].boxes
        
        # Draw found bboxes
        for box in boxes:
            xyxy = box.xyxy
            cls_id = int(box.cls.item())
            name = cls_id if cls_id >= len(classes) else classes[cls_id]
            conf = float(box.conf.item())
            
            x1, y1, x2, y2 = map(int, xyxy[0].tolist())
            color = compute_color_for_labels(cls_id)
            text = "{0} {1:.3f}".format(name, conf)
            
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), color, -1)
            
            cv2.putText(img, text, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            
        
        cv2.imshow('img', img)
        
        # If 'Esc' is pressed, then break
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()
