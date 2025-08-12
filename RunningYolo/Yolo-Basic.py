from ultralytics import YOLO
import cv2

model = YOLO('../YoloWeights/yolov8l.pt')
results = model("RunningYolo/Images/1.jpg",show=True)
cv2.waitKey(0)