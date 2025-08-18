from ultralytics import YOLO
import cv2
 
model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8n.pt')
results = model("C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/1 - Running Yolo/Images/Bus.png", show=True)
cv2.waitKey(0)