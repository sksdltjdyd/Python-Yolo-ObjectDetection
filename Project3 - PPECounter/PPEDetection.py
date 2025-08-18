from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Videos/ppe-1-1.mp4")  # For Video
 
 
model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Project3 - PPECounter/ppe.pt')
 
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
 
prev_frame_time = 0
new_frame_time = 0

mycolor = (0, 0, 255)
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            if conf>0.5:
                if currentclass == 'NO-Hardhat' or currentclass == 'NO-Safety Vest' or currentclass == 'NO-Mask':
                    mycolor = (0,0,255)
                elif currentclass == 'Hardhat' or currentclass == 'Safety Vest' or currentclass == 'Mask':
                    mycolor = (0,255,0)
                else:
                    mycolor = (255,0,0)
 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                               (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=mycolor,
                               colorT=(255, 255, 255), colorR=mycolor, offset=5)
            
            # cvzone.cornerRect(img, (x1, y1, w, h))
            cv2.rectangle(img, (x1,y1), (x2, y2), mycolor, 3)
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)