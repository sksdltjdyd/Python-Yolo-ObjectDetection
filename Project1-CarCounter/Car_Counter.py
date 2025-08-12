from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0) # Webcam
cap = cv2.VideoCapture("C:/Users/User/Documents/Git/PythonProject/venv310/Videos/cars.mp4") # Video

model = YOLO("C:/Users/User/Documents/Git/PythonProject/venv310/YoloWeights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("C:/Users/User/Documents/Git/PythonProject/venv310/Project1-CarCounter/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# 1. 비디오의 첫 프레임을 읽어서 크기 입력
success, img = cap.read()
if not success:
    print("비디오를 읽을 수 없습니다.")
else:
    # 2. 마스크의 크기를 비디오 프레임의 크기와 동일하게 조절
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # 3. 비디오를 처음부터 다시 읽기 위해 프레임 위치를 0으로 리셋
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while(True):
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2), (255, 0, 255), 3)
            w, h = x2-x1,y2-y1
            # x1, y1, w, h = box.xywh[0]
            # bbox = int(x1), int(y1), int(w), int(h)
           

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100

            #class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus"\
                or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0, x1), max(35, y1)), scale = 2, thickness=2, offset=3)
                # cvzone.cornerRect(img,(x1, y1, w, h),l=9,rt = 5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))



    resultTracker = tracker.update(detections)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        print(result)
        cvzone.cornerRect(img,(x1, y1, w, h),l=9,rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img,f'{id}',(max(0, x1), max(35, y1)), scale = 2, thickness=2, offset=10)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)