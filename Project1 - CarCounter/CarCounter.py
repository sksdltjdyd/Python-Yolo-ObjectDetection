from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Videos/cars.mp4")  # For Video
 
 
model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8l.pt')
 
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

mask = cv2.imread("C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Project1 - CarCounter/mask.png")

success, img = cap.read()
if not success:
    print("비디오를 읽을 수 없습니다.")
else:
    # 2. 마스크의 크기를 비디오 프레임의 크기와 동일하게 조절합니다.
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # 3. 비디오를 처음부터 다시 읽기 위해 프레임 위치를 0으로 리셋합니다.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

prev_frame_time = 0
new_frame_time = 0
 
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
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
            currentClass = classNames[cls]
            if currentClass == "car" or  currentClass == "truck" or currentClass == "bus"\
                or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                                   scale=1, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
 
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)