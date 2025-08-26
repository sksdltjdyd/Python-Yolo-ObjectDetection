# Import required libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import pickle
import numpy as np
import pyrealsense2 as rs
import time

# --- 1. 기본 설정 ---
camWidth, camHeight = 640, 480
confidence = 0.70
yoloModelPath = "C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt"
classNames = ['ball']
mode = "circle"
calibrationFilePath = 'realsense_calibration_data.p' 
scale = 3
camFPS = 60
ballArea = {"min": 1300, "max": 2500}

# --- 2. 캘리브레이션 파일 로드 ---
try:
    with open(calibrationFilePath, 'rb') as fileObj:
        points = pickle.load(fileObj)
    print(f"✅ Calibration file '{calibrationFilePath}' loaded.")
except FileNotFoundError:
    print(f"❌ ERROR: Calibration file not found at '{calibrationFilePath}'")
    print("Please run the calibration script first to generate the file.")
    exit()

# --- 3. 출력 이미지 및 데이터 초기화 ---
imgOutput = np.zeros((1080, 1920, 3), np.uint8)
collision_points = [] 
tracked_ball_last_pos = None 

# 캘리브레이션용 변수를 별도로 관리
circles = np.zeros((4, 2), int)
counter = 0

# --- 4. RealSense 카메라 초기화 ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, camWidth, camHeight, rs.format.bgr8, 60)
try:
    pipeline.start(config)
    print("✅ RealSense camera started.")
except Exception as e:
    print(f"❌ Failed to start RealSense camera: {e}")
    exit()

# --- 5. YOLO 모델 로드 및 FPS 계산 변수 ---
model = YOLO(yoloModelPath)
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

# --- 6. 함수 정의 ---
def warpImage(imgMain, circles, width, height):
    """이미지를 투시 변환하여 위에서 본 것처럼 만듭니다."""
    pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(imgMain, matrix, (width, height))
    return imgWarped

def detectObject(imgMain):
    """YOLO를 사용하여 이미지에서 객체를 탐지합니다."""
    results = model(imgMain, stream=False, verbose=False, conf=confidence)
    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            cv2.rectangle(imgMain, (x1, y1), (x2, y2), (0, 255, 0), 2)
            objects.append({'center': center})
    return imgMain, objects

def mousePoints(event, x, y, flags, param):
    """캘리브레이션 재설정 모드를 위한 마우스 콜백 함수"""
    global counter
    # 4개 이상의 점이 찍히지 않도록 방지
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        circles[counter] = (x, y)
        counter += 1
        print("Clicked points:")
        print(circles[:counter])


# --- 7. 메인 루프 ---
# 캘리브레이션 모드 플래그 추가
is_calibrating = False 

try:
    while True:
        # RealSense 프레임 읽기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())

        # 캘리브레이션 모드일 때와 아닐 때를 분리
        if is_calibrating:
            # 캘리브레이션 중에는 점만 찍고 표시
            for i in range(counter):
                cv2.circle(img, (circles[i][0], circles[i][1]), 5, (0, 255, 0), cv2.FILLED)
            
            if counter == 4:
                # 4개의 점이 다 찍히면, 새로운 좌표를 points 변수에 저장하고 모드 종료
                points = circles.copy()
                with open(calibrationFilePath, 'wb') as fileObj:
                    pickle.dump(points, fileObj)
                print("✅ New calibration points saved!")
                is_calibrating = False
                counter = 0 # 카운터 초기화
        else:
            # 캘리브레이션 모드가 아닐 때 (일반 추적 모드)
            imgProjector = warpImage(img, points, camWidth, camHeight)
            imgWithObjects, objects = detectObject(imgProjector)

            if objects:
                current_ball_center = objects[0]['center']
                tracked_ball_last_pos = current_ball_center
            else:
                if tracked_ball_last_pos is not None:
                    collision_x = int(tracked_ball_last_pos[0] * scale)
                    collision_y = int(tracked_ball_last_pos[1] * scale)
                    collision_points.append((collision_x, collision_y))
                    tracked_ball_last_pos = None

            for point in collision_points:
                cv2.circle(imgOutput, point, 15, (0, 0, 255), cv2.FILLED)

            cv2.imshow("Projector View (Warped)", imgWithObjects)
            cv2.imshow("Collision Wall", imgOutput)

        # 공통 로직: FPS 계산 및 원본 이미지 표시
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # 현재 모드에 대한 안내 문구 표시
        if is_calibrating:
            cv2.putText(img, f"CALIBRATING: Click {4 - counter} more points", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(img, "TRACKING MODE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Original Image (RealSense)", img)
        cv2.setMouseCallback("Original Image (RealSense)", mousePoints)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            imgOutput.fill(0)
            collision_points.clear()
            print("✨ Collision points reset.")
        
        # 'c' 키를 눌러 캘리브레이션 모드 시작/리셋
        elif key == ord('c'):
            is_calibrating = True
            counter = 0
            circles.fill(0)
            print("\n🔄 CALIBRATION MODE: Please click 4 new points on the 'Original Image' window.")
            print("Order: Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right")

        if key == ord('q'):
            break
finally:
    print("Stopping RealSense camera...")
    pipeline.stop()
    cv2.destroyAllWindows()