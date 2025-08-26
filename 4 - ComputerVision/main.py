# Import required libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import pickle
import numpy as np
import pyrealsense2 as rs
import time
from pythonosc import udp_client # ✨ OSC 라이브러리 import

# --- 1. 기본 설정 ---
camWidth, camHeight = 640, 480
confidence = 0.70
yoloModelPath = "C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt"
classNames = ['ball']
calibrationFilePath = 'realsense_calibration_data.p' 
scale = 3 

# --- ✨ OSC 설정 ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
try:
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"✅ OSC client configured for {OSC_IP}:{OSC_PORT}")
except Exception as e:
    print(f"❌ Could not initialize OSC client: {e}")
    osc_client = None

# --- 2. 캘리브레이션 파일 로드 ---
try:
    with open(calibrationFilePath, 'rb') as fileObj:
        points = pickle.load(fileObj)
    print(f"✅ Calibration file '{calibrationFilePath}' loaded.")
except FileNotFoundError:
    print(f"❌ ERROR: Calibration file not found at '{calibrationFilePath}'")
    exit()

# --- 3. 출력 이미지 및 데이터 초기화 ---
imgOutput = np.zeros((1080, 1920, 3), np.uint8)
collision_points = [] 
tracked_ball_last_pos = None 
circles = np.zeros((4, 2), int)
counter = 0

# --- 4. RealSense 카메라 초기화 ---
pipeline = rs.pipeline()
config = rs.config()
# ✨ 뎁스 스트림을 추가로 활성화합니다.
config.enable_stream(rs.stream.depth, camWidth, camHeight, rs.format.z16, 60)
config.enable_stream(rs.stream.color, camWidth, camHeight, rs.format.bgr8, 60)
try:
    profile = pipeline.start(config)
    # ✨ 정렬 객체 및 뎁스 파라미터 획득
    align = rs.align(rs.stream.color)
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print("✅ RealSense camera started with Depth stream.")
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
    pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(imgMain, matrix, (width, height))
    return imgWarped

def detectObject(imgMain):
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
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        circles[counter] = (x, y)
        counter += 1
        print("Clicked points:", circles[:counter])

# ✨ 3D 좌표 변환 함수
def pixel_to_3d_point(x, y, depth_frame, intrinsics):
    if depth_frame is None or intrinsics is None: return None
    depth = depth_frame.get_distance(x, y)
    if depth == 0: return None # 거리가 측정되지 않으면 None 반환
    # 실제 3D 좌표 (미터 단위)
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point_3d

# --- 7. 메인 루프 ---
is_calibrating = False 
try:
    while True:
        # ✨ RealSense에서 정렬된 프레임 읽기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() # 뎁스 프레임 획득
        if not color_frame or not depth_frame:
            continue
        img = np.asanyarray(color_frame.get_data())

        if is_calibrating:
            # (캘리브레이션 로직은 이전과 동일)
            for i in range(counter):
                cv2.circle(img, (circles[i][0], circles[i][1]), 5, (0, 255, 0), cv2.FILLED)
            if counter == 4:
                points = circles.copy()
                with open(calibrationFilePath, 'wb') as fileObj:
                    pickle.dump(points, fileObj)
                print("✅ New calibration points saved!")
                is_calibrating = False
                counter = 0
        else:
            # 일반 추적 모드
            imgProjector = warpImage(img, points, camWidth, camHeight)
            # ✨ 투시 변환된 뎁스 프레임도 생성
            depth_image = np.asanyarray(depth_frame.get_data())
            depthProjector = warpImage(depth_image, points, camWidth, camHeight)
            # 뎁스 프레임을 OpenCV에서 다룰 수 있도록 래핑
            warped_depth_frame = rs.depth_frame(rs.frame(depthProjector))

            imgWithObjects, objects = detectObject(imgProjector)

            if objects:
                current_ball_center = objects[0]['center']
                tracked_ball_last_pos = current_ball_center
                
                # ✨ 현재 공의 3D 좌표 계산 및 OSC 전송
                point3d = pixel_to_3d_point(current_ball_center[0], current_ball_center[1], warped_depth_frame, depth_intrinsics)
                if point3d and osc_client:
                    # 주소: /ball/position, 값: x, y, z 좌표 (미터 단위)
                    osc_client.send_message("/ball/position", [float(p) for p in point3d])

            else:
                if tracked_ball_last_pos is not None:
                    collision_x = int(tracked_ball_last_pos[0] * scale)
                    collision_y = int(tracked_ball_last_pos[1] * scale)
                    collision_points.append((collision_x, collision_y))

                    # ✨ 충돌 지점의 3D 좌표 계산 및 OSC 전송
                    point3d = pixel_to_3d_point(tracked_ball_last_pos[0], tracked_ball_last_pos[1], warped_depth_frame, depth_intrinsics)
                    if point3d and osc_client:
                        # 주소: /ball/collision, 값: x, y, z 좌표 (미터 단위)
                        osc_client.send_message("/ball/collision", [float(p) for p in point3d])
                        print(f"💥 Collision Sent via OSC at: {point3d}")
                    
                    tracked_ball_last_pos = None

            for point in collision_points:
                cv2.circle(imgOutput, point, 15, (0, 0, 255), cv2.FILLED)

            cv2.imshow("Projector View (Warped)", imgWithObjects)
            cv2.imshow("Collision Wall", imgOutput)

        # (FPS 계산 및 공통 UI 로직은 이전과 동일)
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count; fps_frame_count = 0; fps_start_time = time.time()
        
        if is_calibrating:
            cv2.putText(img, f"CALIBRATING: Click {4 - counter} more points", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(img, "TRACKING MODE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Original Image (RealSense)", img)
        cv2.setMouseCallback("Original Image (RealSense)", mousePoints)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            imgOutput.fill(0); collision_points.clear(); print("✨ Collision points reset.")
        elif key == ord('c'):
            is_calibrating = True; counter = 0; circles.fill(0)
            print("\n🔄 CALIBRATION MODE: Please click 4 new points.")
        if key == ord('q'):
            break
finally:
    print("Stopping RealSense camera...")
    pipeline.stop()
    cv2.destroyAllWindows()