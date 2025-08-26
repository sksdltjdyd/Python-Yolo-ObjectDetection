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
calibrationFilePath = 'realsense_calibration_data.p' 
scale = 3 

# --- OSC 설정 ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
try:
    from pythonosc import udp_client
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"✅ OSC client configured for {OSC_IP}:{OSC_PORT}")
except ImportError:
    print("⚠️ python-osc not found. Skipping OSC.")
    osc_client = None
except Exception as e:
    print(f"❌ Could not initialize OSC client: {e}")
    osc_client = None

# --- 2. 캘리브레이션 파일 로드 및 행렬 계산 ---
try:
    with open(calibrationFilePath, 'rb') as fileObj:
        points = pickle.load(fileObj)
    print(f"✅ Calibration file '{calibrationFilePath}' loaded.")
    
    # ✨ 정방향 및 역방향 투시 변환 행렬을 미리 계산
    pts1 = np.float32([points[0], points[1], points[2], points[3]])
    pts2 = np.float32([[0, 0], [camWidth, 0], [0, camHeight], [camWidth, camHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inverse_matrix = cv2.getPerspectiveTransform(pts2, pts1) # 역변환 행렬

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
config.enable_stream(rs.stream.depth, camWidth, camHeight, rs.format.z16, 60)
config.enable_stream(rs.stream.color, camWidth, camHeight, rs.format.bgr8, 60)
try:
    profile = pipeline.start(config)
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
        circles[counter] = (x, y); counter += 1
        print("Clicked points:", circles[:counter])

def pixel_to_3d_point(x, y, depth_frame, intrinsics):
    if depth_frame is None or intrinsics is None: return None
    # ✨ get_distance는 픽셀 좌표를 정수로 요구함
    depth = depth_frame.get_distance(int(x), int(y))
    if depth == 0: return None
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point_3d

# --- 7. 메인 루프 ---
is_calibrating = False 
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame: continue
            
        img = np.asanyarray(color_frame.get_data())

        if is_calibrating:
            # (캘리브레이션 로직은 이전과 동일)
            for i in range(counter):
                cv2.circle(img, (circles[i][0], circles[i][1]), 5, (0, 255, 0), cv2.FILLED)
            if counter == 4:
                points = circles.copy()
                with open(calibrationFilePath, 'wb') as fileObj: pickle.dump(points, fileObj)
                # ✨ 행렬들을 다시 계산
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
                pts2 = np.float32([[0, 0], [camWidth, 0], [0, camHeight], [camWidth, camHeight]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                inverse_matrix = cv2.getPerspectiveTransform(pts2, pts1)
                print("✅ New calibration points saved and matrices recalculated!")
                is_calibrating = False
                counter = 0
        else:
            # 일반 추적 모드
            imgProjector = cv2.warpPerspective(img, matrix, (camWidth, camHeight))
            imgWithObjects, objects = detectObject(imgProjector)

            if objects:
                warped_center = objects[0]['center']
                tracked_ball_last_pos = warped_center
                
                # ✨ 역변환으로 원본 좌표 찾기
                warped_center_np = np.array([[warped_center]], dtype=np.float32)
                original_center_np = cv2.perspectiveTransform(warped_center_np, inverse_matrix)
                original_cx, original_cy = original_center_np[0][0]

                # ✨ 원본 좌표와 원본 뎁스 프레임으로 3D 위치 계산 및 OSC 전송
                point3d = pixel_to_3d_point(original_cx, original_cy, depth_frame, depth_intrinsics)
                if point3d and osc_client:
                    osc_client.send_message("/ball/position", [float(p) for p in point3d])

            else:
                if tracked_ball_last_pos is not None:
                    collision_x_scaled = int(tracked_ball_last_pos[0] * scale)
                    collision_y_scaled = int(tracked_ball_last_pos[1] * scale)
                    collision_points.append((collision_x_scaled, collision_y_scaled))

                    # ✨ 충돌 지점도 역변환하여 3D 좌표 계산 및 OSC 전송
                    last_pos_np = np.array([[tracked_ball_last_pos]], dtype=np.float32)
                    original_last_pos_np = cv2.perspectiveTransform(last_pos_np, inverse_matrix)
                    original_lx, original_ly = original_last_pos_np[0][0]
                    
                    point3d = pixel_to_3d_point(original_lx, original_ly, depth_frame, depth_intrinsics)
                    if point3d and osc_client:
                        osc_client.send_message("/ball/collision", [float(p) for p in point3d])
                        print(f"💥 Collision Sent via OSC at: {point3d}")
                    
                    tracked_ball_last_pos = None

            for point in collision_points:
                cv2.circle(imgOutput, point, 15, (0, 0, 255), cv2.FILLED)

            cv2.imshow("Projector View (Warped)", imgWithObjects)
            cv2.imshow("Collision Wall", imgOutput)

        # (FPS 및 UI 로직은 이전과 동일)
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