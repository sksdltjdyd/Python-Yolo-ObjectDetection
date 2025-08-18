import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. 리얼센스 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트리밍 시작 및 Align 객체 생성
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# YOLO 모델 로드
model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8n.pt')

# 2. 실제 공간에 벽 정의 (카메라로부터 3미터 앞에)
WALL_DISTANCE_Z = 3.0  # meters

# 이전 프레임의 객체 상태 저장 (충돌 감지용)
was_further = True

try:
    while True:
        # 프레임 대기 및 정렬
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # 카메라 파라미터(Intrinsics) 가져오기 (3D 좌표 변환에 필요)
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 3. YOLO 객체 탐지 (컬러 이미지 사용)
        results = model(color_image, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 사람(0) 또는 스포츠 공(32) 타겟
                cls = int(box.cls[0])
                if cls == 0 or cls == 32:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # 4. 객체의 3D 좌표 획득
                    depth_in_meters = depth_frame.get_distance(cx, cy)
                    
                    # 깊이 값이 0 이상일 때만 처리 (측정 실패 제외)
                    if depth_in_meters > 0:
                        # 2D 픽셀과 깊이 값으로 3D 좌표 계산
                        real_world_coords = rs.rs2_deproject_pixel_to_point(
                            depth_intrinsics, [cx, cy], depth_in_meters
                        )
                        object_z = real_world_coords[2]

                        # 5. 3D 공간에서 충돌 감지
                        is_further_now = object_z > WALL_DISTANCE_Z
                        
                        # 이전 프레임에선 벽보다 멀리 있었는데, 지금은 더 가까워졌다면 '충돌'
                        if was_further and not is_further_now:
                            print(f"Collision Detected! Object at Z = {object_z:.2f}m")
                            # 충돌 지점에 시각적 피드백
                            cv2.circle(color_image, (cx, cy), 15, (0, 255, 0), -1)

                        was_further = is_further_now

                        # 화면에 객체의 실제 거리(Z) 표시
                        cv2.putText(color_image, f"Z: {object_z:.2f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        cv2.imshow('RealSense with YOLO 3D Collision', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()