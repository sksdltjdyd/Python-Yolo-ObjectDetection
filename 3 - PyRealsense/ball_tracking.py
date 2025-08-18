import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def track_ball():
    # 카메라 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 시작
    profile = pipeline.start(config)
    
    # 필터
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    
    # YOLO 모델
    model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8n.pt')

    print("🎾 공 트랙킹 시작! (Q로 종료)")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # 필터 적용
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            
            # numpy 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # YOLO 검출
            results = model(color_image, stream=True, conf=0.5)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # sports ball 클래스만
                        if model.names[int(box.cls[0])] == 'sports ball':
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            
                            # 박스 그리기
                            cv2.rectangle(color_image, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            
                            # 깊이 표시
                            if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                                depth = depth_image[cy, cx]
                                cv2.putText(color_image, f"{depth}mm",
                                          (cx-30, cy-10),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, (255, 255, 0), 2)
            
            # 표시
            cv2.imshow('Ball Tracking', color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    track_ball()