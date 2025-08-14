import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

class YOLOBallTracker:
    def __init__(self, model_path=None):
        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        self.pipeline.start(config)
        
        # YOLO 모델 로드
        if model_path:
            self.model = YOLO(model_path)  # 커스텀 학습 모델
        else:
            self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8n.pt')  # 기본 모델
            print("⚠️ 기본 YOLO 모델 사용 중 - sports ball 클래스 감지")
        
        # 추적 데이터
        self.tracker_data = {}
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
    def train_custom_model(self):
        """커스텀 스펀지 공 모델 학습용 코드"""
        print("""
        커스텀 모델 학습 방법:
        1. 공 이미지 100-200장 촬영
        2. Roboflow나 LabelImg로 라벨링
        3. 아래 코드로 학습:
        
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.train(
            data='path/to/dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='sponge_ball'
        )
        """)
    
    def process_frame(self, frame, depth_frame):
        """YOLO 추론 실행"""
        # FPS 계산
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
        
        # YOLO 추론
        results = self.model(frame, stream=True, conf=0.3, verbose=False)
        
        detections = []
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 클래스 필터링 (sports ball = 32 in COCO)
                    cls = int(box.cls[0])
                    
                    # 커스텀 모델이면 모든 클래스, 아니면 sports ball만
                    if self.model.names[cls] == 'sports ball' or 'ball' in self.model.names[cls].lower():
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        width, height = int(x2 - x1), int(y2 - y1)
                        
                        # 깊이 정보 추가
                        depth_value = 0
                        if depth_image is not None and 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                            depth_value = depth_image[cy, cx]
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (cx, cy),
                            'size': (width, height),
                            'confidence': conf,
                            'depth': depth_value,
                            'class': self.model.names[cls]
                        })
        
        return detections
    
    def update_tracking(self, detections):
        """추적 ID 업데이트 (간단한 nearest neighbor)"""
        # 이전 프레임과 매칭
        for det in detections:
            min_dist = float('inf')
            matched_id = None
            
            for track_id, track_data in self.tracker_data.items():
                if 'last_pos' in track_data:
                    dist = np.linalg.norm(
                        np.array(det['center']) - np.array(track_data['last_pos'])
                    )
                    if dist < min_dist and dist < 50:  # 50픽셀 이내
                        min_dist = dist
                        matched_id = track_id
            
            if matched_id:
                # 기존 추적 업데이트
                self.tracker_data[matched_id]['positions'].append(det['center'])
                self.tracker_data[matched_id]['last_pos'] = det['center']
                self.tracker_data[matched_id]['last_seen'] = self.frame_count
                det['track_id'] = matched_id
            else:
                # 새 추적 시작
                new_id = len(self.tracker_data)
                self.tracker_data[new_id] = {
                    'positions': [det['center']],
                    'last_pos': det['center'],
                    'last_seen': self.frame_count
                }
                det['track_id'] = new_id
        
        return detections
    
    def visualize(self, frame, detections):
        """결과 시각화"""
        vis_frame = frame.copy()
        
        # FPS 표시
        cv2.putText(vis_frame, f"FPS: {self.fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            
            # 바운딩 박스
            color = (0, 255, 0) if det['confidence'] > 0.5 else (0, 255, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 중심점
            cv2.circle(vis_frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # 정보 텍스트
            label = f"{det.get('track_id', '?')} | {det['confidence']:.2f}"
            if det['depth'] > 0:
                label += f" | {det['depth']}mm"
            
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 궤적 그리기
            if 'track_id' in det and det['track_id'] in self.tracker_data:
                positions = self.tracker_data[det['track_id']]['positions']
                if len(positions) > 1:
                    for i in range(1, min(len(positions), 20)):
                        alpha = i / 20.0
                        pt1 = positions[-i]
                        pt2 = positions[-i-1] if i < len(positions)-1 else positions[-i]
                        cv2.line(vis_frame, pt1, pt2, 
                                (int(255*alpha), int(255*alpha), 0), 2)
        
        # 검출 수 표시
        cv2.putText(vis_frame, f"Balls: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_frame
    
    def run(self):
        """메인 루프"""
        print("YOLO 공 트랙킹 시작")
        print("'q' - 종료 / 's' - 스크린샷")
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # YOLO 검출
                detections = self.process_frame(color_image, depth_frame)
                
                # 추적 업데이트
                detections = self.update_tracking(detections)
                
                # 시각화
                vis_frame = self.visualize(color_image, detections)
                
                cv2.imshow('YOLO Ball Tracking', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'screenshot_{time.time()}.jpg', vis_frame)
                    print("📸 스크린샷 저장됨")
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 색상 기반 트랙킹 (빠른 테스트)
    # tracker = SpongeBallTracker()
    
    # YOLO 기반 트랙킹 (더 정확)
    tracker = YOLOBallTracker()
    
    tracker.run()