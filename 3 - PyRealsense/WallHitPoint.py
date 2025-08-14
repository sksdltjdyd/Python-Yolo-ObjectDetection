import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from collections import deque

### 벽 충돌 확인 테스트 ###

class BallWallCollisionTracker:
    def __init__(self):
        # RealSense D455 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)  # 높은 FPS
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 90)
        
        # 파이프라인 시작
        profile = self.pipeline.start(config)
        
        # 센서 설정 최적화
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4)  # High Density
        
        # YOLO 모델 (기본 또는 커스텀)
        self.model = YOLO('yolov8n.pt')  # 또는 학습된 'best.pt'
        
        # 깊이 필터
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # 벽 거리 (캘리브레이션)
        self.wall_distance = None
        self.wall_threshold = 50  # 벽으로부터 50mm 이내 = 충돌
        
        # 추적 데이터
        self.ball_trajectory = deque(maxlen=30)  # 최근 30프레임
        self.collision_points = []  # 충돌 지점 기록
        self.last_collision_time = 0
        self.collision_cooldown = 0.5  # 0.5초 쿨다운
        
        # 상태 변수
        self.is_calibrated = False
        self.show_depth = False
        self.show_trajectory = True
        self.recording = False
        
    def calibrate_wall(self):
        """벽 거리 캘리브레이션"""
        print("\n벽 캘리브레이션")
        print("1. 벽 확인")
        print("2. Enter를 누르면 3초 후 캘리브레이션 시작")
        input("준비되면 Enter...")
        
        print("3초 후 시작...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # 30프레임 수집해서 평균값 계산
        depth_values = []
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            # 필터 적용
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 중앙 영역 (200x200) 깊이값
            h, w = depth_image.shape
            center_region = depth_image[h//2-100:h//2+100, w//2-100:w//2+100]
            
            # 0이 아닌 값들의 중앙값
            valid_depths = center_region[center_region > 0]
            if len(valid_depths) > 0:
                depth_values.append(np.median(valid_depths))
        
        if depth_values:
            self.wall_distance = np.median(depth_values)
            self.is_calibrated = True
            print(f"✅ 벽 거리: {self.wall_distance:.0f}mm")
            return True
        
        print("캘리브레이션 실패")
        return False
    
    def detect_ball(self, frame):
        """YOLO로 공 검출"""
        results = self.model(frame, stream=True, conf=0.4, verbose=False)
        
        balls = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    # sports ball 클래스 또는 커스텀 ball 클래스
                    if self.model.names[cls] in ['sports ball', 'ball', 'tennis ball']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        balls.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (cx, cy),
                            'confidence': float(box.conf[0]),
                            'radius': int(max(x2-x1, y2-y1) / 2)
                        })
        
        # 가장 확실한 공 하나만 반환
        if balls:
            return max(balls, key=lambda x: x['confidence'])
        return None
    
    def check_collision(self, ball, depth_frame):
        """벽 충돌 감지"""
        if not self.is_calibrated or ball is None:
            return False, None
        
        cx, cy = ball['center']
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 경계 체크
        if not (0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]):
            return False, None
        
        # 공 중심 주변 영역의 깊이 (더 정확한 측정)
        radius = min(ball['radius'], 20)
        y1 = max(0, cy - radius)
        y2 = min(depth_image.shape[0], cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(depth_image.shape[1], cx + radius)
        
        # 공 영역의 깊이값들
        ball_region = depth_image[y1:y2, x1:x2]
        valid_depths = ball_region[ball_region > 0]
        
        if len(valid_depths) == 0:
            return False, None
        
        # 가장 가까운 깊이 (공의 앞면)
        ball_depth = np.min(valid_depths)
        
        # 벽과의 거리
        distance_to_wall = abs(ball_depth - self.wall_distance)
        
        # 충돌 판정
        current_time = time.time()
        if distance_to_wall < self.wall_threshold:
            # 쿨다운 체크 (연속 충돌 방지)
            if current_time - self.last_collision_time > self.collision_cooldown:
                self.last_collision_time = current_time
                
                # 충돌 지점 기록
                collision_data = {
                    'position': (cx, cy),
                    'depth': ball_depth,
                    'distance_to_wall': distance_to_wall,
                    'time': current_time,
                    'frame_trajectory': list(self.ball_trajectory)  # 충돌 전 궤적
                }
                self.collision_points.append(collision_data)
                
                return True, collision_data
        
        return False, None
    
    def update_trajectory(self, ball):
        """공 궤적 업데이트"""
        if ball:
            self.ball_trajectory.append(ball['center'])
    
    def calculate_velocity(self):
        """속도 벡터 계산"""
        if len(self.ball_trajectory) >= 2:
            p1 = self.ball_trajectory[-2]
            p2 = self.ball_trajectory[-1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            speed = np.sqrt(dx**2 + dy**2)
            
            return speed, (dx, dy)
        return 0, (0, 0)
    
    def draw_visualization(self, color_image, depth_colormap, ball, collision_data=None):
        """시각화"""
        vis_frame = color_image.copy()
        h, w = vis_frame.shape[:2]
        
        # 상태 정보 패널
        info_panel = np.zeros((h, 300, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)
        
        # 캘리브레이션 상태
        status_color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        cv2.putText(info_panel, f"Wall: {self.wall_distance:.0f}mm" if self.is_calibrated else "Not Calibrated",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 공 검출 및 깊이 정보
        if ball:
            cx, cy = ball['center']
            x1, y1, x2, y2 = ball['bbox']
            
            # 바운딩 박스
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis_frame, (cx, cy), 3, (0, 0, 255), -1)
            
            # 깊이 정보 표시
            if hasattr(self, 'last_depth'):
                depth_text = f"Depth: {self.last_depth:.0f}mm"
                cv2.putText(vis_frame, depth_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # 정보 패널에도 표시
                cv2.putText(info_panel, f"Ball Depth: {self.last_depth:.0f}mm",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if self.is_calibrated:
                    distance = abs(self.last_depth - self.wall_distance)
                    cv2.putText(info_panel, f"To Wall: {distance:.0f}mm",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 궤적 그리기
        if self.show_trajectory and len(self.ball_trajectory) > 1:
            pts = np.array(list(self.ball_trajectory), dtype=np.int32)
            for i in range(1, len(pts)):
                thickness = int(i / len(pts) * 5) + 1
                cv2.line(vis_frame, tuple(pts[i-1]), tuple(pts[i]), 
                        (0, 255, 255), thickness)
        
        # 속도 벡터
        speed, (dx, dy) = self.calculate_velocity()
        if speed > 2 and ball:  # 최소 속도 이상일 때만
            cx, cy = ball['center']
            end_point = (int(cx + dx * 3), int(cy + dy * 3))
            cv2.arrowedLine(vis_frame, (cx, cy), end_point, (255, 0, 0), 2)
            cv2.putText(info_panel, f"Speed: {speed:.1f} px/frame",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 충돌 감지!
        if collision_data:
            cx, cy = collision_data['position']
            
            # 충돌 지점 강조
            cv2.circle(vis_frame, (cx, cy), 20, (0, 0, 255), 3)
            cv2.circle(vis_frame, (cx, cy), 30, (0, 165, 255), 2)
            cv2.circle(vis_frame, (cx, cy), 40, (0, 255, 255), 2)
            
            # 충돌 텍스트
            cv2.putText(vis_frame, "COLLISION!", (cx-50, cy-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 충돌 정보
            cv2.putText(info_panel, "*** COLLISION DETECTED ***",
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(info_panel, f"Position: ({cx}, {cy})",
                       (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Distance: {collision_data['distance_to_wall']:.1f}mm",
                       (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 충돌 히스토리
        cv2.putText(info_panel, f"Total Hits: {len(self.collision_points)}",
                   (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # 최근 충돌 지점들 표시
        for i, cp in enumerate(self.collision_points[-5:]):  # 최근 5개
            px, py = cp['position']
            # 페이드 효과
            alpha = (i + 1) / 5.0
            color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
            cv2.circle(vis_frame, (px, py), 8, color, -1)
            cv2.circle(vis_frame, (px, py), 10, color, 1)
        
        # 컨트롤 안내
        cv2.putText(info_panel, "Controls:", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(info_panel, "[C] Calibrate", (10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "[D] Toggle Depth", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "[T] Toggle Trail", (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "[R] Reset Hits", (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "[S] Save Screenshot", (10, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "[Q] Quit", (10, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 프레임 결합
        if self.show_depth:
            display = np.hstack([vis_frame, depth_colormap, info_panel])
        else:
            display = np.hstack([vis_frame, info_panel])
        
        return display
    
    def save_collision_data(self):
        """충돌 데이터 저장"""
        if self.collision_points:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"collision_data_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Wall Distance: {self.wall_distance}mm\n")
                f.write(f"Total Collisions: {len(self.collision_points)}\n\n")
                
                for i, cp in enumerate(self.collision_points):
                    f.write(f"Collision #{i+1}:\n")
                    f.write(f"  Position: {cp['position']}\n")
                    f.write(f"  Depth: {cp['depth']}mm\n")
                    f.write(f"  Distance to wall: {cp['distance_to_wall']}mm\n")
                    f.write(f"  Time: {cp['time']:.2f}\n\n")
            
            print(f"📁 데이터 저장: {filename}")
    
    def run(self):
        """메인 실행 루프"""
        print("공-벽 충돌 포인트 트랙킹 시스템")
        print("먼저 'C'를 눌러 벽을 캘리브레이션하세요\n")
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                
                # 필터 적용
                depth_frame = frames.get_depth_frame()
                depth_frame = self.spatial.process(depth_frame)
                depth_frame = self.temporal.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
                
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 이미지 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 깊이 컬러맵
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 공 검출
                ball = self.detect_ball(color_image)
                
                # 깊이 정보 저장
                if ball:
                    cx, cy = ball['center']
                    if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                        self.last_depth = depth_image[cy, cx]
                
                # 충돌 체크
                collision_detected, collision_data = self.check_collision(ball, depth_frame)
                
                # 궤적 업데이트
                self.update_trajectory(ball)
                
                # 시각화
                display = self.draw_visualization(
                    color_image, depth_colormap, ball, 
                    collision_data if collision_detected else None
                )
                
                cv2.imshow('Ball-Wall Collision Tracker', display)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.calibrate_wall()
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                elif key == ord('t'):
                    self.show_trajectory = not self.show_trajectory
                elif key == ord('r'):
                    self.collision_points.clear()
                    print("✨ 충돌 기록 초기화")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", display)
                    self.save_collision_data()
                    print(f"📸 스크린샷 저장")
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BallWallCollisionTracker()
    tracker.run()