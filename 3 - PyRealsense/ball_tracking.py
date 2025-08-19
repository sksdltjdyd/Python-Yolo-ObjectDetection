import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
from collections import deque
from pythonosc import udp_client

class UltimateDepthBallTracker:
    def __init__(self):
        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = self.pipeline.start(config)
        
        # 카메라 내부 파라미터
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # 정렬 객체
        self.align = rs.align(rs.stream.color)
        
        # 필터
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # YOLO 모델
        self.model = YOLO('yolov8n.pt')
        
        # OSC (Unreal 통신) - 선택적
        try:
            self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        except:
            self.osc_client = None
            print("OSC 연결 실패 - 계속 진행")
        
        # 벽 설정
        self.wall_distance = None  # 캘리브레이션 필요
        self.collision_threshold = 50  # 50mm
        
        # 추적 데이터 (각 공별)
        self.balls = {}
        self.next_ball_id = 0
        
        # 충돌 기록
        self.collisions = []
        self.last_collision_time = 0
        
    def calibrate_wall(self):
        """벽 거리 자동 캘리브레이션"""
        print("벽 캘리브레이션 - 3초 후 시작...")
        time.sleep(3)
        
        depths = []
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            # 필터 적용
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 중앙 영역 깊이
            h, w = depth_image.shape
            center_region = depth_image[h//2-50:h//2+50, w//2-50:w//2+50]
            valid_depths = center_region[center_region > 0]
            
            if len(valid_depths) > 0:
                depths.append(np.median(valid_depths))
        
        if depths:
            self.wall_distance = np.median(depths)
            print(f"✅ 벽 거리: {self.wall_distance:.0f}mm")
            return True
        return False
    
    def pixel_to_3d_point(self, x, y, depth):
        """2D 픽셀을 3D 좌표로 변환 - numpy 배열 반환"""
        point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics, [x, y], depth
        )
        return np.array(point)  # ⚠️ 중요: numpy 배열로 반환
    
    def calculate_3d_velocity(self, positions, timestamps):
        """실제 3D 속도 계산 (m/s) - 수정된 버전"""
        if len(positions) < 2:
            return 0, np.array([0, 0, 0])
        
        # ⚠️ 중요: deque를 numpy 배열로 변환
        positions_array = np.array(list(positions))
        timestamps_array = np.array(list(timestamps))
        
        # 스무딩을 위해 최근 3개 위치 사용
        if len(positions_array) >= 3:
            p1 = np.mean(positions_array[-3:-1], axis=0)
            p2 = positions_array[-1]
            dt = timestamps_array[-1] - timestamps_array[-3]
        else:
            p1 = positions_array[-2]
            p2 = positions_array[-1]
            dt = timestamps_array[-1] - timestamps_array[-2]
        
        if dt > 0:
            # numpy 배열이므로 연산 가능
            velocity_vector = (p2 - p1) / dt / 1000  # mm/s → m/s
            speed = np.linalg.norm(velocity_vector)
            return speed, velocity_vector
        
        return 0, np.array([0, 0, 0])
    
    def predict_collision(self, ball_data):
        """벽 충돌 예측"""
        if len(ball_data['positions']) < 2:
            return None
        
        # 현재 위치 (numpy 배열)
        positions_array = np.array(list(ball_data['positions']))
        current_pos = positions_array[-1]
        
        # 속도 계산
        speed, velocity = self.calculate_3d_velocity(
            ball_data['positions'], 
            ball_data['timestamps']
        )
        
        # Z축 속도 (벽 방향)
        if velocity[2] <= 0.01:  # 벽으로 움직이지 않음
            return None
        
        # 벽까지 시간 계산
        if self.wall_distance:
            time_to_wall = (self.wall_distance - current_pos[2]) / (velocity[2] * 1000)
            
            if 0 < time_to_wall < 2:  # 2초 이내 충돌 예상
                # 충돌 예상 위치
                hit_x = current_pos[0] + velocity[0] * 1000 * time_to_wall
                hit_y = current_pos[1] + velocity[1] * 1000 * time_to_wall
                
                return {
                    'time_to_impact': time_to_wall,
                    'impact_position': (hit_x, hit_y),
                    'impact_velocity': speed,
                    'confidence': min(1.0, 2.0 - time_to_wall)
                }
        
        return None
    
    def detect_collision(self, ball_data):
        """실제 충돌 감지"""
        if len(ball_data['positions']) < 3:
            return False, None
        
        # numpy 배열로 변환
        positions_array = np.array(list(ball_data['positions']))
        current_depth = positions_array[-1][2]
        
        # 1. 벽 근접 체크
        if self.wall_distance and abs(current_depth - self.wall_distance) < self.collision_threshold:
            
            # 2. 속도 변화 체크 (충돌 시 급변)
            v1 = positions_array[-2][2] - positions_array[-3][2]
            v2 = positions_array[-1][2] - positions_array[-2][2]
            
            # 감속 또는 반대 방향
            if v1 > 0 and v2 <= 0:  # 벽으로 가다가 멈추거나 튕김
                
                # 충돌 강도 계산
                speed, _ = self.calculate_3d_velocity(
                    ball_data['positions'], 
                    ball_data['timestamps']
                )
                
                collision_data = {
                    'ball_id': ball_data['id'],
                    'position': positions_array[-1].tolist(),  # 리스트로 변환
                    'impact_speed': speed,
                    'timestamp': time.time()
                }
                
                return True, collision_data
        
        return False, None
    
    def track_ball(self, detection, depth_image, current_time):
        """개별 공 추적"""
        cx, cy = detection['center']
        
        # 깊이 값 (중심 주변 평균)
        roi_size = 5
        y1 = max(0, cy - roi_size)
        y2 = min(depth_image.shape[0], cy + roi_size)
        x1 = max(0, cx - roi_size)
        x2 = min(depth_image.shape[1], cx + roi_size)
        
        roi_depths = depth_image[y1:y2, x1:x2]
        valid_depths = roi_depths[roi_depths > 0]
        
        if len(valid_depths) == 0:
            return None
        
        depth = np.median(valid_depths)
        
        # 3D 위치 계산 (numpy 배열로)
        point_3d = self.pixel_to_3d_point(cx, cy, depth)
        
        # 가장 가까운 기존 공 찾기
        matched_id = self.match_ball(point_3d)
        
        if matched_id is None:
            # 새 공 생성
            matched_id = self.next_ball_id
            self.next_ball_id += 1
            
            self.balls[matched_id] = {
                'id': matched_id,
                'positions': deque(maxlen=30),
                'timestamps': deque(maxlen=30),
                'pixel_positions': deque(maxlen=30),
                'color': tuple(np.random.randint(0, 255, 3).tolist())
            }
        
        # 데이터 업데이트
        ball = self.balls[matched_id]
        ball['positions'].append(point_3d)  # numpy 배열 저장
        ball['timestamps'].append(current_time)
        ball['pixel_positions'].append((cx, cy))
        ball['last_seen'] = current_time
        ball['current_depth'] = depth
        
        return matched_id
    
    def match_ball(self, point_3d, threshold=200):
        """3D 거리 기반 공 매칭"""
        min_dist = float('inf')
        matched_id = None
        
        for ball_id, ball in self.balls.items():
            if len(ball['positions']) > 0:
                # 마지막 위치와 3D 거리
                last_pos = np.array(ball['positions'][-1])  # numpy 배열로 변환
                dist = np.linalg.norm(point_3d - last_pos)
                
                # 시간 차이 고려 (오래된 공은 매칭 안함)
                time_diff = time.time() - ball.get('last_seen', 0)
                if time_diff < 1.0 and dist < min_dist and dist < threshold:
                    min_dist = dist
                    matched_id = ball_id
        
        return matched_id
    
    def visualize_tracking(self, color_image, depth_image):
        """추적 결과 시각화"""
        vis_image = color_image.copy()
        h, w = vis_image.shape[:2]
        
        # 정보 패널
        info_panel = np.zeros((h, 400, 3), dtype=np.uint8)
        info_panel[:] = (30, 30, 30)
        
        # 벽 거리 표시
        if self.wall_distance:
            cv2.putText(info_panel, f"Wall: {self.wall_distance:.0f}mm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(info_panel, "Wall: Not Calibrated", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 각 공 처리
        y_offset = 60
        for ball_id, ball in self.balls.items():
            if len(ball['positions']) < 1:
                continue
                
            # 최근 본 시간 체크
            if time.time() - ball.get('last_seen', 0) > 1.0:
                continue
            
            # 3D 정보
            current_pos = np.array(ball['positions'][-1])  # numpy 배열로
            speed, velocity = self.calculate_3d_velocity(
                ball['positions'], 
                ball['timestamps']
            )
            
            # 픽셀 위치
            if len(ball['pixel_positions']) > 0:
                px, py = ball['pixel_positions'][-1]
                
                # 공 표시
                cv2.circle(vis_image, (px, py), 15, ball['color'], 2)
                
                # ID 표시
                cv2.putText(vis_image, f"ID:{ball_id}", 
                           (px - 20, py - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball['color'], 2)
                
                # 속도 벡터 그리기
                if speed > 0.1:
                    end_x = int(px + velocity[0] * 100)
                    end_y = int(py + velocity[1] * 100)
                    cv2.arrowedLine(vis_image, (px, py), (end_x, end_y),
                                   (255, 0, 0), 2, tipLength=0.3)
                
                # 3D 정보 표시
                info_text = f"Depth: {current_pos[2]:.0f}mm"
                cv2.putText(vis_image, info_text, (px - 30, py + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 궤적 그리기
            if len(ball['pixel_positions']) > 1:
                pts = np.array(list(ball['pixel_positions']), dtype=np.int32)
                for i in range(1, len(pts)):
                    thickness = int(i / len(pts) * 3) + 1
                    cv2.line(vis_image, tuple(pts[i-1]), tuple(pts[i]),
                            ball['color'], thickness)
            
            # 충돌 예측
            collision_pred = self.predict_collision(ball)
            if collision_pred:
                if len(ball['pixel_positions']) > 0:
                    px, py = ball['pixel_positions'][-1]
                    cv2.putText(vis_image, 
                               f"Impact in {collision_pred['time_to_impact']:.1f}s",
                               (px - 40, py - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 충돌 감지
            is_collision, collision_data = self.detect_collision(ball)
            if is_collision:
                if len(ball['pixel_positions']) > 0:
                    px, py = ball['pixel_positions'][-1]
                    # 충돌 이펙트
                    cv2.circle(vis_image, (px, py), 30, (0, 0, 255), 3)
                    cv2.putText(vis_image, "COLLISION!", (px - 40, py - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # OSC 전송
                if self.osc_client:
                    self.send_collision_to_unreal(collision_data)
                
                # 기록
                if time.time() - self.last_collision_time > 0.5:  # 중복 방지
                    self.collisions.append(collision_data)
                    self.last_collision_time = time.time()
            
            # 정보 패널에 공 정보 추가
            cv2.putText(info_panel, f"Ball {ball_id}:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball['color'], 1)
            cv2.putText(info_panel, f"  Speed: {speed:.2f} m/s", 
                       (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(info_panel, f"  Pos: X:{current_pos[0]:.0f} Y:{current_pos[1]:.0f} Z:{current_pos[2]:.0f}", 
                       (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y_offset += 70
        
        # 충돌 히스토리
        cv2.putText(info_panel, f"Total Collisions: {len(self.collisions)}", 
                   (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 컨트롤 안내
        cv2.putText(info_panel, "[C] Calibrate | [R] Reset | [Q] Quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # 깊이 컬러맵
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # 결합
        combined = np.hstack([vis_image, depth_colormap, info_panel])
        
        return combined
    
    def send_collision_to_unreal(self, collision_data):
        """Unreal Engine으로 충돌 데이터 전송"""
        if self.osc_client:
            try:
                self.osc_client.send_message("/ball/collision", [
                    collision_data['ball_id'],
                    collision_data['position'][0],  # X
                    collision_data['position'][1],  # Y
                    collision_data['position'][2],  # Z
                    collision_data['impact_speed'],
                    collision_data['timestamp']
                ])
            except:
                pass
    
    def run(self):
        """메인 실행 루프"""
        print("=" * 50)
        print("Ultimate Depth Ball Tracker")
        print("=" * 50)
        print("'C'를 눌러 벽 캘리브레이션 시작")
        print("=" * 50)
        
        cv2.namedWindow('3D Ball Tracking', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # 프레임 획득
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 필터 적용
                depth_frame = self.spatial.process(depth_frame)
                depth_frame = self.temporal.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
                
                # numpy 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                current_time = time.time()
                
                # YOLO 검출
                results = self.model(color_image, stream=True, conf=0.5, verbose=False)
                
                detections = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            
                            # sports ball 클래스
                            if self.model.names[cls] in ['sports ball', 'ball', 'tennis ball']:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                
                                detections.append({
                                    'center': (cx, cy),
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(box.conf[0])
                                })
                
                # 각 검출된 공 추적
                for detection in detections:
                    self.track_ball(detection, depth_image, current_time)
                
                # 시각화
                vis_frame = self.visualize_tracking(color_image, depth_image)
                cv2.imshow('3D Ball Tracking', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.calibrate_wall()
                elif key == ord('r'):
                    self.balls.clear()
                    self.collisions.clear()
                    print("✨ 초기화 완료")
                elif key == ord('s'):
                    # 스크린샷 저장
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"tracking_{timestamp}.jpg", vis_frame)
                    print(f"📸 스크린샷 저장: tracking_{timestamp}.jpg")
                    
        except Exception as e:
            print(f"에러 발생: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("프로그램 종료")

if __name__ == "__main__":
    try:
        tracker = UltimateDepthBallTracker()
        tracker.run()
    except Exception as e:
        print(f"초기화 에러: {e}")
        import traceback
        traceback.print_exc()