import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
from collections import deque
import pickle
import os
from enum import Enum
from pythonosc import udp_client

class AppMode(Enum):
    SETUP = "SETUP"
    RUNNING = "RUNNING"

class SetupStep(Enum):
    MASK_AREA = 1
    BACKGROUND = 2
    CALIBRATE = 3

class UltimateBallTracker:
    def __init__(self):
        # === 애플리케이션 전역 상태 ===
        self.app_mode = AppMode.SETUP
        self.setup_step = SetupStep.MASK_AREA
        
        # === 캘리브레이션 파일 ===
        self.CALIBRATION_FILE = 'calibration_data.pkl'
        
        # === 1단계: Mask Area 설정 ===
        self.mask_area_settings = {
            "flip": False,
            "mirror": False,
            "view_mode": 0
        }
        self.mask_points = []  # 초록색 점들
        self.mask_image = None
        
        # === 2단계: Background ===
        self.background_depth = None
        self.background_captured = False
        self.wall_distance = None  # 벽까지 거리
        
        # === 3단계: Calibrate 파라미터 ===
        self.depth_params = {
            "sensitivity": 25,
            "noise_reduction": 3,
            "delay": 0,
            "min_depth_cm": 50,
            "max_depth_cm": 300
        }
        self.view_mode = 0
        
        # === 공 추적 데이터 ===
        self.balls = {}
        self.next_ball_id = 0
        self.ball_trail = deque(maxlen=50)
        
        # === 충돌 데이터 ===
        self.collision_points = []
        self.last_collision_time = 0
        self.collision_cooldown = 0.5
        
        # === RealSense 초기화 ===
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            profile = self.pipeline.start(config)
            print("✅ Camera initialized successfully")
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("Please check USB connection")
            exit(1)
        
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # 정렬 객체 (선택적 사용)
        self.use_align = True
        try:
            self.align = rs.align(rs.stream.color)
        except:
            print("⚠️ Align not available, using raw frames")
            self.use_align = False
            self.align = None
        
        # 필터
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # YOLO 모델 로드
        try:
            if os.path.exists('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/ball.pt'):
                self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/ball.pt')
                print("✅ Custom ball model loaded")
            else:
                self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8l.pt')
                print("⚠️ Using default YOLO model")
        except:
            self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/yolov8l.pt')

        # OSC
        try:
            self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        except:
            self.osc_client = None
        
        # FPS 계산
        self.fps = 0
        self.frame_count = 0
        self.fps_timer = time.time()
        
        # 캘리브레이션 로드
        self.load_calibration()
        
        # 트랙바 생성
        self.create_trackbars()
    
    def get_aligned_frames(self):
        """안전한 프레임 획득 및 정렬"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            if not frames:
                return None, None
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # align 사용 가능하고 활성화된 경우
            if self.use_align and self.align:
                try:
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                except RuntimeError as e:
                    # align 실패 시 원본 사용
                    print(f"Align failed, using raw frames: {e}")
                    pass
            
            return depth_frame, color_frame
            
        except Exception as e:
            print(f"Frame acquisition error: {e}")
            return None, None
    
    def create_trackbars(self):
        """OpenCV 트랙바 생성"""
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 400, 300)
        
        cv2.createTrackbar('Sensitivity', 'Controls', 
                          self.depth_params['sensitivity'], 100, 
                          lambda x: self.update_param('sensitivity', x))
        cv2.createTrackbar('Noise Reduction', 'Controls', 
                          self.depth_params['noise_reduction'], 10, 
                          lambda x: self.update_param('noise_reduction', x))
        cv2.createTrackbar('Min Depth (cm)', 'Controls', 
                          self.depth_params['min_depth_cm'], 500, 
                          lambda x: self.update_param('min_depth_cm', x))
        cv2.createTrackbar('Max Depth (cm)', 'Controls', 
                          self.depth_params['max_depth_cm'], 500, 
                          lambda x: self.update_param('max_depth_cm', x))
    
    def update_param(self, param, value):
        """파라미터 업데이트"""
        self.depth_params[param] = value
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리 - 초록색 점 찍기"""
        if self.app_mode == AppMode.SETUP and self.setup_step == SetupStep.MASK_AREA:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mask_points.append((x, y))
                print(f"🟢 Point added: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.mask_points) > 0:
                    self.mask_points.pop()
                    print("❌ Last point removed")
    
    def pixel_to_3d_point(self, x, y, depth):
        """2D 픽셀을 실제 3D 좌표로 변환"""
        point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics, [x, y], depth
        )
        return np.array(point)
    
    def calculate_real_velocity(self, positions, timestamps):
        """실제 3D 공간에서의 속도 계산 (m/s)"""
        if len(positions) < 2:
            return 0, np.array([0, 0, 0])
        
        positions = np.array(list(positions))
        timestamps = np.array(list(timestamps))
        
        if len(positions) >= 3:
            p1 = np.mean(positions[-3:-1], axis=0)
            p2 = positions[-1]
            dt = timestamps[-1] - timestamps[-3]
        else:
            p1 = positions[-2]
            p2 = positions[-1]
            dt = timestamps[-1] - timestamps[-2]
        
        if dt > 0:
            velocity_vector = (p2 - p1) / dt / 1000  # mm/s → m/s
            speed = np.linalg.norm(velocity_vector)
            return speed, velocity_vector
        
        return 0, np.array([0, 0, 0])
    
    def detect_wall_collision(self, ball_data):
        """벽 충돌 감지 및 포인트 기록"""
        if not self.wall_distance or len(ball_data['positions']) < 2:
            return False, None
        
        current_pos = ball_data['positions'][-1]
        current_depth = current_pos[2]
        
        distance_to_wall = abs(current_depth - self.wall_distance)
        
        if distance_to_wall < 50:  # 50mm 이내
            current_time = time.time()
            
            if current_time - self.last_collision_time > self.collision_cooldown:
                self.last_collision_time = current_time
                
                collision_data = {
                    'position': current_pos,
                    'pixel_position': ball_data['pixel_positions'][-1],
                    'velocity': self.calculate_real_velocity(
                        ball_data['positions'],
                        ball_data['timestamps']
                    )[0],
                    'timestamp': current_time
                }
                
                self.collision_points.append(collision_data)
                return True, collision_data
        
        return False, None
    
    def create_mask(self, shape):
        """마스크 이미지 생성"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        
        if len(self.mask_points) >= 3:
            points = np.array(self.mask_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def apply_filters(self, depth_frame):
        """깊이 프레임 필터 적용"""
        if self.depth_params['noise_reduction'] > 0:
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        return depth_frame
    
    def capture_background(self):
        """배경 및 벽 거리 캡처 - 안전한 버전"""
        print("Capturing background and wall distance...")
        
        depth_frames = []
        successful_captures = 0
        
        # 30프레임 캡처 시도
        for attempt in range(60):  # 최대 60번 시도
            depth_frame, color_frame = self.get_aligned_frames()
            
            if depth_frame:
                try:
                    # 필터 적용
                    filtered_depth = self.apply_filters(depth_frame)
                    depth_image = np.asanyarray(filtered_depth.get_data())
                    
                    # 유효한 깊이 데이터 확인
                    if depth_image.size > 0 and np.any(depth_image > 0):
                        depth_frames.append(depth_image)
                        successful_captures += 1
                        
                        if successful_captures >= 30:
                            break
                            
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
        
        if len(depth_frames) == 0:
            print("❌ Failed to capture background!")
            return False
        
        print(f"✅ Captured {len(depth_frames)} frames")
        
        # 배경 깊이 (중앙값)
        self.background_depth = np.median(depth_frames, axis=0)
        
        # 벽 거리 계산
        if self.mask_image is not None and len(self.mask_points) >= 3:
            # 마스크 크기 확인 및 조정
            if self.mask_image.shape[:2] != self.background_depth.shape[:2]:
                self.mask_image = cv2.resize(
                    self.mask_image,
                    (self.background_depth.shape[1], self.background_depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            masked_depths = self.background_depth[self.mask_image > 0]
            if len(masked_depths) > 0:
                self.wall_distance = np.median(masked_depths[masked_depths > 0])
                print(f"✅ Wall distance: {self.wall_distance:.0f}mm")
        else:
            # 마스크가 없으면 중앙 영역 사용
            h, w = self.background_depth.shape
            center_region = self.background_depth[h//4:3*h//4, w//4:3*w//4]
            valid_depths = center_region[center_region > 0]
            if len(valid_depths) > 0:
                self.wall_distance = np.median(valid_depths)
                print(f"✅ Wall distance (center): {self.wall_distance:.0f}mm")
        
        self.background_captured = True
        print("✅ Background captured successfully!")
        return True
    
    def handle_mask_area_step(self, color_image, depth_image):
        """마스크 영역 설정 - 초록색 점으로"""
        display = color_image.copy()
        
        # 초록색 점들 그리기
        for i, point in enumerate(self.mask_points):
            # 초록색 원 (캘리브레이션 포인트)
            cv2.circle(display, point, 8, (0, 255, 0), -1)
            cv2.circle(display, point, 10, (0, 200, 0), 2)
            
            # 점 번호 표시
            cv2.putText(display, str(i+1), (point[0]-5, point[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 선으로 연결
            if i > 0:
                cv2.line(display, self.mask_points[i-1], point, (0, 255, 0), 2)
        
        # 폴리곤 완성
        if len(self.mask_points) >= 3:
            pts = np.array(self.mask_points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            
            # 반투명 영역 표시
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            
            # 마스크 생성
            self.mask_image = self.create_mask(color_image.shape)
        
        # 안내 텍스트
        cv2.putText(display, "Step 1: CALIBRATION AREA", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Click to add green points | Right-click to remove", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Points: {len(self.mask_points)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Setup', display)
    
    def run_tracking_mode(self):
        """트래킹 모드 - 공 추적 및 충돌 감지"""
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        
        print("🎮 TRACKING MODE")
        print("[Q] Quit | [E] Edit Setup | [R] Reset Collisions")
        
        while self.app_mode == AppMode.RUNNING:
            try:
                # FPS 계산
                self.frame_count += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_timer = time.time()
                
                # 프레임 획득
                depth_frame, color_frame = self.get_aligned_frames()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 필터 적용
                depth_frame = self.apply_filters(depth_frame)
                
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # YOLO 검출
                results = self.model(color_image, stream=True, conf=0.30, verbose=False)
                
                current_time = time.time()
                detected_balls = []
                
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            
                            # ball 클래스 확인
                            if self.model.names[cls] in ['ball', 'sports ball', 'tennis ball']:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                
                                # 깊이 획득
                                if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                                    depth = depth_image[cy, cx]
                                    
                                    if depth > 0:
                                        # 3D 위치 계산
                                        point_3d = self.pixel_to_3d_point(cx, cy, depth)
                                        
                                        detected_balls.append({
                                            'center': (cx, cy),
                                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                            'depth': depth,
                                            'point_3d': point_3d,
                                            'confidence': float(box.conf[0])
                                        })
                
                # 공 추적 업데이트
                for ball in detected_balls:
                    ball_id = 0  # 단일 공 추적
                    
                    if ball_id not in self.balls:
                        self.balls[ball_id] = {
                            'positions': deque(maxlen=30),
                            'pixel_positions': deque(maxlen=30),
                            'timestamps': deque(maxlen=30)
                        }
                    
                    self.balls[ball_id]['positions'].append(ball['point_3d'])
                    self.balls[ball_id]['pixel_positions'].append(ball['center'])
                    self.balls[ball_id]['timestamps'].append(current_time)
                    
                    # 충돌 감지
                    is_collision, collision_data = self.detect_wall_collision(self.balls[ball_id])
                    
                    if is_collision:
                        print(f"💥 Collision detected at {collision_data['position']}")
                
                # 시각화
                self.visualize_tracking(color_image, depth_image, detected_balls)
                
            except Exception as e:
                print(f"Tracking error: {e}")
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.app_mode = AppMode.SETUP
                cv2.destroyWindow('Tracking')
                print("↩ Returning to SETUP mode...")
            elif key == ord('r'):
                self.collision_points.clear()
                print("✨ Collisions reset")
    
    def visualize_tracking(self, color_image, depth_image, detected_balls):
        """트래킹 시각화 - 공과 충돌 포인트 표시"""
        display = color_image.copy()
        
        # FPS 표시
        cv2.putText(display, f"FPS: {self.fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 검출된 공 표시
        for ball in detected_balls:
            x1, y1, x2, y2 = ball['bbox']
            cx, cy = ball['center']
            
            # 바운딩 박스 (초록색)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 중심점
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
            
            # 깊이 및 속도 정보
            if 0 in self.balls and len(self.balls[0]['positions']) >= 2:
                speed, velocity = self.calculate_real_velocity(
                    self.balls[0]['positions'],
                    self.balls[0]['timestamps']
                )
                
                # 정보 표시
                info_text = f"D:{ball['depth']:.0f}mm S:{speed:.1f}m/s"
                cv2.putText(display, info_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # 속도 벡터
                if speed > 0.1:
                    end_x = int(cx + velocity[0] * 50)
                    end_y = int(cy + velocity[1] * 50)
                    cv2.arrowedLine(display, (cx, cy), (end_x, end_y),
                                   (255, 0, 0), 2, tipLength=0.3)
        
        # 공 궤적 그리기
        if 0 in self.balls and len(self.balls[0]['pixel_positions']) > 1:
            pts = np.array(list(self.balls[0]['pixel_positions']), dtype=np.int32)
            for i in range(1, len(pts)):
                thickness = int(i / len(pts) * 3) + 1
                cv2.line(display, tuple(pts[i-1]), tuple(pts[i]),
                        (0, 255, 255), thickness)
        
        # 충돌 포인트 표시
        for i, collision in enumerate(self.collision_points[-10:]):  # 최근 10개
            px, py = collision['pixel_position']
            
            # 충돌 지점 원
            cv2.circle(display, (px, py), 10, (0, 0, 255), -1)
            cv2.circle(display, (px, py), 15, (0, 0, 200), 2)
            
            # 충돌 번호
            cv2.putText(display, str(i+1), (px-5, py+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 캘리브레이션 영역 표시 (초록선)
        if len(self.mask_points) >= 3:
            pts = np.array(self.mask_points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            
            # 캘리브레이션 포인트들
            for point in self.mask_points:
                cv2.circle(display, point, 5, (0, 255, 0), -1)
        
        # 정보 패널
        info_y = 60
        if self.wall_distance:
            cv2.putText(display, f"Wall: {self.wall_distance:.0f}mm", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            info_y += 30
        
        cv2.putText(display, f"Collisions: {len(self.collision_points)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # 깊이 컬러맵
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # 결합 표시
        combined = np.hstack([display, depth_colormap])
        cv2.imshow('Tracking', combined)
    
    def save_calibration(self):
        """캘리브레이션 저장"""
        calibration_data = {
            'mask_points': self.mask_points,
            'background_depth': self.background_depth,
            'wall_distance': self.wall_distance,
            'depth_params': self.depth_params,
            'collision_points': self.collision_points
        }
        
        with open(self.CALIBRATION_FILE, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"✅ Calibration saved")
    
    def load_calibration(self):
        """캘리브레이션 로드"""
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                with open(self.CALIBRATION_FILE, 'rb') as f:
                    data = pickle.load(f)
                
                self.mask_points = data.get('mask_points', [])
                self.background_depth = data.get('background_depth', None)
                self.wall_distance = data.get('wall_distance', None)
                self.depth_params = data.get('depth_params', self.depth_params)
                self.collision_points = data.get('collision_points', [])
                
                if self.background_depth is not None:
                    self.background_captured = True
                
                print(f"✅ Calibration loaded")
                return True
            except Exception as e:
                print(f"❌ Load error: {e}")
        return False
    
    def run_setup_mode(self):
        """셋업 모드"""
        cv2.namedWindow('Setup', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Setup', self.mouse_callback)
        
        while self.app_mode == AppMode.SETUP:
            # 안전한 프레임 획득
            depth_frame, color_frame = self.get_aligned_frames()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_frame = self.apply_filters(depth_frame)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            if self.setup_step == SetupStep.MASK_AREA:
                self.handle_mask_area_step(color_image, depth_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                self.capture_background()
            elif key == ord('s'):
                self.save_calibration()
            elif key == ord('r'):
                if self.background_captured and len(self.mask_points) >= 3:
                    self.app_mode = AppMode.RUNNING
                    cv2.destroyWindow('Setup')
                    print("🎮 Starting RUNNING mode...")
                else:
                    if not self.background_captured:
                        print("❌ Capture background first (press 'B')")
                    if len(self.mask_points) < 3:
                        print("❌ Add at least 3 calibration points")
            elif key == ord('c'):
                self.mask_points.clear()
                print("🔄 Calibration points cleared")
    
    def run(self):
        """메인 실행"""
        print("=" * 50)
        print("Ultimate Ball Tracking System")
        print("=" * 50)
        print("Setup: Click green points for calibration area")
        print("[B] Capture background | [S] Save | [R] Run")
        print("[C] Clear points | [Q] Quit")
        print("=" * 50)
        
        try:
            while True:
                if self.app_mode == AppMode.SETUP:
                    self.run_setup_mode()
                elif self.app_mode == AppMode.RUNNING:
                    self.run_tracking_mode()
                else:
                    break
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Program terminated")

if __name__ == "__main__":
    tracker = UltimateBallTracker()
    tracker.run()