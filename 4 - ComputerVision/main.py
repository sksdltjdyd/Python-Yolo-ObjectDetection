import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import pickle
import time
from collections import deque
from pythonosc import udp_client
import json

class UnrealFirstPersonTracker:
    def __init__(self):
        # === 기본 설정 ===
        self.cam_width, self.cam_height = 640, 480
        self.confidence = 0.70
        self.yolo_path = 'C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt'  # YOLO 모델 경로
        self.calibration_file = 'unreal_calibration.json'
        
        # === 언리얼 좌표계 설정 ===
        # 실제 카메라 설치 위치 (cm)
        self.camera_real_height = 170  # 눈높이
        self.camera_real_distance = 300  # 벽에서 거리
        
        # 언리얼 캐릭터 설정 (언리얼 단위 = cm)
        self.unreal_player_height = 170
        self.unreal_player_distance = 300
        
        # FOV 매칭
        self.camera_fov_h = 87  # RealSense D455 수평 FOV
        self.camera_fov_v = 58  # 수직 FOV
        
        # === OSC 설정 ===
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        print(f"OSC client ready: 127.0.0.1:8000")
        
        # === 트래킹 데이터 ===
        self.tracking_history = deque(maxlen=20)
        self.collision_history = []
        self.last_collision_time = 0
        self.collision_cooldown = 0.5
        
        # 캘리브레이션 데이터
        self.calibration_offset = np.array([0, 0, 0])
        self.calibration_scale = 1.0
        self.calibration_rotation = np.eye(3)
        
        # 캘리브레이션 포인트
        self.calibration_points = []
        self.warp_matrix = None
        self.inverse_warp_matrix = None
        
        # 로드 캘리브레이션
        self.load_calibration()
        
        # === RealSense 초기화 ===
        self.init_realsense()
        
        # === YOLO 모델 로드 ===
        self.init_yolo()
        
        # === 디스플레이 ===
        self.display_output = np.zeros((720, 1280, 3), np.uint8)
        self.is_calibrating = False
        self.calib_step = 0
        
    def init_realsense(self):
        """RealSense 카메라 초기화"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 60)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            # 카메라 내부 파라미터
            depth_profile = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            print("✅ RealSense initialized")
            
            # 초기 설정 언리얼로 전송
            self.send_camera_settings()
            
        except Exception as e:
            print(f"❌ RealSense init failed: {e}")
            exit(1)
    
    def init_yolo(self):
        """YOLO 모델 초기화"""
        try:
            self.model = YOLO(self.yolo_path)
            print(f"✅ YOLO model loaded: {self.yolo_path}")
        except:
            self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt')
            print("⚠️ Using default YOLOv8n model")
    
    def send_camera_settings(self):
        """언리얼에 카메라 설정 전송"""
        settings = {
            'fov_horizontal': self.camera_fov_h,
            'fov_vertical': self.camera_fov_v,
            'camera_height': self.unreal_player_height,
            'camera_distance': self.unreal_player_distance
        }
        
        # 카메라 설정 전송
        self.osc_client.send_message("/camera/settings", [
            float(settings['fov_horizontal']),
            float(settings['fov_vertical']),
            float(settings['camera_height']),
            float(settings['camera_distance'])
        ])
        
        print(f"📡 Sent camera settings to Unreal: FOV={self.camera_fov_h}°, Height={self.unreal_player_height}cm")
    
    def realsense_to_unreal_coords(self, rs_x, rs_y, rs_z):
        """
        RealSense 좌표를 언리얼 1인칭 좌표로 변환
        RealSense: X(right), Y(down), Z(forward) - 미터
        Unreal: X(forward), Y(right), Z(up) - 센티미터
        """
        # 기본 변환 (미터 -> 센티미터)
        point = np.array([rs_x * 100, rs_y * 100, rs_z * 100])
        
        # 캘리브레이션 적용
        point = point * self.calibration_scale
        point = np.dot(self.calibration_rotation, point)
        point = point + self.calibration_offset
        
        # 언리얼 좌표계로 매핑
        unreal_x = point[2]  # Z -> X (전방)
        unreal_y = point[0]  # X -> Y (우측)
        unreal_z = -point[1] + self.unreal_player_height  # -Y -> Z (상하, 플레이어 높이 기준)
        
        return [unreal_x, unreal_y, unreal_z]
    
    def pixel_to_3d_point(self, x, y, depth_frame):
        """픽셀 좌표를 3D 월드 좌표로 변환"""
        depth = depth_frame.get_distance(int(x), int(y))
        if depth == 0:
            return None
        
        # RealSense 3D 좌표 (미터)
        point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)
        
        # 언리얼 좌표로 변환
        unreal_coords = self.realsense_to_unreal_coords(point_3d[0], point_3d[1], point_3d[2])
        
        return unreal_coords
    
    def detect_ball(self, img):
        """YOLO로 공 검출"""
        results = self.model(img, stream=False, verbose=False, conf=self.confidence)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # 시각화
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Ball {float(box.conf[0]):.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'center': (cx, cy),
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf[0])
                    })
        
        return img, detections
    
    def detect_collision(self, current_pos_3d):
        """충돌 감지 로직"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_collision_time < self.collision_cooldown:
            return False
        
        # 트래킹 히스토리에 추가
        self.tracking_history.append({
            'position': current_pos_3d,
            'time': current_time
        })
        
        # 충돌 감지: 속도 급변
        if len(self.tracking_history) >= 5:
            recent = list(self.tracking_history)[-5:]
            
            # Z축(전방) 속도 계산
            velocities = []
            for i in range(1, len(recent)):
                dt = recent[i]['time'] - recent[i-1]['time']
                if dt > 0:
                    dz = recent[i]['position'][0] - recent[i-1]['position'][0]  # X축이 전방
                    v = dz / dt
                    velocities.append(v)
            
            if len(velocities) >= 2:
                # 가속도 계산
                acceleration = velocities[-1] - velocities[-2]
                
                # 급격한 감속 = 충돌
                if acceleration < -500:  # cm/s² 임계값
                    self.last_collision_time = current_time
                    return True
        
        return False
    
    def send_to_unreal(self, position, is_collision=False):
        """언리얼로 데이터 전송"""
        try:
            if is_collision:
                # 충돌 이벤트
                self.osc_client.send_message("/ball/collision", [
                    float(position[0]),  # X (forward)
                    float(position[1]),  # Y (right)
                    float(position[2]),  # Z (up)
                    time.time()         # timestamp
                ])
                print(f"💥 Collision sent: X={position[0]:.1f}, Y={position[1]:.1f}, Z={position[2]:.1f}")
            else:
                # 위치 업데이트
                self.osc_client.send_message("/ball/position", [
                    float(position[0]),
                    float(position[1]),
                    float(position[2])
                ])
        except Exception as e:
            print(f"OSC error: {e}")
    
    def calibrate_space(self, event, x, y, flags, param):
        """공간 캘리브레이션 마우스 콜백"""
        if event == cv2.EVENT_LBUTTONDOWN and self.is_calibrating:
            self.calibration_points.append([x, y])
            print(f"Calibration point {len(self.calibration_points)}: ({x}, {y})")
            
            if len(self.calibration_points) == 4:
                # Warp 매트릭스 계산
                pts1 = np.float32(self.calibration_points)
                pts2 = np.float32([[0, 0], [self.cam_width, 0], 
                                  [0, self.cam_height], [self.cam_width, self.cam_height]])
                
                self.warp_matrix = cv2.getPerspectiveTransform(pts1, pts2)
                self.inverse_warp_matrix = cv2.getPerspectiveTransform(pts2, pts1)
                
                self.save_calibration()
                self.is_calibrating = False
                self.calibration_points = []
                print("✅ Calibration complete!")
    
    def manual_offset_adjustment(self, key):
        """수동 오프셋 조정"""
        adjustment = 5  # cm
        
        if key == ord('i'):  # Forward
            self.calibration_offset[0] += adjustment
        elif key == ord('k'):  # Backward
            self.calibration_offset[0] -= adjustment
        elif key == ord('j'):  # Left
            self.calibration_offset[1] -= adjustment
        elif key == ord('l'):  # Right
            self.calibration_offset[1] += adjustment
        elif key == ord('u'):  # Up
            self.calibration_offset[2] += adjustment
        elif key == ord('o'):  # Down
            self.calibration_offset[2] -= adjustment
        
        print(f"Offset adjusted: {self.calibration_offset}")
        self.save_calibration()
    
    def save_calibration(self):
        """캘리브레이션 데이터 저장"""
        data = {
            'offset': self.calibration_offset.tolist(),
            'scale': self.calibration_scale,
            'rotation': self.calibration_rotation.tolist(),
            'warp_matrix': self.warp_matrix.tolist() if self.warp_matrix is not None else None,
            'inverse_warp_matrix': self.inverse_warp_matrix.tolist() if self.inverse_warp_matrix is not None else None
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """캘리브레이션 데이터 로드"""
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            self.calibration_offset = np.array(data['offset'])
            self.calibration_scale = data['scale']
            self.calibration_rotation = np.array(data['rotation'])
            
            if data['warp_matrix']:
                self.warp_matrix = np.array(data['warp_matrix'])
                self.inverse_warp_matrix = np.array(data['inverse_warp_matrix'])
            
            print(f"✅ Calibration loaded from {self.calibration_file}")
        except:
            print("⚠️ No calibration file found, using defaults")
    
    def run(self):
        """메인 루프"""
        cv2.namedWindow("RealSense Tracking")
        cv2.setMouseCallback("RealSense Tracking", self.calibrate_space)
        
        fps_timer = time.time()
        fps_count = 0
        fps = 0
        
        print("\n" + "="*50)
        print("CONTROLS:")
        print("C - Start calibration (click 4 corners)")
        print("I/K - Forward/Backward offset")
        print("J/L - Left/Right offset")
        print("U/O - Up/Down offset")
        print("R - Reset collision points")
        print("Q - Quit")
        print("="*50 + "\n")
        
        while True:
            # 프레임 획득
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 캘리브레이션 모드
            if self.is_calibrating:
                for i, pt in enumerate(self.calibration_points):
                    cv2.circle(color_image, tuple(pt), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, str(i+1), (pt[0]-10, pt[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(color_image, f"CALIBRATING: Click {4-len(self.calibration_points)} more corners",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            else:
                # Warp 적용 (캘리브레이션된 경우)
                if self.warp_matrix is not None:
                    warped_color = cv2.warpPerspective(color_image, self.warp_matrix, 
                                                       (self.cam_width, self.cam_height))
                else:
                    warped_color = color_image.copy()
                
                # 공 검출
                detected_img, balls = self.detect_ball(warped_color)
                
                if balls:
                    # 가장 신뢰도 높은 공 선택
                    best_ball = max(balls, key=lambda x: x['confidence'])
                    
                    # 원본 좌표로 역변환
                    if self.inverse_warp_matrix is not None:
                        warped_pt = np.array([[best_ball['center']]], dtype=np.float32)
                        original_pt = cv2.perspectiveTransform(warped_pt, self.inverse_warp_matrix)
                        cx, cy = original_pt[0][0]
                    else:
                        cx, cy = best_ball['center']
                    
                    # 3D 좌표 계산
                    pos_3d = self.pixel_to_3d_point(cx, cy, depth_frame)
                    
                    if pos_3d:
                        # 언리얼로 위치 전송
                        self.send_to_unreal(pos_3d, is_collision=False)
                        
                        # 충돌 체크
                        if self.detect_collision(pos_3d):
                            self.send_to_unreal(pos_3d, is_collision=True)
                            
                            # 충돌 시각화
                            collision_point = (int(cx), int(cy))
                            self.collision_history.append(collision_point)
                        
                        # 정보 표시
                        info_text = f"3D: X={pos_3d[0]:.0f} Y={pos_3d[1]:.0f} Z={pos_3d[2]:.0f}"
                        cv2.putText(color_image, info_text, (20, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Warped 뷰 표시
                if self.warp_matrix is not None:
                    cv2.imshow("Warped View", detected_img)
            
            # 충돌 포인트 표시
            for pt in self.collision_history[-10:]:  # 최근 10개
                cv2.circle(color_image, pt, 10, (0, 0, 255), -1)
            
            # FPS 계산
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            
            # 정보 표시
            cv2.putText(color_image, f"FPS: {fps}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(color_image, f"Offset: {self.calibration_offset}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 깊이 맵 컬러
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # 화면 표시
            combined = np.hstack([color_image, depth_colormap])
            cv2.imshow("RealSense Tracking", combined)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.is_calibrating = True
                self.calibration_points = []
                print("🎯 Calibration mode - click 4 corners")
            elif key == ord('r'):
                self.collision_history = []
                self.tracking_history.clear()
                print("✨ Reset")
            elif key in [ord('i'), ord('k'), ord('j'), ord('l'), ord('u'), ord('o')]:
                self.manual_offset_adjustment(key)
        
        # 종료
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = UnrealFirstPersonTracker()
    tracker.run()