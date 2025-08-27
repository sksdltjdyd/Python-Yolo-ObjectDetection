import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import pickle
import time
from collections import deque
from pythonosc import udp_client

class ImprovedBallTracker:
    def __init__(self):
        # 기본 설정
        self.cam_width, self.cam_height = 640, 480
        self.confidence = 0.70
        self.yolo_path = "C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt"
        self.calibration_file = 'realsense_calibration_data.p'
        self.scale = 3
        
        # OSC 설정
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        
        # 추적 데이터
        self.tracking_history = deque(maxlen=10)
        self.collision_cooldown = 0
        self.last_collision_time = 0
        self.collision_threshold = 50  # mm
        
        # 캘리브레이션 로드
        self.load_calibration()
        
        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 60)
        
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # YOLO 모델
        self.model = YOLO(self.yolo_path)
        
        # 출력 이미지
        self.img_output = np.zeros((1080, 1920, 3), np.uint8)
        self.collision_points = []
        
    def load_calibration(self):
        try:
            with open(self.calibration_file, 'rb') as f:
                points = pickle.load(f)
            
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [self.cam_width, 0], 
                              [0, self.cam_height], [self.cam_width, self.cam_height]])
            self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.inverse_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            print("✅ Calibration loaded")
        except:
            print("❌ Calibration file not found")
            # 기본값 설정
            self.matrix = np.eye(3)
            self.inverse_matrix = np.eye(3)
    
    def detect_ball(self, img):
        results = self.model(img, stream=False, verbose=False, conf=self.confidence)
        balls = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    balls.append({'center': (cx, cy), 'conf': float(box.conf[0])})
        
        return img, balls
    
    def get_3d_position(self, x, y, depth_frame):
        """픽셀 좌표를 3D 위치로 변환 (언리얼 좌표계)"""
        depth = depth_frame.get_distance(int(x), int(y))
        if depth == 0:
            return None
        
        # RealSense 3D 좌표 (미터)
        point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)
        
        # 언리얼 좌표계로 변환 (센티미터)
        # RealSense: X(right), Y(down), Z(forward)
        # Unreal: X(forward), Y(right), Z(up)
        unreal_x = point_3d[2] * 100  # Z → X (forward)
        unreal_y = point_3d[0] * 100  # X → Y (right)
        unreal_z = -point_3d[1] * 100  # -Y → Z (up)
        
        return [unreal_x, unreal_y, unreal_z]
    
    def detect_collision(self, ball_pos, depth_frame):
        """개선된 충돌 감지"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_collision_time < 0.5:
            return False
        
        # 3D 위치 획득
        pos_3d = self.get_3d_position(ball_pos[0], ball_pos[1], depth_frame)
        if not pos_3d:
            return False
        
        # 추적 히스토리에 추가
        self.tracking_history.append({
            'pos_2d': ball_pos,
            'pos_3d': pos_3d,
            'time': current_time,
            'depth': pos_3d[0]  # 전방 거리
        })
        
        # 충돌 감지: 최소 5프레임 이상 추적했고, 깊이 변화가 급격히 멈춤
        if len(self.tracking_history) >= 5:
            recent = list(self.tracking_history)[-5:]
            depths = [h['depth'] for h in recent]
            
            # 속도 계산
            velocities = []
            for i in range(1, len(depths)):
                dt = recent[i]['time'] - recent[i-1]['time']
                if dt > 0:
                    v = (depths[i] - depths[i-1]) / dt
                    velocities.append(v)
            
            if velocities:
                # 평균 속도와 가속도 체크
                avg_velocity = np.mean(velocities[-3:]) if len(velocities) >= 3 else velocities[-1]
                
                # 벽에 접근 중이고 갑자기 멈춤
                if len(velocities) >= 2:
                    acceleration = velocities[-1] - velocities[-2]
                    
                    # 급격한 감속 = 충돌
                    if acceleration < -500 and abs(depths[-1] - depths[-2]) < self.collision_threshold:
                        self.last_collision_time = current_time
                        return True
        
        return False
    
    def send_osc_position(self, pos_3d, is_collision=False):
        """OSC 메시지 전송"""
        if not self.osc_client:
            return
        
        try:
            if is_collision:
                # 충돌 위치 전송
                self.osc_client.send_message("/ball/collision", [
                    float(pos_3d[0]),  # X
                    float(pos_3d[1]),  # Y
                    float(pos_3d[2]),  # Z
                    1.0  # 충돌 강도
                ])
                print(f"💥 Collision OSC: X={pos_3d[0]:.1f}, Y={pos_3d[1]:.1f}, Z={pos_3d[2]:.1f}")
            else:
                # 일반 위치 전송
                self.osc_client.send_message("/ball/position", [
                    float(pos_3d[0]),
                    float(pos_3d[1]),
                    float(pos_3d[2])
                ])
        except Exception as e:
            print(f"OSC error: {e}")
    
    def run(self):
        fps_timer = time.time()
        fps_count = 0
        fps = 0
        
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            img = np.asanyarray(color_frame.get_data())
            
            # Warp image
            img_warped = cv2.warpPerspective(img, self.matrix, (self.cam_width, self.cam_height))
            
            # Detect balls
            img_detected, balls = self.detect_ball(img_warped)
            
            if balls:
                # 가장 confidence 높은 공 선택
                best_ball = max(balls, key=lambda x: x['conf'])
                warped_center = best_ball['center']
                
                # 원본 좌표로 역변환
                warped_pt = np.array([[warped_center]], dtype=np.float32)
                original_pt = cv2.perspectiveTransform(warped_pt, self.inverse_matrix)
                original_x, original_y = original_pt[0][0]
                
                # 3D 위치 계산
                pos_3d = self.get_3d_position(original_x, original_y, depth_frame)
                
                if pos_3d:
                    # 위치 전송
                    self.send_osc_position(pos_3d, is_collision=False)
                    
                    # 충돌 체크
                    if self.detect_collision((original_x, original_y), depth_frame):
                        # 충돌 감지됨
                        self.send_osc_position(pos_3d, is_collision=True)
                        
                        # 시각화용 충돌 포인트 추가
                        collision_x = int(warped_center[0] * self.scale)
                        collision_y = int(warped_center[1] * self.scale)
                        self.collision_points.append((collision_x, collision_y))
            
            # 충돌 포인트 그리기
            for pt in self.collision_points:
                cv2.circle(self.img_output, pt, 15, (0, 0, 255), -1)
                cv2.circle(self.img_output, pt, 20, (0, 0, 200), 2)
            
            # FPS 계산
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            
            # 화면 표시
            cv2.putText(img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Original", img)
            cv2.imshow("Warped", img_detected)
            cv2.imshow("Collisions", self.img_output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.img_output.fill(0)
                self.collision_points.clear()
                self.tracking_history.clear()
                print("✨ Reset")
        
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ImprovedBallTracker()
    tracker.run()