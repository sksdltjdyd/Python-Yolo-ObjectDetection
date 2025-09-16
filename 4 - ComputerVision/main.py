# main.py
import cv2
import numpy as np
import time

from camera_manager import CameraManager
from vision_processor import VisionProcessor
from calibration_manager import CalibrationManager
from osc_manager import OSCManager
from tracker import BallTracker

class App:
    def __init__(self):
        # 5. 트래킹 감도 조절 (confidence 값 낮춤)
        self.vision = VisionProcessor(
            model_path="C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt",
            confidence=0.4 
        )
        self.camera = CameraManager()
        self.calibration = CalibrationManager(file_path='unreal_calibration.json')
        self.tracker = BallTracker(self.camera.depth_intrinsics, player_height_cm=170)
        self.osc = OSCManager()

        self.is_calibrating = len(self.calibration.points) < 4
        self.collision_points_display = []
        
        self.fps = 0
        self.fps_timer = time.time()
        self.fps_frame_count = 0

    def mouse_callback(self, event, x, y, flags, param):
        if self.is_calibrating and event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration.add_point(x, y):
                self.is_calibrating = False
                print("✅ Calibration complete! Tracking will now start.")

    def handle_key_input(self, key):
        if key == ord('q'): return False
        
        if key == ord('c'):
            self.is_calibrating = True
            self.calibration.reset_points()
        
        elif key == ord('r'):
            self.collision_points_display.clear()
            print("✨ Collision points reset.")
        
        # 2. 3D 오프셋 조정 기능 복원
        elif key in [ord('i'), ord('k')]: # Forward/Backward (Unreal X)
            self.calibration.adjust_offset('x', 5 if key == ord('i') else -5)
        elif key in [ord('j'), ord('l')]: # Left/Right (Unreal Y)
            self.calibration.adjust_offset('y', -5 if key == ord('j') else 5)
        elif key in [ord('u'), ord('o')]: # Up/Down (Unreal Z)
            self.calibration.adjust_offset('z', 5 if key == ord('u') else -5)
            
        return True

    def run(self):
        cv2.namedWindow("Tracker")
        cv2.setMouseCallback("Tracker", self.mouse_callback)
        print("\n--- CONTROLS ---\nC: Reset Warp Calib | R: Reset Collisions\nI/K/J/L/U/O: Adjust 3D Offset | Q: Quit\n------------------")

        while True:
            depth_frame, color_frame = self.camera.get_frames()
            if color_frame is None or depth_frame is None: continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # FPS 계산
            self.fps_frame_count += 1
            if time.time() - self.fps_timer > 1.0:
                self.fps = self.fps_frame_count
                self.fps_frame_count = 0
                self.fps_timer = time.time()
            
            # 3. 화면 정보 표시 기능 복원
            cv2.putText(color_image, f"FPS: {int(self.fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            offset = self.calibration.offset
            offset_text = f"Offset(RS): X({offset[0]:.0f}) Y({offset[1]:.0f}) Z({offset[2]:.0f})"
            cv2.putText(color_image, offset_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if self.is_calibrating:
                cv2.putText(color_image, f"Click {4 - len(self.calibration.points)} more points", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                for pt in self.calibration.points:
                    cv2.circle(color_image, pt, 5, (0, 255, 0), -1)
            else:
                warped_image = self.calibration.warp(color_image)
                warped_image_detected, detected_balls = self.vision.detect_ball(warped_image)
                
                current_ball, collision_event = self.tracker.update(detected_balls, depth_frame, self.calibration)

                if current_ball and current_ball.get('3d'):
                    self.osc.send("/ball/position", current_ball['3d'])
                    # 4. 공 좌표 표시 기능 복원
                    pos_text = f"Ball Pos(UE): X({current_ball['3d'][0]:.0f}) Y({current_ball['3d'][1]:.0f}) Z({current_ball['3d'][2]:.0f})"
                    cv2.putText(warped_image_detected, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


                if collision_event and collision_event.get('3d'):
                    self.osc.send("/ball/collision", collision_event['3d'])
                    # 2. 공 사라짐 감지 기능 복원
                    self.osc.send(f"/ball/0/destroy", [])
                    
                    warped_pos = collision_event['2d_warped']
                    scaled_x = int(warped_pos[0] * (1920 / self.camera.width))
                    scaled_y = int(warped_pos[1] * (1080 / self.camera.height))
                    self.collision_points_display.append((scaled_x, scaled_y))

                cv2.imshow("Warped View", warped_image_detected)

            output_display = np.zeros((1080, 1920, 3), dtype=np.uint8)
            for pt in self.collision_points_display:
                cv2.circle(output_display, pt, 15, (0, 0, 255), -1)
            cv2.imshow("Collision Wall", output_display)

            cv2.imshow("Tracker", color_image)
            
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key_input(key): break

        self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    app.run()