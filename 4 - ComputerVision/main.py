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
        self.camera = CameraManager()
        self.calibration = CalibrationManager(file_path='unreal_calibration.json')
        self.vision = VisionProcessor(model_path="C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Yolo-Weights/Ball.pt", confidence=0.7)
        self.tracker = BallTracker(self.camera.depth_intrinsics, player_height_cm=170)
        self.osc = OSCManager()

        self.is_calibrating = len(self.calibration.warp_points) < 4
        self.collision_points_display = []

    def mouse_callback(self, event, x, y, flags, param):
        if self.is_calibrating and event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration.add_warp_point(x, y):
                self.is_calibrating = False
                print("✅ Warp calibration complete! Tracking will now start.")

    def handle_key_input(self, key):
        if key == ord('q'): return False # 'q'를 누르면 False를 반환하여 루프 종료
        
        # 2D 투시 보정 리셋
        if key == ord('c'):
            self.is_calibrating = True
            self.calibration.reset_warp_points()
        
        # 충돌 기록 리셋
        elif key == ord('r'):
            self.collision_points_display.clear()
            print("✨ Collision points reset.")
        
        # 3D 오프셋 조정
        elif key in [ord('i'), ord('k')]: # Forward/Backward
            self.calibration.adjust_offset(2, 5 if key == ord('i') else -5)
        elif key in [ord('j'), ord('l')]: # Left/Right
            self.calibration.adjust_offset(0, -5 if key == ord('j') else 5)
        elif key in [ord('u'), ord('o')]: # Up/Down
            self.calibration.adjust_offset(1, -5 if key == ord('u') else 5)
            
        return True

    def run(self):
        cv2.namedWindow("Tracker")
        cv2.setMouseCallback("Tracker", self.mouse_callback)
        print("\n--- CONTROLS ---\nC: Reset Warp Calibration | R: Reset Collisions\nI/K/J/L/U/O: Adjust 3D Offset | Q: Quit\n------------------")

        while True:
            depth_frame, color_frame = self.camera.get_frames()
            if color_frame is None or depth_frame is None: continue

            color_image = np.asanyarray(color_frame.get_data())
            
            if self.is_calibrating:
                cv2.putText(color_image, f"Click {4 - len(self.calibration.warp_points)} more points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                for pt in self.calibration.warp_points:
                    cv2.circle(color_image, pt, 5, (0, 255, 0), -1)
            else:
                warped_image = self.calibration.warp(color_image)
                warped_image_detected, detected_balls = self.vision.detect_ball(warped_image)
                
                current_ball, collision_event = self.tracker.update(detected_balls, depth_frame, self.calibration)

                if current_ball and current_ball.get('3d'):
                    self.osc.send("/ball/position", current_ball['3d'])

                if collision_event and collision_event.get('3d'):
                    self.osc.send("/ball/collision", collision_event['3d'])
                    self.osc.send(f"/ball/0/destroy", []) # ✨ 공 사라짐(소멸) 신호 복원

                    # 시각화용 충돌 포인트 추가
                    warped_pos = collision_event['2d_warped']
                    scaled_x = int(warped_pos[0] * (1920 / self.camera.width))
                    scaled_y = int(warped_pos[1] * (1080 / self.camera.height))
                    self.collision_points_display.append((scaled_x, scaled_y))

                cv2.imshow("Warped View", warped_image_detected)

            # 충돌 벽 시각화
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