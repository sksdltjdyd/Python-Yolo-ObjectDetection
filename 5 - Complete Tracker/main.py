import numpy as np
import cv2
import pyrealsense2 as rs # rs 임포트 추가

# 분리된 모듈들을 import합니다.
from state_manager import StateManager, AppMode, SetupStep
from camera_manager import CameraManager
from vision_processor import VisionProcessor
from data_manager import DataManager
from ui_manager import UIManager
from interaction_engine import InteractionEngine

class Application:
    """모든 모듈을 총괄하고 메인 루프를 실행"""
    def __init__(self):
        self.state = StateManager()
        self.camera = CameraManager()
        self.vision = VisionProcessor(model_path='yolov8n.pt')
        self.data = DataManager()
        self.ui = UIManager('Ultimate Tracker', self.data)
        self.engine = InteractionEngine(self.camera.depth_intrinsics)
        
        self.ui.set_mouse_callback(self.mouse_callback)
        self.collision_points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.state.app_mode == AppMode.SETUP and self.state.setup_step == SetupStep.MASK_AREA:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.data.mask_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.data.mask_points:
                self.data.mask_points.pop()

    def run_setup_mode(self):
        depth_frame, color_frame = self.camera.get_frames()
        if color_frame is None: return

        color_image = np.asanyarray(color_frame.get_data())
        self.ui.draw_setup_ui(color_image, self.state.setup_step, 
                               self.data.mask_points, self.data.background_depth is not None)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): self.state.app_mode = None
        elif key == ord('r'):
            if self.data.is_calibrated():
                self.state.app_mode = AppMode.RUNNING
            else:
                print("❌ Complete setup first! (Need 4 points and background capture)")
        elif key == ord('b'):
            self.capture_background_action() # Action 호출
        elif key == ord('s'):
            self.data.save_calibration()
    
    def capture_background_action(self):
        print("Capturing background...")
        # ... 배경 캡처 로직 ...
        # 3D 경계면 계산 로직 추가
        if len(self.data.mask_points) >= 4:
            points_3d = []
            # 배경 캡처 시 사용했던 뎁스 프레임이 필요하므로, 이 함수 내에서 프레임 획득
            # 간단하게 하기 위해, 현재 data에 저장된 background_depth를 사용
            if self.data.background_depth is None:
                print("Capturing stable background for plane definition...")
                # 안정적인 배경 프레임 캡처 (실제 구현에서는 더 많은 프레임 평균 필요)
                _, d_frame = self.camera.get_frames()
                if d_frame: self.data.background_depth = np.asanyarray(d_frame.get_data())

            if self.data.background_depth is not None:
                for p2d in self.data.mask_points[:4]: # 4개의 점만 사용
                    y, x = p2d[1], p2d[0]
                    if 0 <= y < self.data.background_depth.shape[0] and 0 <= x < self.data.background_depth.shape[1]:
                        depth = self.data.background_depth[y, x]
                        if depth > 0:
                            point_3d = rs.rs2_deproject_pixel_to_point(self.camera.depth_intrinsics, [x, y], depth)
                            points_3d.append(point_3d)

                if len(points_3d) == 4:
                    self.data.wall_plane_points_3d = np.array(points_3d, dtype=np.float32)
                    v1 = self.data.wall_plane_points_3d[1] - self.data.wall_plane_points_3d[0]
                    v2 = self.data.wall_plane_points_3d[3] - self.data.wall_plane_points_3d[0]
                    normal = np.cross(v1, v2)
                    self.data.wall_plane_equation = (normal / np.linalg.norm(normal), self.data.wall_plane_points_3d[0])
                    print("✅ 3D Wall Plane defined.")
                else:
                    print("❌ Could not define 3D Wall Plane. Check depth at corner points.")
        else:
            print("❌ Need at least 4 points to define a plane.")

    def run_tracking_mode(self):
        depth_frame, color_frame = self.camera.get_frames()
        if color_frame is None or depth_frame is None: return

        color_image = np.asanyarray(color_frame.get_data())
        filtered_depth_frame = self.camera.apply_filters(depth_frame, self.data.depth_params['noise_reduction'])
        if filtered_depth_frame is None: return
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        
        detected_balls = self.vision.detect_balls(color_image)
        self.engine.update(detected_balls, depth_image)
        
        # 새로운 충돌 감지 로직 호출
        new_collisions = self.engine.detect_collisions(
            self.data.wall_plane_equation,
            self.data.mask_points
        )
        self.collision_points.extend(new_collisions)

        self.ui.draw_tracking_ui(color_image, self.engine.tracked_balls, self.collision_points)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): self.state.app_mode = None
        elif key == ord('e'): self.state.app_mode = AppMode.SETUP

    def run(self):
        while self.state.app_mode is not None:
            if self.state.app_mode == AppMode.SETUP:
                self.run_setup_mode()
            elif self.state.app_mode == AppMode.RUNNING:
                self.run_tracking_mode()
        
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    app = Application()
    app.run()