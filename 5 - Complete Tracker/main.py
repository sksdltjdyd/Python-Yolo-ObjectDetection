import numpy as np
import cv2
import pyrealsense2 as rs
import tkinter as tk

from state_manager import StateManager, AppMode, SetupStep
from camera_manager import CameraManager
from vision_processor import VisionProcessor
from data_manager import DataManager
from visualizer import Visualizer # 이름 변경
from interaction_engine import InteractionEngine
from gui_manager import GUIManager # 신규 GUI 매니저 import

class Application:
    """모든 모듈을 총괄하고 메인 루프를 실행"""
    def __init__(self, root):
        self.root = root
        self.state = StateManager()
        self.camera = CameraManager()
        self.vision = VisionProcessor(model_path='yolov8n.pt')
        self.data = DataManager()
        self.visualizer = Visualizer('Tracker') # 이름 변경
        self.engine = InteractionEngine(self.camera.depth_intrinsics)

        # ✨ GUI에 전달할 콜백 함수들 정의
        app_callbacks = {
            'capture_background': self.capture_background_action,
            'update_param': self.update_depth_param,
            'start_running': self.start_running_mode,
            'stop_running': self.stop_running_mode,
            'reset_points': self.reset_mask_points
        }
        self.gui = GUIManager(self.root, app_callbacks)
        
        self.visualizer.set_mouse_callback(self.mouse_callback)
        self.collision_points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.state.app_mode == AppMode.SETUP and self.state.setup_step == SetupStep.MASK_AREA:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.data.mask_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.data.mask_points:
                self.data.mask_points.pop()

    def update_depth_param(self, param, value):
        self.data.depth_params[param] = int(value)
        # 카메라 필터에도 실시간 반영 (필요시)
        if param == 'noise_reduction' and value > 0:
            self.camera.spatial.set_option(rs.option.filter_magnitude, int(value))

    def reset_mask_points(self):
        self.data.mask_points.clear()
        print("Mask points reset.")

    def start_running_mode(self):
        if self.data.is_calibrated():
            self.state.app_mode = AppMode.RUNNING
            self.gui.show_running_mode()
            self.data.save_calibration() # 실행 시 자동 저장
        else:
            print("❌ Complete setup first! (Need at least 4 points and background capture)")
    
    def stop_running_mode(self):
        self.state.app_mode = AppMode.SETUP
        self.gui.show_setup_mode()

    def run_setup_mode(self):
        depth_frame, color_frame = self.camera.get_frames()
        if color_frame is None: return

        color_image = np.asanyarray(color_frame.get_data())
        self.visualizer.draw_setup_ui(color_image, self.state.setup_step, 
                                      self.data.mask_points, self.data.background_depth is not None)

    def capture_background_action(self):
        if len(self.data.mask_points) < 4:
            print("❌ Need at least 4 points to define the area before capturing background.")
            return
        
        print("Capturing background...")
        # 안정적인 배경 캡처를 위해 여러 프레임 평균
        stable_depth_frame = None
        for _ in range(10): # 10 프레임 시도
            d_frame, _ = self.camera.get_frames()
            if d_frame: stable_depth_frame = d_frame
        
        if stable_depth_frame:
            self.data.background_depth = np.asanyarray(stable_depth_frame.get_data())
            self.calculate_3d_plane()
            self.gui.update_status(True)
        else:
            print("❌ Failed to capture stable depth frame for background.")
            self.gui.update_status(False)

    def calculate_3d_plane(self):
        points_3d = []
        for p2d in self.data.mask_points[:4]:
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
            print("❌ Could not define 3D Wall Plane.")

    def run_tracking_mode(self):
        depth_frame, color_frame = self.camera.get_frames()
        if color_frame is None or depth_frame is None: return

        color_image = np.asanyarray(color_frame.get_data())
        filtered_depth_frame = self.camera.apply_filters(depth_frame, self.data.depth_params['noise_reduction'])
        if filtered_depth_frame is None: return
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        
        detected_balls = self.vision.detect_balls(color_image)
        self.engine.update(detected_balls, depth_image)
        
        new_collisions = self.engine.detect_collisions(
            self.data.wall_plane_equation, self.data.mask_points)
        self.collision_points.extend(new_collisions)

        self.visualizer.draw_tracking_ui(color_image, self.engine.tracked_balls, self.collision_points)

    def update(self):
        """메인 업데이트 루프. Tkinter와 함께 실행됩니다."""
        if self.state.app_mode == AppMode.SETUP:
            self.run_setup_mode()
        elif self.state.app_mode == AppMode.RUNNING:
            self.run_tracking_mode()
        
        # 15ms 마다 이 함수를 다시 호출
        self.root.after(15, self.update)

    def on_close(self):
        """창을 닫을 때 리소스를 정리"""
        print("Closing application...")
        self.camera.stop()
        self.visualizer.destroy_windows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close) # 닫기 버튼 콜백 설정
    app.update() # 메인 루프 시작
    root.mainloop()