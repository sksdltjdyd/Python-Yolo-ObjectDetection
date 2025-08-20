import numpy as np
import cv2
import time # time 임포트 추가

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
        
        # ✨ InteractionEngine 초기화 (카메라 파라미터 전달)
        self.engine = InteractionEngine(self.camera.depth_intrinsics)
        
        self.ui.set_mouse_callback(self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.state.app_mode == AppMode.SETUP:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.data.mask_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.data.mask_points:
                self.data.mask_points.pop()

    def run_setup_mode(self):
        # ... (기존과 동일)
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
                print("❌ Complete setup first!")
        elif key == ord('b'):
            # ... capture_background_action 로직 ...
            pass
        elif key == ord('s'):
            self.data.save_calibration()

    def run_tracking_mode(self):
        depth_frame, color_frame = self.camera.get_frames()
        if color_frame is None or depth_frame is None: return

        color_image = np.asanyarray(color_frame.get_data())
        
        # ✨ 필터 적용된 뎁스 이미지 가져오기
        filtered_depth_frame = self.camera.apply_filters(depth_frame, self.data.depth_params['noise_reduction'])
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        
        # 1. VisionProcessor로 공 탐지
        detected_balls = self.vision.detect_balls(color_image)
        
        # 2. InteractionEngine으로 추적 상태 업데이트
        self.engine.update(detected_balls, depth_image)
        
        # 3. UI에 추적 결과(tracked_balls)를 넘겨 시각화
        #    (ui_manager도 tracked_balls를 받도록 수정 필요)
        self.ui.draw_tracking_ui(color_image, self.engine.tracked_balls)

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