import cv2
import numpy as np
from state_manager import SetupStep

class UIManager:
    """OpenCV 창, 트랙바, 시각화 등 UI 관련 로직을 담당"""
    def __init__(self, window_name, data_manager):
        self.window_name = window_name
        self.data_manager = data_manager
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        self.create_trackbars()

    def create_trackbars(self):
        cv2.resizeWindow('Controls', 400, 300)
        cv2.createTrackbar('Sensitivity', 'Controls', self.data_manager.depth_params['sensitivity'], 100,
                           lambda x: self.update_param('sensitivity', x))
        cv2.createTrackbar('Noise Reduction', 'Controls', self.data_manager.depth_params['noise_reduction'], 5,
                           lambda x: self.update_param('noise_reduction', x))
        # ... 다른 트랙바들 ...

    def update_param(self, param, value):
        self.data_manager.depth_params[param] = value

    def draw_setup_ui(self, image, setup_step, mask_points, background_captured):
        display = image.copy()
        if setup_step == SetupStep.MASK_AREA:
            # ... UI 그리기 로직 ...
            cv2.putText(display, "Step 1: MASK AREA", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if len(mask_points) >= 3:
                cv2.polylines(display, [np.array(mask_points, dtype=np.int32)], True, (0, 255, 0), 2)
            for p in mask_points:
                cv2.circle(display, p, 5, (0,255,0), -1)
        elif setup_step == SetupStep.BACKGROUND:
            cv2.putText(display, "Step 2: BACKGROUND", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if background_captured:
                cv2.putText(display, "Background Captured!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display)

    def draw_tracking_ui(self, image, tracked_balls): # ✨ 파라미터 변경
        display = image.copy()
        
        # ✨ 추적 중인 모든 공에 대해 ID와 BBox 그리기
        for ball_id, ball_data in tracked_balls.items():
            x1, y1, x2, y2 = ball_data['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 공 ID 표시
            id_text = f"ID: {ball_id}"
            cv2.putText(display, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ... (충돌 포인트 그리기 로직은 나중에 추가) ...

        cv2.imshow(self.window_name, display)
        
    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self.window_name, callback)