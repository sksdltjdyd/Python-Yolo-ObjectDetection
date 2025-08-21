import cv2
import numpy as np
from state_manager import SetupStep

class Visualizer:
    """OpenCV 창 시각화를 담당합니다."""
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _simulate_ir_effect(self, image):
        """
        컬러 이미지를 흑백 적외선 사진 효과로 시뮬레이션합니다.
        """
        image_float = image.astype(np.float32)
        b, g, r = cv2.split(image_float)
        
        # 채널 믹싱
        ir_image = r * 0.6 + g * 0.4 - b * 0.1
        
        # 0-255 범위로 정규화
        ir_image = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
        ir_image = ir_image.astype(np.uint8)
        
        # 대비 강화
        ir_image_eq = cv2.equalizeHist(ir_image)
        
        # 부드러운 글로우 효과
        blurred = cv2.GaussianBlur(ir_image_eq, (15, 15), 0)
        final_ir_image = cv2.addWeighted(ir_image_eq, 1.5, blurred, -0.5, 0)
        
        # 3채널 흑백 이미지로 반환하여 다른 정보들과 함께 표시될 수 있도록 함
        return cv2.cvtColor(final_ir_image, cv2.COLOR_GRAY2BGR)

    def draw_setup_ui(self, image, setup_step, mask_points, background_captured, ir_mode=False):
        display = image.copy()

        # ✨ IR 모드가 활성화되었으면 필터 적용
        if ir_mode:
            display = self._simulate_ir_effect(display)

        # (기존 UI 그리기 로직)
        if setup_step == SetupStep.MASK_AREA:
            cv2.putText(display, "Step 1: Define Interaction Area", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if len(mask_points) >= 3:
                cv2.polylines(display, [np.array(mask_points, dtype=np.int32)], True, (0, 255, 0), 2)
            for p in mask_points:
                cv2.circle(display, p, 5, (0,255,0), -1)
        
        cv2.imshow(self.window_name, display)

    def draw_tracking_ui(self, image, tracked_balls, collision_points, ir_mode=False):
        display = image.copy()
        
        # ✨ IR 모드가 활성화되었으면 필터 적용
        if ir_mode:
            display = self._simulate_ir_effect(display)

        # (기존 UI 그리기 로직)
        for ball_id, ball_data in tracked_balls.items():
            x1, y1, x2, y2 = map(int, ball_data['bbox'])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            id_text = f"ID:{ball_id} S:{ball_data.get('speed', 0.0):.2f}m/s"
            cv2.putText(display, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for p_info in collision_points:
             cv2.circle(display, p_info['pixel_position'], 15, (0,0,255), -1)
             
        cv2.imshow(self.window_name, display)
        
    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self.window_name, callback)
    
    def destroy_windows(self):
        cv2.destroyAllWindows()