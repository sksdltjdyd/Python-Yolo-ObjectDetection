import cv2
import numpy as np
from state_manager import SetupStep

class Visualizer:
    """OpenCV 창 시각화를 담당"""
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_setup_ui(self, image, setup_step, mask_points, background_captured):
        display = image.copy()
        if setup_step == SetupStep.MASK_AREA:
            cv2.putText(display, "Step 1: Define Interaction Area by clicking points", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if len(mask_points) >= 3:
                cv2.polylines(display, [np.array(mask_points, dtype=np.int32)], True, (0, 255, 0), 2)
            for p in mask_points:
                cv2.circle(display, p, 5, (0,255,0), -1)
        
        cv2.imshow(self.window_name, display)

    def draw_tracking_ui(self, image, tracked_balls, collision_points):
        display = image.copy()
        for ball_id, ball_data in tracked_balls.items():
            x1, y1, x2, y2 = map(int, ball_data['bbox'])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            id_text = f"ID: {ball_id}"
            cv2.putText(display, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for p_info in collision_points:
             cv2.circle(display, p_info['pixel_position'], 10, (0,0,255), -1)
             
        cv2.imshow(self.window_name, display)
        
    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self.window_name, callback)
    
    def destroy_windows(self):
        cv2.destroyAllWindows()