# tracker.py
import time
import pyrealsense2 as rs
import numpy as np

class BallTracker:
    def __init__(self, intrinsics, player_height_cm=170):
        self.intrinsics = intrinsics
        self.player_height = player_height_cm
        self.last_known_pos = None
        self.last_collision_time = 0
        self.collision_cooldown = 0.5
        self.tracking_history = []

    def _realsense_to_unreal(self, point_rs, calibration):
        # RealSense (미터) -> 기본 좌표계 (센티미터)
        point_cm = np.array([point_rs[0] * 100, point_rs[1] * 100, point_rs[2] * 100])

        # 저장된 3D 캘리브레이션 값 적용
        point_cm = point_cm * calibration.scale
        point_cm = np.dot(calibration.rotation, point_cm)
        point_cm = point_cm + calibration.offset
        
        # 언리얼 좌표계 축 매핑 및 플레이어 높이 보정
        unreal_x = point_cm[2]  # Z -> X (forward)
        unreal_y = point_cm[0]  # X -> Y (right)
        unreal_z = -point_cm[1] + self.player_height  # -Y -> Z (up)
        return [unreal_x, unreal_y, unreal_z]

    def get_3d_position(self, x, y, depth_frame, calibration):
        depth = depth_frame.get_distance(int(x), int(y))
        if depth == 0: return None
        
        point_rs = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return self._realsense_to_unreal(point_rs, calibration)

    def update(self, detected_balls, depth_frame_unwarped, calib_manager):
        current_time = time.time()
        ball_detected = bool(detected_balls)
        ball_was_present = self.last_known_pos is not None

        collision_event = None
        current_ball_data = None

        if ball_detected:
            best_ball = max(detected_balls, key=lambda x: x['confidence'])
            warped_center = best_ball['center']
            unwarped_center = calib_manager.unwarp_point(warped_center)
            
            pos_3d = self.get_3d_position(unwarped_center[0], unwarped_center[1], depth_frame_unwarped, calib_manager)
            
            current_ball_data = {'2d_unwarped': unwarped_center, '2d_warped': warped_center, '3d': pos_3d, 'time': current_time}
            self.last_known_pos = current_ball_data

        if not ball_detected and ball_was_present:
            if current_time - self.last_collision_time > self.collision_cooldown:
                collision_event = self.last_known_pos
                print(f"💥 Collision Detected (Ball Disappeared) at 3D pos: {collision_event.get('3d')}")
                self.last_collision_time = current_time
            self.last_known_pos = None

        return current_ball_data, collision_event