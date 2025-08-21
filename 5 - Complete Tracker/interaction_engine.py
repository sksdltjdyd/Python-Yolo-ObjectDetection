import pyrealsense2 as rs
import time
import numpy as np
import cv2
from collections import deque
from scipy.spatial.distance import cdist

class InteractionEngine:
    def __init__(self, camera_intrinsics):
        self.camera_intrinsics = camera_intrinsics
        self.tracked_balls = {}
        self.next_ball_id = 0
        self.max_disappeared = 20
        self.collision_cooldown = 0.5
        self.last_collision_times = {}

    def _create_kalman_filter(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        return kf

    def pixel_to_3d_point(self, x, y, depth):
        if self.camera_intrinsics is None or depth <= 0: return None
        point = rs.rs2_deproject_pixel_to_point(self.camera_intrinsics, [x, y], depth)
        return np.array(point)

    def _calculate_physics(self, ball_data):
        positions = ball_data['positions_3d']
        timestamps = ball_data['timestamps']
        
        if len(positions) < 2:
            ball_data['speed'] = 0.0
            ball_data['velocity_vector'] = np.array([0, 0, 0])
            return

        p1, p2 = positions[-2], positions[-1]
        dt = timestamps[-1] - timestamps[-2]

        if dt > 0:
            velocity_vector = (p2 - p1) / dt / 1000.0 # m/s
            speed = np.linalg.norm(velocity_vector)
            ball_data['speed'], ball_data['velocity_vector'] = speed, velocity_vector
        else:
            ball_data['speed'], ball_data['velocity_vector'] = 0.0, np.array([0, 0, 0])

    def update(self, detected_balls, depth_image):
        for ball_id in self.tracked_balls:
            predicted_state = self.tracked_balls[ball_id]['kf'].predict()
            self.tracked_balls[ball_id]['center'] = (int(predicted_state[0]), int(predicted_state[1]))

        if len(detected_balls) == 0:
            for ball_id in list(self.tracked_balls.keys()):
                self.tracked_balls[ball_id]['disappeared'] += 1
                if self.tracked_balls[ball_id]['disappeared'] > self.max_disappeared:
                    self._deregister_ball(ball_id)
            return

        detected_centroids = np.array([b['center'] for b in detected_balls])
        if len(self.tracked_balls) == 0:
            for ball in detected_balls: self._register_ball(ball, depth_image)
            return

        tracked_ids = list(self.tracked_balls.keys())
        tracked_centroids = np.array([b['center'] for b in self.tracked_balls.values()])
        D = cdist(tracked_centroids, detected_centroids)
        rows, cols = D.min(axis=1).argsort(), D.argmin(axis=1)[D.min(axis=1).argsort()]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            ball_id = tracked_ids[row]
            self.tracked_balls[ball_id]['disappeared'] = 0
            self._update_ball_data(ball_id, detected_balls[col], depth_image)
            used_rows.add(row); used_cols.add(col)

        for row in set(range(len(tracked_ids))) - used_rows:
            ball_id = tracked_ids[row]
            self.tracked_balls[ball_id]['disappeared'] += 1
            if self.tracked_balls[ball_id]['disappeared'] > self.max_disappeared:
                self._deregister_ball(ball_id)

        for col in set(range(len(detected_centroids))) - used_cols:
            self._register_ball(detected_balls[col], depth_image)
            
        for ball_id, ball_data in self.tracked_balls.items():
            if not ball_data['disappeared'] > 0:
                self._calculate_physics(ball_data)

    def _register_ball(self, ball_data, depth_image):
        ball_id = self.next_ball_id; self.next_ball_id += 1
        kf = self._create_kalman_filter()
        kf.statePost = np.array([ball_data['center'][0], ball_data['center'][1], 0, 0], np.float32)
        
        self.tracked_balls[ball_id] = {
            'id': ball_id, 'kf': kf, 'bbox': ball_data['bbox'], 'center': ball_data['center'],
            'positions_3d': deque(maxlen=30), 'timestamps': deque(maxlen=30), 'disappeared': 0,
            'speed': 0.0, 'velocity_vector': np.array([0,0,0]), 'was_outside': True
        }
        self._update_ball_data(ball_id, ball_data, depth_image)
        print(f"✅ Ball Registered: ID {ball_id}")

    def _deregister_ball(self, ball_id):
        del self.tracked_balls[ball_id]; print(f"❌ Ball Deregistered: ID {ball_id}")

    def _update_ball_data(self, ball_id, ball_data, depth_image):
        measurement = np.array(ball_data['center'], np.float32)
        corrected_state = self.tracked_balls[ball_id]['kf'].correct(measurement)
        smoothed_center = (int(corrected_state[0]), int(corrected_state[1]))
        self.tracked_balls[ball_id]['center'] = smoothed_center
        dx, dy = smoothed_center[0] - ball_data['center'][0], smoothed_center[1] - ball_data['center'][1]
        x1, y1, x2, y2 = ball_data['bbox']
        self.tracked_balls[ball_id]['bbox'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        cx, cy = smoothed_center
        if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
            depth = depth_image[cy, cx]
            point_3d = self.pixel_to_3d_point(cx, cy, depth)
            if point_3d is not None:
                self.tracked_balls[ball_id]['positions_3d'].append(point_3d)
                self.tracked_balls[ball_id]['timestamps'].append(time.time())

    def detect_collisions(self, wall_plane_equation, mask_points_2d):
        if wall_plane_equation is None or len(mask_points_2d) < 4: return []
        collided_balls, (normal_vector, point_on_plane) = [], wall_plane_equation
        mask_polygon, current_time = np.array(mask_points_2d, dtype=np.int32), time.time()

        for ball_id, ball_data in self.tracked_balls.items():
            if not ball_data['positions_3d']: continue
            last_pos_3d, last_pos_2d = ball_data['positions_3d'][-1], ball_data['center']
            distance_to_plane = np.dot(normal_vector, last_pos_3d - point_on_plane)
            is_outside_now = distance_to_plane > 50.0

            if (ball_data.get('was_outside', True) and not is_outside_now and
                cv2.pointPolygonTest(mask_polygon, last_pos_2d, False) >= 0 and
                current_time - self.last_collision_times.get(ball_id, 0) > self.collision_cooldown):
                collision_info = {
                    'ball_id': ball_id, 'position_3d': last_pos_3d, 'pixel_position': last_pos_2d,
                    'timestamp': current_time, 'speed': ball_data.get('speed', 0.0),
                    'velocity_vector': ball_data.get('velocity_vector', np.array([0,0,0]))
                }
                collided_balls.append(collision_info)
                self.last_collision_times[ball_id] = current_time
                print(f"💥 Collision! ID:{ball_id} Speed:{collision_info['speed']:.2f}m/s")
            
            ball_data['was_outside'] = is_outside_now
        return collided_balls