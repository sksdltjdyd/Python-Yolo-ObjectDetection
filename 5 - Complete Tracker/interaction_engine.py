import pyrealsense2 as rs
import time
import numpy as np
import cv2 # ✨ cv2 import 추가
from collections import deque
from scipy.spatial.distance import cdist

class InteractionEngine:
    """객체 추적(칼만 필터 포함), 3D 변환, 충돌 감지 등 모든 인터랙션 로직을 담당"""
    def __init__(self, camera_intrinsics):
        self.camera_intrinsics = camera_intrinsics
        self.tracked_balls = {}
        self.next_ball_id = 0
        self.max_disappeared = 20

    def _create_kalman_filter(self):
        """새로운 공을 위한 칼만 필터를 생성하고 초기화"""
        # 상태 변수: [x, y, vx, vy] (위치 x, y, 속도 vx, vy)
        # 측정 변수: [x, y] (탐지된 위치)
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        return kf

    def pixel_to_3d_point(self, x, y, depth):
        if self.camera_intrinsics is None or depth <= 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(self.camera_intrinsics, [x, y], depth)
        return np.array(point)

    def update(self, detected_balls, depth_image):
        """매 프레임 호출되어 공 추적 상태를 업데이트합니다."""
        
        # 1. 기존에 추적하던 모든 공의 다음 위치를 '예측'
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
            for ball in detected_balls:
                self._register_ball(ball, depth_image)
            return

        tracked_ids = list(self.tracked_balls.keys())
        # 예측된 위치를 기반으로 거리 계산
        tracked_centroids = np.array([b['center'] for b in self.tracked_balls.values()])

        D = cdist(tracked_centroids, detected_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            
            ball_id = tracked_ids[row]
            self.tracked_balls[ball_id]['disappeared'] = 0
            # ✨ 실제 측정값으로 칼만 필터를 '보정'하고 최종 위치 업데이트
            self._update_ball_data(ball_id, detected_balls[col], depth_image)
            
            used_rows.add(row)
            used_cols.add(col)

        # ... (매칭되지 않은 공 처리 로직은 이전과 동일) ...
        unused_rows = set(range(len(tracked_ids))) - used_rows
        for row in unused_rows:
            ball_id = tracked_ids[row]
            self.tracked_balls[ball_id]['disappeared'] += 1
            if self.tracked_balls[ball_id]['disappeared'] > self.max_disappeared:
                self._deregister_ball(ball_id)

        unused_cols = set(range(len(detected_centroids))) - used_cols
        for col in unused_cols:
            self._register_ball(detected_balls[col], depth_image)

    def _register_ball(self, ball_data, depth_image):
        """새로운 공을 등록하고, 공을 위한 칼만 필터를 생성합니다."""
        ball_id = self.next_ball_id
        self.next_ball_id += 1
        
        kf = self._create_kalman_filter()
        # 칼만 필터의 초기 상태를 처음 감지된 위치로 설정
        kf.statePost = np.array([ball_data['center'][0], ball_data['center'][1], 0, 0], np.float32)
        
        self.tracked_balls[ball_id] = {
            'id': ball_id,
            'kf': kf, # 칼만 필터 인스턴스 저장
            'bbox': ball_data['bbox'],
            'center': ball_data['center'],
            'positions_3d': deque(maxlen=30),
            'timestamps': deque(maxlen=30),
            'disappeared': 0
        }
        self._update_ball_data(ball_id, ball_data, depth_image, is_new=True)
        print(f"✅ Ball Registered: ID {ball_id}")

    def _deregister_ball(self, ball_id):
        del self.tracked_balls[ball_id]
        print(f"❌ Ball Deregistered: ID {ball_id}")

    def _update_ball_data(self, ball_id, ball_data, depth_image, is_new=False):
        """공의 데이터를 업데이트하고, 칼만 필터를 보정합니다."""
        # 칼만 필터 보정
        measurement = np.array([ball_data['center'][0], ball_data['center'][1]], np.float32)
        corrected_state = self.tracked_balls[ball_id]['kf'].correct(measurement)
        
        # 보정된 위치를 최종 위치로 사용
        smoothed_center = (int(corrected_state[0]), int(corrected_state[1]))
        
        self.tracked_balls[ball_id]['center'] = smoothed_center
        
        # BBox 위치도 보정된 중심점에 맞춰 이동
        cx_raw, cy_raw = ball_data['center']
        cx_smooth, cy_smooth = smoothed_center
        dx, dy = cx_smooth - cx_raw, cy_smooth - cy_raw
        x1, y1, x2, y2 = ball_data['bbox']
        self.tracked_balls[ball_id]['bbox'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        # 3D 위치 업데이트
        cx, cy = smoothed_center
        if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
            depth = depth_image[cy, cx]
            point_3d = self.pixel_to_3d_point(cx, cy, depth)
            if point_3d is not None:
                self.tracked_balls[ball_id]['positions_3d'].append(point_3d)
                self.tracked_balls[ball_id]['timestamps'].append(time.time())