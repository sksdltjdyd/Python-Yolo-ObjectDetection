# calibration_manager.py
# 캘리브레이션 관리 파일

import json
import numpy as np
import cv2
import os

class CalibrationManager:
    def __init__(self, file_path='unreal_calibration.json', width=640, height=480):
        self.file_path = file_path
        self.width = width
        self.height = height

        # 2D 투시 변환 데이터
        self.points = []
        self.matrix = np.eye(3)
        self.inverse_matrix = np.eye(3)

        # 3D 좌표 보정 데이터
        self.offset = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        self.rotation = np.eye(3)
        
        self.load()

    def load(self):
        if not os.path.exists(self.file_path):
            print(f"Calibration file not found. Using default values.")
            return

        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            self.offset = np.array(data.get('offset', [0.0, 0.0, 0.0]))
            self.scale = data.get('scale', 1.0)
            self.rotation = np.array(data.get('rotation', np.eye(3).tolist()))
            
            if 'points' in data and len(data['points']) == 4:
                self.points = data['points']
                self.calculate_matrices()

            print(f"Calibration loaded from {self.file_path}")
        except Exception as e:
            print(f"Error loading calibration file: {e}")

    def save(self):
        data = {
            'offset': self.offset.tolist(),
            'scale': self.scale,
            'rotation': self.rotation.tolist(),
            'points': self.points
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Calibration saved to {self.file_path}")

    def add_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))
            if len(self.points) == 4:
                self.calculate_matrices()
                self.save()
                return True
        return False

    def reset_points(self):
        self.points = []
        print("Warp points reset.")

    def calculate_matrices(self):
        if len(self.points) == 4:
            pts1 = np.float32(self.points)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.inverse_matrix = cv2.getPerspectiveTransform(pts2, pts1)

    def warp(self, image):
        return cv2.warpPerspective(image, self.matrix, (self.width, self.height))

    def unwarp_point(self, point):
        pt_np = np.array([[point]], dtype=np.float32)
        return cv2.perspectiveTransform(pt_np, self.inverse_matrix)[0][0]

    def adjust_offset(self, axis, amount):
        # 언리얼 좌표계 기준 (X:앞뒤, Y:좌우, Z:상하)
        if axis == 'x': self.offset[2] += amount # RealSense Z축
        elif axis == 'y': self.offset[0] += amount # RealSense X축
        elif axis == 'z': self.offset[1] -= amount # RealSense -Y축
        print(f"Offset adjusted: {self.offset}")
        self.save()