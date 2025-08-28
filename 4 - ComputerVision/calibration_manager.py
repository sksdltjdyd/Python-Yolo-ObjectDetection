# calibration_manager.py
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
        self.warp_points = []
        self.warp_matrix = np.eye(3)
        self.inverse_warp_matrix = np.eye(3)

        # 3D 좌표 보정 데이터
        self.offset = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        self.rotation = np.eye(3)
        
        self.load()

    def load(self):
        if not os.path.exists(self.file_path):
            print(f"⚠️ Calibration file not found. Using default values.")
            return

        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            self.offset = np.array(data.get('offset', [0.0, 0.0, 0.0]))
            self.scale = data.get('scale', 1.0)
            self.rotation = np.array(data.get('rotation', np.eye(3).tolist()))
            
            if 'warp_points' in data and len(data['warp_points']) == 4:
                self.warp_points = data['warp_points']
                self.calculate_matrices()

            print(f"✅ Calibration loaded from {self.file_path}")
        except Exception as e:
            print(f"❌ Error loading calibration file: {e}")

    def save(self):
        data = {
            'offset': self.offset.tolist(),
            'scale': self.scale,
            'rotation': self.rotation.tolist(),
            'warp_points': self.warp_points
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Calibration saved to {self.file_path}")

    def add_warp_point(self, x, y):
        if len(self.warp_points) < 4:
            self.warp_points.append((x, y))
            if len(self.warp_points) == 4:
                self.calculate_matrices()
                self.save()
                return True
        return False

    def reset_warp_points(self):
        self.warp_points = []
        print("🔄 Warp points reset.")

    def calculate_matrices(self):
        if len(self.warp_points) == 4:
            pts1 = np.float32(self.warp_points)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            self.warp_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.inverse_warp_matrix = cv2.getPerspectiveTransform(pts2, pts1)

    def warp(self, image):
        if self.warp_matrix is None: return image
        return cv2.warpPerspective(image, self.warp_matrix, (self.width, self.height))

    def unwarp_point(self, point):
        if self.inverse_warp_matrix is None: return point
        pt_np = np.array([[point]], dtype=np.float32)
        return cv2.perspectiveTransform(pt_np, self.inverse_warp_matrix)[0][0]

    def adjust_offset(self, axis, amount):
        self.offset[axis] += amount
        print(f"Offset adjusted: {self.offset}")
        self.save()