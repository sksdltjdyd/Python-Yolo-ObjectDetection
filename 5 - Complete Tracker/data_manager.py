import pickle
import os

class DataManager:
    """캘리브레이션 데이터, 설정 값 등을 관리하고 파일로 저장/로드"""
    def __init__(self, calibration_file='calibration_data.pkl'):
        self.CALIBRATION_FILE = calibration_file
        
        self.mask_points = []
        self.background_depth = None
        self.wall_distance = None
        self.depth_params = {
            "sensitivity": 25, 
            "noise_reduction": 3, 
            "min_depth_cm": 50, 
            "max_depth_cm": 300
        }
        
        # 추가된 변수: 3D 벽 경계면 정보
        self.wall_plane_points_3d = None # 4개의 코너 3D 좌표
        self.wall_plane_equation = None  # (normal_vector, point)

    def is_calibrated(self):
        # 이제 벽 경계면이 정의되었는지도 확인
        return self.background_depth is not None and len(self.mask_points) >= 4 and self.wall_plane_points_3d is not None

    def save_calibration(self):
        data = {
            'mask_points': self.mask_points,
            'background_depth': self.background_depth,
            'wall_distance': self.wall_distance,
            'depth_params': self.depth_params,
            # ✨ 저장할 데이터 추가
            'wall_plane_points_3d': self.wall_plane_points_3d,
            'wall_plane_equation': self.wall_plane_equation
        }
        with open(self.CALIBRATION_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Calibration saved to {self.CALIBRATION_FILE}")

    def load_calibration(self):
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                with open(self.CALIBRATION_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.mask_points = data.get('mask_points', [])
                self.background_depth = data.get('background_depth', None)
                self.wall_distance = data.get('wall_distance', None)
                self.depth_params = data.get('depth_params', self.depth_params)
                # ✨ 로드할 데이터 추가
                self.wall_plane_points_3d = data.get('wall_plane_points_3d', None)
                self.wall_plane_equation = data.get('wall_plane_equation', None)
                print(f"✅ Calibration loaded from {self.CALIBRATION_FILE}")
            except Exception as e:
                print(f"❌ Failed to load calibration: {e}")