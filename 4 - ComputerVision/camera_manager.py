# camera_manager.py
# RealSense 카메라의 초기화, 프레임 획득, 정렬 등 하드웨어와 관련된 모든 것을 담당

import pyrealsense2 as rs

class CameraManager:
    def __init__(self, width=640, height=480, fps=60):
        # 추가된 부분: width와 height를 클래스 변수로 저장
        self.width = width
        self.height = height
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        # 저장된 클래스 변수를 사용하도록 수정
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            print("RealSense camera initialized.")
        except Exception as e:
            print(f"RealSense init failed: {e}")
            exit(1)

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None
            return depth_frame, color_frame
        except RuntimeError:
            return None, None

    def stop(self):
        self.pipeline.stop()