import pyrealsense2 as rs
import numpy as np

class CameraManager:
    """RealSense 카메라 연결, 프레임 획득, 필터링을 담당"""
    def __init__(self, width=640, height=480):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)
            print("✅ Camera initialized successfully")
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            exit(1)
            
        # 필터
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.decimation = rs.decimation_filter()

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
            # 이 에러는 흔하므로 간단히 건너뜁니다.
            return None, None

    def apply_filters(self, depth_frame, noise_reduction_level):
        if noise_reduction_level > 0:
            # 값이 0일 경우 필터 비활성화를 위해 1로 설정 (필터는 1-5 범위만 허용)
            safe_value = max(1, noise_reduction_level)
            self.spatial.set_option(rs.option.filter_magnitude, safe_value)

            if noise_reduction_level > 3: # 높은 레벨에서만 해상도 감소 필터 사용
                 depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        return depth_frame

    def stop(self):
        self.pipeline.stop()