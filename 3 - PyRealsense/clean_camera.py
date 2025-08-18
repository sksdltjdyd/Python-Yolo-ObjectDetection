import pyrealsense2 as rs
import numpy as np
import cv2

def setup_camera_with_filters():
    # 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 스트림 활성화
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 파이프라인 시작
    profile = pipeline.start(config)
    
    # 센서 설정
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    
    # 프리셋 설정 (노이즈 감소)
    depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    
    # 필터 생성
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    
    print("✅ 필터 적용 카메라 시작!")
    
    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # 필터 적용
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            
            # numpy 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 깊이 컬러맵
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # 표시
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('Clean Camera', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_camera_with_filters()