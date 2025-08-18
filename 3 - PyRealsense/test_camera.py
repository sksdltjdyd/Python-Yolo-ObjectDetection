import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense():
    # 파이프라인 생성
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 스트림 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 시작
    pipeline.start(config)
    print("✅ 카메라 연결 성공!")
    
    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 깊이 이미지 컬러맵
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 화면에 표시
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense Test', images)
            
            # Q 키로 종료
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("카메라 종료")

if __name__ == "__main__":
    test_realsense()