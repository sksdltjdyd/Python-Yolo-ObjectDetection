import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

class AdvancedDepthBallTracker:
    """
    RealSense 뎁스 카메라와 YOLO를 사용하여 3D 공간의 공을 추적하고,
    사용자가 지정한 벽 영역과의 충돌을 감지하는 클래스.
    """
    def __init__(self):
        # RealSense 파이프라인 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 스트리밍 시작 및 카메라 내부 파라미터 획득
        profile = self.pipeline.start(config)
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # YOLO 모델 로드 (가장 가벼운 nano 모델)
        self.model = YOLO('C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Baek/models/v4_yolov8n_last.pt')

        # 캘리브레이션 및 ROI 관련 변수 초기화
        self.wall_avg_depth = 0      # 캘리브레이션된 벽의 '평균' 뎁스를 저장
        self.is_calibrated = False   # 캘리브레이션 완료 여부 플래그
        
        self.roi_box = []            # 마우스로 그린 ROI 좌표 [x1, y1, x2, y2]
        self.drawing_box = False     # 현재 마우스를 드래그하여 박스를 그리고 있는지 여부

    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트를 처리하여 ROI를 그립니다."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_box = True
            self.roi_box = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:
                self.roi_box[2] = x
                self.roi_box[3] = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_box = False
            self.roi_box[2] = x
            self.roi_box[3] = y

    def _calibrate_wall(self, depth_frame):
        """지정된 ROI 영역의 평균 뎁스를 계산하여 벽으로 설정합니다."""
        if len(self.roi_box) != 4:
            print("Error: ROI not set. Please draw a box on the wall.")
            return

        # 좌표 정렬 (왼쪽-위 -> 오른쪽-아래)
        lx, rx = sorted([self.roi_box[0], self.roi_box[2]])
        ty, by = sorted([self.roi_box[1], self.roi_box[3]])

        if lx == rx or ty == by:
            print("Error: Invalid ROI size.")
            return

        depth_map = np.asanyarray(depth_frame.get_data())
        roi_depth_values = depth_map[ty:by, lx:rx]
        
        # 뎁스 값이 0인 (측정 실패) 픽셀은 제외하고 평균 계산
        valid_depths = roi_depth_values[roi_depth_values > 0]
        
        if valid_depths.size == 0:
            print("Error: No valid depth data in the selected ROI.")
            return

        self.wall_avg_depth = np.mean(valid_depths)
        self.is_calibrated = True
        print(f"Wall calibrated successfully with average depth: {self.wall_avg_depth:.0f}mm")

    def run(self):
        """메인 실행 함수"""
        window_name = '3D Ball Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback) # 마우스 콜백 등록

        try:
            while True:
                # RealSense에서 프레임 대기
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # 프레임을 NumPy 배열로 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                key = cv2.waitKey(1) & 0xFF
                
                # 캘리브레이션 UI 처리
                if not self.is_calibrated:
                    cv2.putText(color_image, "Draw a box on the wall, then press 'c'",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 마우스로 ROI 그리기 시각화
                    if self.drawing_box or len(self.roi_box) == 4:
                        cv2.rectangle(color_image, (self.roi_box[0], self.roi_box[1]),
                                      (self.roi_box[2], self.roi_box[3]), (0, 255, 0), 2)

                    if key == ord('c'):
                        self._calibrate_wall(depth_frame)
                
                # 'q'를 누르면 종료
                if key == ord('q'):
                    break
                
                # 캘리브레이션이 완료된 후에만 추적 및 충돌 감지 실행
                if self.is_calibrated:
                    # YOLO를 사용하여 객체 탐지 (sports ball 클래스 등)
                    results = self.model(color_image, classes=[32], conf=0.5, verbose=False)
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                # 객체 정보 추출
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                                
                                # 객체 중심의 뎁스 값 획득
                                depth = depth_image[cy, cx]

                                # 유효한 뎁스 값일 경우 처리
                                if depth > 0:
                                    # 평균 뎁스 값을 기준으로 벽까지의 거리 계산
                                    distance_to_wall = depth - self.wall_avg_depth
                                    
                                    # 충돌 감지 (5cm 이내)
                                    if abs(distance_to_wall) < 50:
                                        cv2.putText(color_image, "COLLISION!", (cx-50, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                                        # 충돌 지점에 빨간색 원 그리기
                                        cv2.circle(color_image, (cx, cy), 10, (0, 0, 255), -1)

                                    # 정보 텍스트 표시
                                    cv2.putText(color_image, f"To Wall: {distance_to_wall:.0f}mm", (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                                    # 객체 바운딩 박스 그리기
                                    cv2.rectangle(color_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

                # 뎁스 맵을 시각화용 컬러맵으로 변경
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 컬러 이미지와 뎁스 컬러맵을 합쳐서 표시
                images = np.hstack((color_image, depth_colormap))
                cv2.imshow(window_name, images)
                
        finally:
            # 스트리밍 중지 및 창 닫기
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = AdvancedDepthBallTracker()
    tracker.run()