import cv2
import numpy as np
import pickle
import pyrealsense2 as rs

# --- 1. 기본 설정 ---
width, height = 640, 480
filename = 'realsense_calibration_data.p' # 리얼센스용 파일 이름 변경
circles = np.zeros((4, 2), int)
counter = 0

# --- 2. RealSense 파이프라인 초기화 ---
pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림을 설정합니다. (BGR 포맷, 30fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)
print("RealSense 카메라가 시작되었습니다. 'Original Image' 창에 4개의 점을 클릭하세요.")
print("클릭 순서: 좌상단 -> 우상단 -> 좌하단 -> 우하단")

def mousePoints(event, x, y, flags, param):
    """마우스 클릭 이벤트를 처리하여 4개의 점 좌표를 저장합니다."""
    global counter
    # 4개 이상의 점이 찍히지 않도록 방지
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        circles[counter] = (x, y)
        counter += 1
        print("클릭된 좌표:")
        print(circles)

# --- 3. 메인 루프 ---
try:
    while True:
        # --- RealSense에서 프레임 읽기 ---
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue # 프레임이 없으면 다음 루프로 넘어감

        # 프레임을 OpenCV에서 사용할 수 있는 NumPy 배열로 변환
        img = np.asanyarray(color_frame.get_data())

        # 4개의 점이 모두 찍혔을 때 투시 변환 실행
        if counter == 4:
            # 캘리브레이션 데이터 저장
            with open(filename, 'wb') as fileobj:
                pickle.dump(circles, fileobj)
            
            # 투시 변환을 위한 원본 좌표와 목적지 좌표 설정
            # 중요: 클릭 순서(좌상단, 우상단, 좌하단, 우하단)를 지켜야 올바른 결과가 나옵니다.
            pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            
            # 변환 행렬 계산 및 적용
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgOutput = cv2.warpPerspective(img, matrix, (width, height))
            
            # 변환된 이미지 표시
            cv2.imshow("Output Image", imgOutput)

        # 클릭한 점들을 원본 이미지에 그리기
        for x in range(counter):
            cv2.circle(img, (circles[x][0], circles[x][1]), 5, (0, 255, 0), cv2.FILLED)
        
        # 원본 이미지 표시 및 마우스 콜백 설정
        cv2.imshow("Original Image", img)
        cv2.setMouseCallback("Original Image", mousePoints)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- 4. 리소스 정리 ---
    print("리얼센스 카메라를 중지합니다...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"캘리브레이션 좌표가 '{filename}' 파일에 저장되었습니다.")

   
calibrationFilePath = 'C:\Users\User\Documents\Git\Python-Yolo-ObjectDetection\realsense_calibration_data.p'