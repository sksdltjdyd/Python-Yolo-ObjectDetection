# vision_processor.py
# YOLO 모델을 로드하고 이미지에서 객체를 탐지하는 역할

from ultralytics import YOLO
import cv2

class VisionProcessor:
    def __init__(self, model_path, confidence=0.7):
        self.confidence = confidence
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None

    def detect_ball(self, img):
        if not self.model: return img, []
        
        results = self.model(img, stream=False, verbose=False, conf=self.confidence)
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    conf = float(box.conf[0]) # 신뢰도 값 가져오기
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # --- ✨ 수정된 부분: 'confidence' 키 추가 ---
                    detections.append({
                        'center': center, 
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf  # 여기에 신뢰도 값을 추가합니다.
                    })
        return img, detections