from ultralytics import YOLO

class VisionProcessor:
    """YOLO 모델 로드 및 객체 탐지를 담당합니다."""
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            exit(1)
    
    def detect_balls(self, image):
        detected_balls = []
        # 'sports ball' 클래스 ID는 32입니다.
        results = self.model(image, classes=[32], conf=0.4, verbose=False) 
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_balls.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (int((x1+x2)/2), int((y1+y2)/2)),
                    'confidence': float(box.conf[0])
                })
        return detected_balls