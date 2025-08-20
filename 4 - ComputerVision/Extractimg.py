import cv2
cap = cv2.VideoCapture(0) #Webcam

count = 0

while True:
    success, img = cap.read()
    cv2.imwrite(f'C:/Users/User/Documents/Git/Python-Yolo-ObjectDetection/Extract_Ball_Images/{count}.jpg', img)
    count += 1
    cv2.imshow("Original Image", img)
    cv2.waitKey(1)