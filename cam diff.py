import time
import cv2
import numpy as np

cap = cv2.VideoCapture(2)
f = None
threshold2 = None
frames = []

def on_threshold_change(value):
    global threshold_value
    threshold_value = value


cv2.namedWindow('Threshold Slider')


threshold_value = 60 
cv2.createTrackbar('Threshold', 'Threshold Slider', threshold_value, 150, on_threshold_change)

while True:
    ret, frame = cap.read()
    if f is None:
        f = frame.copy()

    gray1 = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    gray2 = np.dot(f[...,:3], [0.299, 0.587, 0.114])
    
    diff = np.abs(gray1 - gray2)
    
    threshold = ((diff > threshold_value) & (diff < 2 * threshold_value)).astype(np.uint8)
    
    kernel = np.ones((2, 2), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    
    if threshold2 is None:
        threshold2 = threshold.copy()
        
    if np.all(threshold == 0):
        threshold = threshold2

    threshold2=threshold
    cv2.imshow('Webcam 1', threshold * 255)
        
    frame1 = cv2.add(frame, cv2.cvtColor(threshold * 255, cv2.COLOR_GRAY2BGR))
    cv2.imshow('contours', frame1)

    f = frame.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
