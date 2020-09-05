import cv2
import numpy as np

cap = cv2.VideoCapture('./Desktop/Openc/baggage.mp4')
ret, current_frame = cap.read()
previous_frame = current_frame
while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    current_frame_gray = cv2.GaussianBlur(current_frame_gray,(5, 5), 3)
    previous_frame_gray = cv2.GaussianBlur(previous_frame_gray,(5, 5), 3)
    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    new_frame = cv2.resize(frame_diff,(1400, 720))
    ret2,th2 = cv2.threshold(new_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('frame diff ',th2)      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
    

cap.release()
cv2.destroyAllWindows()
