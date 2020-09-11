import cv2
import numpy as np

def Processing(y):
    x = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    x = cv2.GaussianBlur(x,(5, 5), 3)
    return x

cap = cv2.VideoCapture('./Desktop/Openc/baggage.mp4')
ret, current_frame = cap.read()
current_frame = Processing(current_frame)
previous_frame = current_frame
m_prev = current_frame




while(cap.isOpened()):    
    m = cv2.add(cv2.multiply(0.9,m_prev), cv2.multiply(0.1,current_frame))
    frame_diff = cv2.absdiff(m, current_frame)
    new_frame = cv2.resize(frame_diff,(128, 128))
    new_frame2 = cv2.resize(current_frame,(128,128))
    ret2,th2 = cv2.threshold(new_frame,0,255,cv2.THRESH_BINARY)
    cv2.imshow('frame diff ',th2)
    cv2.imshow('video', new_frame2)      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    m_prev = m
    ret, current_frame = cap.read()
    current_frame = Processing(current_frame)
    

cap.release()
cv2.destroyAllWindows()
