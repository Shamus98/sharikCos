import cv2 
import numpy as np

cap = cv2.VideoCapture('./Desktop/Openc/SVO_03-04/svo_oct05-04.mov')
#cap = cv2.VideoCapture('./Desktop/Openc/fish.mov')
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80, detectShadows=False)

ret, frame = cap.read()
ret, prevframe = cap.read()
ret, prevprevframe = cap.read()
while cap.isOpened():
    median = cv2.medianBlur(frame,41)
    median1 = cv2.medianBlur(prevframe,31)
    median2 = cv2.medianBlur(prevprevframe,19)

    blur1 = cv2.GaussianBlur(frame, (3,41), 20)
    blur2 = cv2.GaussianBlur(prevframe, (3,31), 20)
    blur3 = cv2.GaussianBlur(prevprevframe,(3,19), 20)
    #mask = subtractor.apply(mask)
    mask1 = subtractor.apply(median)
    mask2 = subtractor.apply(median1)
    mask3 = subtractor.apply(median2)

    mask1b = subtractor.apply(blur1)
    mask2b = subtractor.apply(blur2)
    mask3b = subtractor.apply(blur3)
    #dst = cv2.fastNlMeansDenoising(mask,None,10,7,11)
    frame1 = cv2.resize(median, (256,256))
    frame2 = cv2.resize(median1, (256,256))
    frame3 = cv2.resize(median2, (256,256))

    frame1b = cv2.resize(blur1, (256,256))
    frame2b = cv2.resize(blur2, (256,256))
    frame3b = cv2.resize(blur3, (256,256))

    cv2.imshow("median 41", frame1)
    cv2.imshow("median 31", frame2)
    cv2.imshow("median 19", frame3)

    cv2.imshow("blur (3,41), 20", frame1b)
    cv2.imshow("blur (3,31), 20", frame2b)
    cv2.imshow("blur (3,19), 20", frame3b)
    key = cv2.waitKey(30)
    if key == 27:
        break
    ret, frame = cap.read()
    ret, prevframe = cap.read()
    ret, prevprevframe = cap.read()
cap.release()
cv2.destroyAllWindows()

