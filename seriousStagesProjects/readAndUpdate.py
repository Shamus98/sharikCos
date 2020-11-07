import cv2 
import numpy as np
import time
import os
#ffmpeg  -r 30 -i ./SVO-03-04-in/%04d.png -c:v libx264 -pix_fmt yuv420p SVO-03-04.mp4
#cap = cv2.VideoCapture('./Desktop/Openc/baggage.mov')
cap = cv2.VideoCapture('./Desktop/Openc/SVO_03-04/svo_oct05-04.mov')
ret, frame1 = cap.read()
ret, frame2 = cap.read()
pts = []
k = 0
boole = False
count = 0
files = 0
Count = 0
countValues = 0
os.mkdir('./Desktop/Openc/SVO-20201005-DXXX-04/SVO-20201005-DXXX-04-0000')
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2] + boxes[:,0]
	y2 = boxes[:,3] + boxes[:,1]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
while cap.isOpened():
    k += 1
    if k == 1:
      fac = frame1
    #cv2.line(frame1, (100, 0), (100, 1024), (255, 0, 255), 10)
    diff = cv2.absdiff(frame1, fac)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
    #dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    #for i, c in enumerate(contours):
    #    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #   boundRect[i] = cv2.boundingRect(contours_poly[i])

    for i, c in enumerate(contours):
      boundRect[i] = cv2.boundingRect(c)

    color = (255,0,255)
    countValues += 1 
    maxbox = non_max_suppression_fast(np.array(boundRect), .95)
    for i in range(len(maxbox)):
      if maxbox[i][2] > 40 and maxbox[i][3] > 40 and not(maxbox[i][0]>0 and maxbox[i][0] <= 400 and maxbox[i][1]>0 and maxbox[i][1]<=400):
        #cv2.rectangle(frame1, (maxbox[i][0], maxbox[i][1]), \
        #      (maxbox[i][0]+maxbox[i][2], maxbox[i][1]+maxbox[i][3]), color, 8) 
        if maxbox[i][0] < 200 and boole == False:
          count += 1
          boole = True
        elif maxbox[i][0] > 400 and boole == True:
          boole = False
        frame256 = cv2.resize(frame1, (512,512))
        frame228 = cv2.resize(thresh, (512,512))
        #cv2.putText(frame256, str(count), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("feed", frame256)
        cv2.imshow("thresh", frame228)
        if countValues > 60:
            print("Следующее видео" + str(files))
            os.mkdir('./Desktop/Openc/SVO-20201005-DXXX-04/SVO-20201005-DXXX-04-{:04d}'.format(files+1))
            files += 1 
            Count = 0
            print(countValues)
        path = './Desktop/Openc/SVO-20201005-DXXX-04/SVO-20201005-DXXX-04-{:04d}/frame{:d}.jpg'.format(files, Count)
        cv2.imwrite(path, frame1)
        Count += 1
        #cap.set(1, Count)
        countValues = 0      
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(files)
cv2.destroyAllWindows()
cap.release()