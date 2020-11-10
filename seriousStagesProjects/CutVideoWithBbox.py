import cv2 
import numpy as np
import time
#cap = cv2.VideoCapture('./Desktop/Openc/baggage.mov')
cap = cv2.VideoCapture('./Desktop/Openc/SVO_03-04/svo_oct05-04.mov')
ret, frame1 = cap.read()
first_frame_count = 0



########################################################################################################
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
###########################################################################



while cap.isOpened():
    maxbox_if = []
    first_frame_count += 1
    if first_frame_count == 1:
      fac = frame1
    #cv2.line(frame1, (100, 0), (100, 1024), (255, 0, 255), 10)
    diff = cv2.absdiff(frame1, fac)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
    #dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        boundRect[i] = cv2.boundingRect(c)


    color = (255,0,255)
    maxbox = non_max_suppression_fast(np.array(boundRect), .95)

    for i in range(len(maxbox)):
      if (maxbox[i][2] > 20 and maxbox[i][3] > 20):
          maxbox_if.append(maxbox[i])

    for i in range(len(maxbox_if)):
        cv2.rectangle(frame1, (maxbox_if[i][0], maxbox_if[i][1]), \
            (maxbox_if[i][0]+maxbox_if[i][2], maxbox_if[i][1]+maxbox_if[i][3]), color, 4)       

    if len(maxbox_if) > 0:
        showBbox = cv2.resize(frame1, (512,512))
        thresh1 = cv2.resize(thresh, (512,512))
        cv2.imshow("thresh", thresh1)
        cv2.imshow("BBox", showBbox)

    ret, frame1 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
