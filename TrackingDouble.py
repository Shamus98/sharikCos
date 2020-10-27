import cv2 
import numpy as np
import time
cap = cv2.VideoCapture('./Desktop/Openc/fish.mov')
#cap = cv2.VideoCapture('./Desktop/Openc/baggage.mov')
ret, frame1 = cap.read()
ret, frame2 = cap.read()
pts = []
def non_max_suppression_fast(boxes, overlapThresh):
  if len(boxes) == 0:
    return []

  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
  pick = []

  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(area)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]

    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    overlap = (w * h) / area[idxs[:last]]
    ### Modification
    selected_idx = np.where(overlap > overlapThresh)[0]
    selected_idx = np.concatenate(([last], selected_idx))
    min_area_idx = min(selected_idx, key=lambda i: area[i])
    pick.append(min_area_idx)
    ### Modification end
    idxs = np.delete(idxs, selected_idx)

  return boxes[pick].astype("int")
while cap.isOpened():
    #cv2.line(frame1, (100, 0), (100, 1024), (255, 0, 255), 10)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    color = (255,0,255)
    maxbox = non_max_suppression_fast(np.array(boundRect), .85)
    for i in range(len(maxbox)):
      cv2.rectangle(frame1, (maxbox[i][0], maxbox[i][1]), \
              (maxbox[i][0]+maxbox[i][2], maxbox[i][1]+maxbox[i][3]), color, 8)
    frame256 = cv2.resize(frame1, (256,256))
    cv2.imshow("feed", frame256)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()