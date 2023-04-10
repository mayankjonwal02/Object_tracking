import cv2
from matplotlib import axis


cap = cv2.VideoCapture(0)

# object detector from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    #  object detection
    mask = object_detector.apply(frame)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours,_ =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #  calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:

            # cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
             x,y,w,h = cv2.boundingRect(cnt)
             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
     # cv2.imshow("mask",mask)
    cv2.imshow("frame",frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
