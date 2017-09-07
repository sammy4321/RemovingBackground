import numpy as np
import cv2

cap = cv2.VideoCapture(1)
global backGroundImageSave
 
while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray,(15,15),0)
    #cv2.line(gray,(0,0),(150,150),(0,255,0),3)
 
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	global backGroundImageSave
    	backGroundImageSave = gray
        break

#cap.release()
#cv2.destroyAllWindows()

while(True):
    ret1, frame1 = cap.read()
    max_limit=120
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #gray1 = cv2.GaussianBlur(gray1,(15,15),0)
    kernel = np.ones((2,2),np.uint8)
    #mask=np.bitwise_xor(gray1,backGroundImageSave)
    mask=np.subtract(gray1,backGroundImageSave)
    mask[mask<=max_limit]=0
    #mask[mask>=-max_limit]=0
    
    #mask=cv2.erode(mask,kernel,iterations = 5)
    #dilation = cv2.dilate(mask,kernel,iterations = 3)
    retval,threshold = cv2.threshold(mask,50,255,cv2.THRESH_BINARY_INV)

 
    cv2.imshow('frame',gray1)
    cv2.imshow('Detected',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	
        break

cap.release()
cv2.destroyAllWindows()