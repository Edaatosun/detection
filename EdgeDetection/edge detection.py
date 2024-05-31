import cv2
import numpy as np
img = cv2.imread('cicek.jpg')

# gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = gray
cv2.imshow("bu sen", img)
# filters
blur = cv2.GaussianBlur(gray,(3,3),0)
#cv2.imshow("blur", blur) # Gaussian filtresi

mean_filter = cv2.blur(gray, (3,3))
#cv2.imshow("mean", mean_filter )

median_filter = cv2.medianBlur(gray,3)
#cv2.imshow("median", median_filter )  

sobel1 = cv2.Sobel(blur , cv2.CV_64F,0,1,ksize = 5)
sobel2 = cv2.Sobel(blur, cv2.CV_64F,1,0,ksize = 5)
Sobel = sobel1 + sobel2
cv2.imshow("sobel", Sobel ) 

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
 
img_previt1 = cv2.filter2D(blur,-1, kernelx)
img_previt2 = cv2.filter2D(blur,-1, kernely)
img_previt = img_previt1 +img_previt2
cv2.imshow("kernel", img_previt) 
                       
cv2.waitKey(0)
