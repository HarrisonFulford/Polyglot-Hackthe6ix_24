# Import packages
import cv2
import numpy as np
 
img = cv2.imread('screenshots\currentFrame.jpg')
cv2.imshow("original", img)
# Cropping an image
cropped_image = img[80:280, 150:330] 
# Save the cropped image
cv2.imwrite("screenshots\currentFrame.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()