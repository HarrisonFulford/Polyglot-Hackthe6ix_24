import tensorflow
import numpy
import cv2
import matplotlib

# define a video capture object 
vid = cv2.VideoCapture(0) 
frameno = 0
ssnum = 0
framesPerPhoto = 5 #How often a photo will be taken (Per frame)
photoType = '.jpg' #Photo type (png, jpg, etc)
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    if (frameno%framesPerPhoto == 0): 
        ssnum += 1
        name = 'screenshots/' + str(ssnum) + photoType
        print ('new frame captured...' + name)
        cv2.imwrite(name, frame)
        frameno = 0
    frameno += 1

    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 