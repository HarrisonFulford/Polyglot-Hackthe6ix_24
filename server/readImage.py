import cv2
 
imSeq = cv2.VideoCapture("screenshots/currentFrame.jpg")
 
if(imSeq.isOpened() == False):
    print("Error opening the video")
 
while(imSeq.isOpened()):
    ret, frame = imSeq.read()
 
    if ret == True:
        cv2.imshow("Frames",frame)
 
        if cv2.waitKey(25):
            breakpoint()
    else:
        break
 
imSeq.release()
cv2.destroyAllWindows()