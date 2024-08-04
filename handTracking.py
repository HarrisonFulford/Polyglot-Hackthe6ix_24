import cv2
import numpy as np
import mediapipe as mp
from fruitRecognitionRead import checkImg

vid = cv2.VideoCapture("screenshots/currentFrame.jpg")
vid.set(3, 1280)
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6 )
mp_draw = mp.solutions.drawing_utils
height = 512
width = 512
screenWidth = int(285*6.8)
screenHeight = int(550*2)

frameno = 0
framesPerPhoto = 30 #How often a photo will be taken (Per frame)
photoType = '.jpg' #Photo type (png, jpg, etc)

def transferAxisCordInfo():
    return bottomAxis, topAxis, leftAxis, rightAxis

def cropImg():
    # Import packages    
    img = cv2.imread("screenshots/currentFrame" + photoType)
    #cv2.imshow("original", img)
    # Cropping an image
    print(int(indexTip.x*screenWidth))
    print(int(indexTip.y*screenHeight))
    cropped_image = img[topAxis:bottomAxis, leftAxis:rightAxis] 
    # Save the cropped image
    if cropped_image is not None and cropped_image.size > 0:
        cv2.imwrite("screenshots/currentFrameCropped" + photoType, cropped_image)
        print("Saved crop")
    else:
        print("Error: cropped_image is empty")
    return "A"

while (True):
    _, frame = vid.read()
    # convert from bgr to rgb
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGBframe)
    if result.multi_hand_landmarks:
        #print("hand found")
        for handLandmarks in result.multi_hand_landmarks :
            name = "screenshots/currentFrame" + photoType
            cv2.imwrite(name, frame)
            indexTip = handLandmarks.landmark[8]
            bottomAxis = int((indexTip.y*screenHeight))
            topAxis = int(bottomAxis-height)
            leftAxis = int((indexTip.x*screenWidth-(width/2)))
            rightAxis = int((indexTip.x*screenWidth+(width/2)))
            print("X:" + str(indexTip.x))
            print("Y:" + str(indexTip.y))
            cv2.rectangle(frame, (rightAxis, bottomAxis), (leftAxis, topAxis), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, handLandmarks, mp_hands.HAND_CONNECTIONS)
            cropImg()
            print(checkImg())
            
    # Display the resulting frame 
    cv2.imshow('video', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    #ssnum += 1
    #name = 'screenshots/' + str(ssnum) + photoType
    #print ('new frame captured...' + name)
    #cv2.imwrite(name, frame)
    #frameno = 0

    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 