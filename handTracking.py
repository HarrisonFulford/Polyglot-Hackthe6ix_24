import cv2
import numpy as np
import mediapipe as mp
from fruitRecognitionRead import checkImg
import pyscreenshot

# importing time to allow for main loop managment
import time

# Save image in set directory
# Read RGB image
# img_name_count = 1

# image path variable, set to a string representing the image I am currently looking for
img_path = '/Users/jinwoo/Desktop/hackthe6ix2024/Hackthe6ix_24/screenshots/currentFrame.png'

# initializing this variable
condition = True
vid = cv2.VideoCapture("screenshots/currentFrame.png")
vid.set(3, 1280)
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6 )
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
height = 512
width = 512
screenWidth = int(285*6.8)
screenHeight = int(550*2)

frameno = 0
framesPerPhoto = 120 #How often a photo will be taken (Per frame)
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

    # main loop for ML identifying function, should run at all times 
while condition == True:
        # waiting at the start of each loop so the consol doesn't fill up 
    time.sleep(0.2)

    pic = pyscreenshot.grab(bbox=(40, 150, 280, 540))

    pic.save(img_path)

    # For static images:
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(img_path), 1)
        # Convert the BGR image to RGB before processing.
        result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            #print("hand found")
            for handLandmarks in result.multi_hand_landmarks :
                indexTip = handLandmarks.landmark[8]
                bottomAxis = int((indexTip.y*screenHeight))
                topAxis = int(bottomAxis-height)
                leftAxis = int((indexTip.x*screenWidth-(width/2)))
                rightAxis = int((indexTip.x*screenWidth+(width/2)))
                print("X:" + str(indexTip.x))
                print("Y:" + str(indexTip.y))
                cv2.rectangle(image, (rightAxis, bottomAxis), (leftAxis, topAxis), (0, 255, 0), 2)
                mp_draw.draw_landmarks(image, handLandmarks, mp_hands.HAND_CONNECTIONS)
                cropImg()
                print(checkImg())
                
        # Display the resulting frame 
    cv2.imshow('video', image) 
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