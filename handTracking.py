import cv2
import mediapipe as mp
from fruitRecognitionTraining import checkImg

screenWidth = int(1512)
screenHeight = int(2016)
drawingMp = mp.solutions.drawing_utils
handMp = mp.solutions.hands
hand = handMp.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6)
# define a video capture object 
# (x, y, width, height) = cv2.getWindowImageRect('Frame')
vid = cv2.VideoCapture(0) 
vid.set(3, 1280)
frameno = 0
ssnum = 0
framesPerPhoto = 3 #How often a photo will be taken (Per frame)
photoType = '.jpg' #Photo type (png, jpg, etc)
bottomAxis = 0
topAxis = 0
leftAxis = 0
rightAxis = 0
height = 256
width = 256

def transferAxisCordInfo():
    return bottomAxis, topAxis, leftAxis, rightAxis

while(True): 
    # Capture the video frame by frame 
    _, frame = vid.read() 
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGBframe)
    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            name = "screenshots/currentFrame" + photoType
            print ('New frame captured')
            cv2.imwrite(name, frame)
            #print(handLandmarks)
            indexTip = handLandmarks.landmark[8]
            print("X:" + str(indexTip.x*screenWidth))
            print("Y:" + str(indexTip.y*screenHeight))
            bottomAxis = int((indexTip.y*screenHeight)*0.357)
            topAxis = int(bottomAxis-height)
            leftAxis = int((indexTip.x*screenWidth-(width/2))*0.83)
            rightAxis = int((indexTip.x*screenWidth+(width/2))*0.83)
            cv2.rectangle(frame, (rightAxis, bottomAxis), (leftAxis, topAxis), (0, 255, 0), 2)
            drawingMp.draw_landmarks(frame, handLandmarks, handMp.HAND_CONNECTIONS)
            checkImg()
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