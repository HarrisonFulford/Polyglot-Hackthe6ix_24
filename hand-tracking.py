import cv2
import mediapipe as mp

drawingMp = mp.solutions.drawing_utils
handMp = mp.solutions.hands
hand = handMp.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6)
# define a video capture object 
vid = cv2.VideoCapture(0) 
vid.set(3, 1280)
frameno = 0
ssnum = 0
framesPerPhoto = 5 #How often a photo will be taken (Per frame)
photoType = '.jpg' #Photo type (png, jpg, etc)
while(True): 
      
    # Capture the video frame 
    # by frame 
    _, frame = vid.read() 
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGBframe)
    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            #print(handLandmarks)
            drawingMp.draw_landmarks(frame, handLandmarks, handMp.HAND_CONNECTIONS)
            indexTip = handLandmarks.landmark[8]
            print("X:" + indexTip.x)
            print("Y:" + indexTip.y)

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