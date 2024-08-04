# pip install opencv-python
# pip install mediapipe
import cv2
import mediapipe as mp

vid = cv2.VideoCapture(0)
vid.set(3, 1280)
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6 )
mp_draw = mp.solutions.drawing_utils
height = 256
width = 128
screenWidth = int(285)
screenHeight = int(550)
while True :
    _, frame = vid.read()
    # convert from bgr to rgb
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGBframe)
    if result.multi_hand_landmarks:
        #print("hand found")
        for handLandmarks in result.multi_hand_landmarks :
            #print(handLm)
            indexTip = handLandmarks.landmark[8]
            bottomAxis = int((indexTip.y*screenHeight)*0.357)
            topAxis = int(bottomAxis-height)
            leftAxis = int((indexTip.x*screenWidth-(width/2))*0.83)
            rightAxis = int((indexTip.x*screenWidth+(width/2))*0.83)
            cv2.rectangle(frame, (rightAxis, bottomAxis), (leftAxis, topAxis), (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, handLandmarks, mp_hands.HAND_CONNECTIONS)

    #cv2.imshow("rgbvideo", RGBframe)
    cv2.imshow("video", frame)
    cv2.waitKey(1)