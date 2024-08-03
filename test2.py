import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'ssd_mobilenet_v2_coco.caffemodel')

# Load the class labels
with open('coco.names', 'r') as f:
    class_labels = f.read().strip().split("\n")

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'video.mp4' for a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = f"{class_labels[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
