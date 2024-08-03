# importing OpenCV(cv2) module for image reading
import cv2
# importing os module to allow for file existance checks without try blocks
import os
# importing time to allow for main loop managment
import time

# Save image in set directory
# Read RGB image
img_name_count = 1

# image path variable, set to a string representing the image I am currently looking for
img_path = '/Users/harrisonfulford/downloads/Hackthe6ix_24/screenshots/' + str(img_name_count) + '.png'

# initializing this variable
condition = True

# main loop for ML identifying function, should run at all times 
while condition == True:
    # waiting at the start of each loop so the consol doesn't fill up 
    time.sleep(0.5)

    # setting a value to the state of the file we are waiting for  
    path_state = os.path.exists(img_path)
    
    # if the file exists, run the image recognition function, otherwise continue the loop
    if path_state == True:
            
            # variable check on file state to help identify possible error points
            print("File " + str(img_name_count) + " exists")
            
            # adding to the counter to decide what file we check for next
            img_name_count += 1
            img_path = '/Users/harrisonfulford/downloads/Hackthe6ix_24/screenshots/' + str(img_name_count) + '.png'

    # if there is no available file, loop again
    else:
        print("File does not exist")

 