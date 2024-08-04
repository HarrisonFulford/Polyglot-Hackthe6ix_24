import pyscreenshot

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
    time.sleep(0.2)

    pic = pyscreenshot.grab(bbox=(58, 135, 285, 550))

    pic.save(img_path)

    print ('image ' + str(img_name_count) + ' saved')
    
    #adding one to the current image Id, so the picutres are sotred sequentally
    img_name_count += 1
    img_path = '/Users/harrisonfulford/downloads/Hackthe6ix_24/screenshots/' + str(img_name_count) + '.png'

