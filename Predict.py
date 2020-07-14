import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils

background = None

def resize(image):
    width = 100
    img = Image.open(image)
    wpercent = (width/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((width,hsize), Image.ANTIALIAS)
    img.save(image)

def average(image, avg_weight):
    
    global background
    # Create background
    if background is None:
        background = image.copy().astype("float")
        return

    # Weighted average
    cv2.accumulateWeighted(image, background, avg_weight)

def segmentation(image, threshold=25):
    global background
    # Diff. bet. background and hand_region
    diff = cv2.absdiff(background.astype("uint8"), image)

    # Threshold the diff to obtain the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Contours in the thresholded image
    (_,contours,_) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # When there are no contours
    if len(contours) == 0:
        return
    else:
        # Using contour area, get the hand_region (max. contour)
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # Weight for finding average
    avg_weight = 0.5

    # To open Camera
    cam = cv2.VideoCapture(0)

    # Region of interest
    top, right, bottom, left = 10, 350, 225, 590

    # Initialize no of frames
    frames = 0
    start = False

    #Until quitting 
    while(True):
        # Current frame
        (captured, frame) = cam.read()

        # Resize frame
        frame = imutils.resize(frame, width = 700)

        # Flip to avoid inverted view
        frame = cv2.flip(frame, 1)

        # Frame copy
        copy = frame.copy()
        
        (height, width) = frame.shape[:2]

        # Region of interest
        region = frame[top:bottom, right:left]

        # Grayscale
        grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        #Gaussian Blur
        grayscale = cv2.GaussianBlur(grayscale, (7, 7), 0)

        # Frame Calibration
        if frames < 30:
            average(grayscale, avg_weight)
        else:
            # Hand segmentation
            hand_region = segmentation(grayscale)
            
            if hand_region is not None:
                # Unpacking
                (thresholded, segmented) = hand_region

                # Draw and display the frame
                cv2.drawContours(copy, [segmented + (right, top)], -1, (0, 0, 255))
                if start:
                    cv2.imwrite('Capture.png', thresholded)
                    resize('Capture.png')
                    prediction_result, probability = predictGesture()
                    showResults(prediction_result, probability)
                cv2.imshow("Theshold", thresholded)

        # Segmented hand_region
        cv2.rectangle(copy, (left, top), (right, bottom), (0,255,0), 2)
        
        frames += 1

        # Frame with segmented hand_region
        cv2.imshow('Video Feed', copy)

        # User input
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break
        
        if key == ord('s'):
            start = True

def predictGesture():
    # Predict
    image = cv2.imread('Capture.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction_result = model.predict([gray_img.reshape(89, 100, 1)])
    return np.argmax(prediction_result), (np.amax(prediction_result) / (prediction_result[0][0] + prediction_result[0][1] + prediction_result[0][2]))

def showResults(prediction_result, probability):

    text = np.zeros((300,512,3), np.uint8)
    classification = ""

    if prediction_result == 0:
        classification = "Swing"
    elif prediction_result == 1:
        classification = "Palm"
    elif prediction_result == 2:
        classification = "Fist"

    cv2.putText(text,"Predicted Class : " + classification, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(text,"Probability : " + str(probability * 100) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Statistics :", text)

# Creating CNN Model
tf.reset_default_graph()

# Creating Convolution Layer
network=input_data(shape=[None,89,100,1],name='input')
network=conv_2d(network,32,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,64,2,activation='relu')
network=max_pool_2d(network,2)

network=conv_2d(network,128,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,256,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,256,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,128,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,64,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

# Fully connected layer
network=fully_connected(network,1000,activation='relu')

# Apply Dropout
network=dropout(network,0.75)

# Fully connected layer
network=fully_connected(network,3,activation='softmax')

network=regression(network,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(network,tensorboard_verbose=0)

# Loading the trained model
model.load("TrainedModel/Hand_Gesture_Recognition.tfl")

main()
