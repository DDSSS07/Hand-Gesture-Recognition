import cv2
import imutils

# global variables
background = None
cam=0

def average(image, avg_weight):
    global background
    #Create background
    if background is None:
        background = image.copy().astype("float")
        return
    #Weighted average 
    cv2.accumulateWeighted(image, background, avg_weight)

def segmentation(image, threshold=25):
    global background
    # Diff. bet. background and hand_region
    difference = cv2.absdiff(background.astype("uint8"), image)

    # Threshold the diff to obtain the foreground
    thresholded = cv2.threshold(difference,threshold,255,cv2.THRESH_BINARY)[1]

    #Contours in the thresholded image
    (_, contours, _) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # When there are no contours
    if len(contours) == 0:
        return
    else:
        # Using contour area, get the hand_region (max. contour)
        segment = max(contours, key=cv2.contourArea)
        return (thresholded, segment)

# free up memory
def release():
    cam.release()
    cv2.destroyAllWindows()

def main():
    # Weight for finding average
    avg_weight = 0.5
    global cam
    # To open Camera
    cam = cv2.VideoCapture(0)   
    top, right, bottom, left = 10, 350, 225, 590
    
    frames = 0
    images = 0

    start = False

    while(True):
        # Current cam_frame
        (captured, cam_frame) = cam.read()
        if (captured == True):
            # Resize
            cam_frame = imutils.resize(cam_frame, width=700)
            # To avoid inverted view
            cam_frame = cv2.flip(cam_frame, 1)
            copy = cam_frame.copy()
            (height, width) = cam_frame.shape[:2]
            # Region of interest
            region = cam_frame[top:bottom, right:left]
            #Grayscale 
            grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
     
            #Gaussian Blur
            grayscale = cv2.GaussianBlur(grayscale, (7, 7), 0)

            #Frame Calibration
            if frames < 30:
                average(grayscale, avg_weight)
                print(frames)
            else:
                #Hand Segmentation
                hand_region = segmentation(grayscale)

                if hand_region is not None:                 
                    (thresholded, segmented) = hand_region

                    # Draw and display the frame
                    cv2.drawContours(copy, [segmented + (right, top)], -1, (0, 0, 255))
                    
                    if start:
                        cv2.imwrite("Dataset/PlamTest/palm_" + str(images) + '.png', thresholded)
                        images += 1
                    cv2.imshow("Threshold", thresholded)

            # Segmented hand_region
            cv2.rectangle(copy, (left, top), (right, bottom), (0,255,0), 2)

            frames += 1
            # Frame display
            cv2.imshow('Vid feed', copy)
            key = cv2.waitKey(1) & 0xFF
            # Quit
            if key == ord('q') or images > 100:
                break        
            if key == ord('s'):
                start = True
        else:
            print("Error:Check the camera")
            break
main()
release()
