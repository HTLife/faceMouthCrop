import cv2
import sys
import glob 
import numpy as np

def main():
    cascPath_face = "frontalface.xml"
    cascPath_mouth = "mouth.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath_face)
    mouthCascade = cv2.CascadeClassifier(cascPath_mouth)

    files=glob.glob("*.png")   
    for file in files:

        # Read the image
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        region = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        print "Found {0} region!".format(len(region))
        if (len(region) == 0):
            region = mouthCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            #cv2.imwrite("./out/cropped_{1}_{0}".format(str(file),str(x)), image)
            #continue

        # Crop Padding
        left = 50
        right = 50
        top = 50
        bottom = 50
        
        img_w,img_h,bpp = np.shape(image)
        

        # Draw a rectangle around the region
        for (x, y, w, h) in region:
            print "x,y,w,h"
            print x, y, w, h

            # Dubugging boxes
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        image  = image[y-top:y+w+bottom, x-left:x+w+right]
        #shift = h*3/5


        #image  = image[y:y+h, x:x+w]
        

        print "cropped_{1}{0}".format(str(file),str(x))
        cv2.imwrite("./out/cropped_{1}_{0}".format(str(file),str(x)), image)

if __name__ == "__main__":
    main()    