
import numpy as np 
import math
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

class LineDetector:
    thresh = 60
    maxValue = 255
    horizontalRes = 32
    verticalRes = 24
    currentImage = None

    hough_minLineLength = 10
    hough_maxLineGap = 0
    
    #picamera values
    __cam_xres = 320
    __cam_yres = 240

    def __init__(self,turn_precision):
        self.turn_res = turn_precision

        #Initialize raspberry picamera!
        self.__camera = PiCamera()
        #self.__camera.rotation = 90
        self.__camera.vflip = True
        self.__camera.hflip = True
        self.__camera.resolution = (self.__cam_xres, self.__cam_yres)
        self.__rawCapture = PiRGBArray(self.__camera, size=(self.__cam_xres,
            self.__cam_yres))
        time.sleep(0.1) #allow the camera to warm up

    def getTurnDir(self):
        
        line_xval = self.__findLine()
        if(line_xval >= 0.0): #Line detected, proceed calculate turn signal
            divisor = float(self.turn_res) / self.horizontalRes
            return int(math.floor(
                (line_xval * divisor) + 0.5)) - (self.turn_res / 2)
        else: #No line detected so return landing signal
            return self.turn_res;

    def __grabImage(self):
        self.__camera.capture(self.__rawCapture, format="bgr")
        self.currentImage = self.__rawCapture.array
        #Clear rawCapture variable
        self.__rawCapture = PiRGBArray(self.__camera, size=(self.__cam_xres,
            self.__cam_yres))
    
    def __findLine(self):
        self.__grabImage();

        if(self.currentImage is None):
            #grabbing image failed
            return -2.0

        #Convert to Grayscale
        img = cv2.cvtColor(self.currentImage, cv2.COLOR_BGR2GRAY)
        
        #Blur to reduce noise
        img = cv2.medianBlur(img,29)

        #Do Thresholding
        h,img = cv2.threshold(img, self.thresh, self.maxValue, cv2.THRESH_BINARY_INV)

        img = cv2.blur(img,(2,2))

        #Make image smaller
        img = cv2.resize(img, (self.horizontalRes, self.verticalRes))
        #org_img = cv2.resize(org_img, (self.horizontalRes, self.verticalRes))

        #Create skeleton
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False
        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True

        #Do Line Detection
        lines = cv2.HoughLinesP(skel,1,np.pi/180,2,
                self.hough_minLineLength,self.hough_maxLineGap)


        #get minimum and maximum x-coordinate from lines
        x_min = self.horizontalRes+1.0
        x_max = -1.0;
        if(lines is not None and len(lines[0]) > 0):
            for x1,y1,x2,y2 in lines[0]:
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
                #cv2.line(org_img,(x1,y1),(x2,y2),(0,255,0),2)

        #write output visualization
        #cv2.imwrite("output-img.png",org_img);

        #find the middle point x of the line and return
        #return -1.0 if no lines found
        if(x_max == -1.0 or x_min == (self.horizontalRes+1.0) ):
            return -1.0 #no line found
        else:
            return (x_min + x_max) / 2.0
        
