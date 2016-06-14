import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Threshold the image to get binary inverted image
thresh = cv2.adaptiveThreshold(~gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-2)


# fill the broken lines
kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(1 , 5))
thresh = cv2.dilate(thresh,kernal, iterations = 2)
thresh = cv2.erode(thresh,kernal, iterations = 1)

# Make a copy for horizontal and vertical
horz = np.copy(thresh)
vert = np.copy(thresh)

# Set the kernel
(_,horzcol) = np.shape(horz)
horzSize = horzcol/26
horzStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horzSize,1))

# perform a series of erosions and dilations
horz = cv2.erode(horz,horzStructure, iterations = 1)
horz = cv2.dilate(horz,horzStructure, iterations = 1)

# Get the output image after inversion
outimage = cv2.bitwise_and(~gray,~horz)
outimage = ~outimage
cv2.imwrite("lineout.png", outimage)
'''
edges = cv2.Canny(outimage,50,150,apertureSize = 3)
minLineLength = 5
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
#print lines
for x1,y1,x2,y2 in lines[0]:
    cv2.line(outimage,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.png',outimage)
'''

