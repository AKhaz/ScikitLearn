from posixpath import split
import numpy as np
import argparse
import cv2
import os


#Open every image in a given folder
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--Folder", required=True, help="Folder Path")
args = vars(ap.parse_args())

for filename in os.listdir(args["Folder"]):
    print(args["Folder"] + '\\' + filename)
    image = cv2.imread(args["Folder"] + "\\" + filename)
    cv2.imshow("Original", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    cannyEdge = cv2.Canny(blurred, 30, 150)
    #cv2.imshow("Canny Edge Detection", cannyEdge)
    (cnts, _ ) = cv2.findContours(cannyEdge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Save data to a file named after the parent folder
    #Get the name of the final folder in the directory
    folderName = os.path.basename(args["Folder"])
    newFileName = filename.split('.')[0]
    newFileName = folderName + newFileName[5:]
    print(newFileName)
    print(args["Folder"] + "\\" + newFileName + ".txt")
    fileOpen = open(args["Folder"] + "\\" + newFileName + ".txt", "a")

    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        #print("Contour #" + str(i) + "||X:" + str(x) + "||Y:" + str(y) + "||Width:" + str(w) + "||Height:" + str(h))
        print(f"{x},{y},{w},{h}", file = fileOpen)
        # original = image.copy()
        # cv2.drawContours(original, cnts, -1, (0, 255, 0), 2)
        # cv2.imshow("Contours", original)
        # cv2.waitKey(0)