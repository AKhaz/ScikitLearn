# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from pandas import read_csv
import re
import argparse
import numpy as np

# matplotlib 3.3.1
from matplotlib import pyplot
import os

#Use pandas to convert to a numpy array

#Open all .txt files in a directory
#For every .txt in a folder
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--Folder", required=True, help="Folder Path")
args = vars(ap.parse_args())

imageX = []
imageY = []

for folder in os.listdir(args["Folder"]):
    for filename in os.listdir(args["Folder"] + "\\" + folder):
        #find all files in the directory that end with .txt
        if filename.endswith(".txt"):
            imageArray = read_csv(args["Folder"] + "\\" + folder + "\\" + filename)
            imageArray = imageArray.values.tolist()
            #print("List sorted by w and h:")
            #Sorting the list by the magnitude of w and h to only get tshe largest contour coordinates
            imageArray.sort(key = lambda l: int(l[2])**2 + int(l[3])**2)
            #print(imageArray)
            #print("And then the shortened list:")
            #Remove any values from the imageArrays list based on a set list length (of 10, in this instance)
            imageArray = imageArray[len(imageArray)-10:]
            #print(imageArray)
            #Set the classifier tag equal to the string in the filename
            classifierTag = filename.split('.')[0]
            pattern = r'[0-9]'
            classifierTag = re.sub(pattern, '', classifierTag)

            if(len(imageArray) == 10):
                imageX.append(imageArray)
                imageY.append(classifierTag)

#imageY = np.array(imageY)
print("-----")
#print(imageX)
#print(imageY)
#print(imageY.shape)
# for entry in imageX:
#     print(len(entry))
# #Classifier for type of shoe, this will be the y value for training data
classifier = ["boots", "flip_flops", "loafers", "sandals", "sneakers", "soccer_shoes"]

tmpList = []
for entry in imageY:
    tmpList.append(classifier.index(entry))
imageY = np.array(tmpList, dtype=np.uint8)
#imageY.reshape(-1)
imageX = np.array(imageX)
imageX = imageX.reshape(-1, 40)
# print(imageY.shape)

trainX, testX, trainY, testY = train_test_split(
    imageX, imageY, test_size = 0.3, shuffle = True
    )

# print(trainX.shape)
# print(trainY.shape)

#Defines our prediction algorithm
classifier = LogisticRegression(max_iter = 10000)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Number of Correct Predicitons: {correct}, Incorrect Predictions: {incorrect}, % Correct %: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
pyplot.show()
