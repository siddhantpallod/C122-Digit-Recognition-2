import cv2
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time
from PIL import Image
import ssl
import PIL.ImageOps
import numpy as np

X,y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classesLen = len(classes)

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 2500, train_size = 7500, random_state = 10)

xTrainScale = xTrain / 255
xTestScale = xTest / 255

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial')

lr.fit(xTrainScale, yTrain)

yPredict = lr.predict(xTestScale)

accuracy = accuracy_score(yTest, yPredict)
print(accuracy)

video = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = video.read()
        gray = cv2.CvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width / 2 - 60), int(height / 2 - 60))
        bottomRight = (int(width / 2 + 60), int(height / 2 + 60))
        cv2.rectange(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]
        imagePil = Image.fromarray(roi)
        imageBw = imagePil.convert('L')
        imageBwResize = imageBw.resize((28, 28), Image.ANTIALIAS)
        imageBwResizeInverted = PIL.ImageOps.invert(imageBwResize)
        pixelFilter = 20
        minPixel = np.percentile(imageBwResizeInverted, pixelFilter)
        imageBwResizeInvertedScaled = np.clip(imageBwResizeInverted - minPixel, 0, 255)
        maxPixel = np.max(imageBwResizeInverted)
        imageArray = np.asarray(imageBwResizeInvertedScaled) / maxPixel
        sample = np.array(imageBwResizeInvertedScaled).reshape(1784)
        testPredict = lr.predict(sample)
        print(testPredict)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('Q')  :
            break
    
    except Exception as e:
        pass

video.release()
cv2.destroyAllWindows()