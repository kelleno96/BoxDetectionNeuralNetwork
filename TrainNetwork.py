import tensorflow as tf
import random
from tensorflow.keras import datasets, layers, models
import cv2
import ast
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from RenderCubes import CubeGenerator, maxCartLength


imgWidth = 480.0
imgHeight = 270.0

def GetSingleSample():
    for data in CubeGenerator(1):
        input = data[0]
        target = data[1]
        return np.array(input[0]*255, dtype=np.uint8), input, target

def CompareAnglesWithImage(angle1, angle2):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    origin = (250,250)
    pt1 = (int(origin[0] + 100*np.cos(angle1)), int(origin[1] + 100*np.sin(angle1)))
    pt2 = (int(origin[0] + 100 * np.cos(angle2)), int(origin[1] + 100 * np.sin(angle2)))
    img = cv2.line(img, origin, pt1, color=(0,255,0), thickness=3, lineType=cv2.LINE_AA)
    img = cv2.line(img, origin, pt2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    return img


dropout = False

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(imgHeight, imgWidth, 3)))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
if dropout:
    model.add(layers.Dropout(0.25))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# if dropout:
#     model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='linear'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.mse,
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=False, mode='auto', period=1)

# x_train, y_train = GetTrainingData("BoxLabels.txt")
# print("Splitting data.")
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

fit = True
hotstart = True
predict = True
if fit:
    validationData = None
    for data in CubeGenerator(100):
        validationData = data
        break
    if hotstart:
        model = tf.keras.models.load_model("best_model.hdf5")
    # history = model.fit(x_train, y_train,  batch_size=3, epochs=2000, validation_data=(x_test, y_test),
    #                     verbose=1,
    #                     callbacks=[checkpoint])
    history = model.fit(CubeGenerator(batchSize=32), epochs=1000, steps_per_epoch=100, verbose=1, validation_data = validationData, callbacks=[checkpoint])
if predict:
    model = tf.keras.models.load_model("best_model.hdf5")
    while True:
        print("Getting one sample...")
        img, sample, label = GetSingleSample()
        preds = model.predict(sample)
        # print(preds[0][:8]*imgWidth)
        # print(label[0][:8] * imgWidth)
        predList = preds[0][:8] * imgWidth
        predList = [int(p) for p in predList]
        print(predList)
        for i in range(0,4):
            x = predList[2*i]
            y = predList[2*i+1]
            img = cv2.circle(img, (x,y), 2, (0,255,0), 1, lineType=cv2.LINE_AA)
        print("Predicted rot, actual rot: {}, {}".format(preds[0][8]*180/np.pi, label[0][8]*180/np.pi))
        angleGraph = CompareAnglesWithImage(preds[0][8], label[0][8])
        print("Predicted length, actual length: {}, {}".format(preds[0][9]*maxCartLength, label[0][9]*maxCartLength))

        cv2.imshow("Single sample", cv2.resize(img, (0,0), fx=3, fy=3))
        cv2.imshow("Angle Graph", angleGraph)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break