import numpy as np
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D,Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, MaxPool2D , Cropping2D
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Lambda

import scipy.misc
import cv2
import os


adam = Adam(0.0001)

L2NormConst = 0.001

model = Sequential()

model.add(Cropping2D(cropping=((80,0), (0, 0)), input_shape=(160,320,3)))
model.add(BatchNormalization())
model.add(Lambda(lambda x: tf.image.resize(x, (66,200)))) 

model.add(Conv2D(24,(5,5),activation='relu',kernel_regularizer=regularizers.l2(0.001),strides = (2,2)))
model.add(BatchNormalization())

model.add(Conv2D(36,(5,5),activation='relu',kernel_regularizer=regularizers.l2(0.001),strides = (2,2)))
model.add(BatchNormalization())

model.add(Conv2D(48,(5,5),activation='relu',kernel_regularizer=regularizers.l2(0.001),strides = (2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())


model.add(Flatten())

model.add(Dense(1164,activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(200,activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(50,activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(10,activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(1,activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.summary()

model.compile(optimizer=Adam(0.001), loss = 'mse', metrics = ['mae'])

model.load_weights("/Users/krishanukashyap/Downloads/weights_final.h5")
img_str = cv2.imread('/Users/krishanukashyap/Downloads/steering1.jpeg',0)
rows,cols = img_str.shape

smoothed_angle = 0

i = 40000
while(cv2.waitKey(1) != ord('q') and i<50000):
    file = os.path.exists("/Users/krishanukashyap/Downloads/driving_dataset/" + str(i) + ".jpg")
    if(file):
        img1 = image.load_img(
            "/Users/krishanukashyap/Downloads/driving_dataset/" + str(i) + ".jpg",color_mode='rgb')
        img1 = image.img_to_array(img1)/255.0
        img = image.load_img("/Users/krishanukashyap/Downloads/driving_dataset/" + str(i) + ".jpg",color_mode='rgb',target_size=[160,320])
        img = image.img_to_array(img)/255.0
        img_resh = np.reshape(img,[1,160,320,3])
        degrees = float(model.predict(img_resh) * 180.0 / scipy.pi)
        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("frame", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        #rotated = imutils.rotate_bound(img_str, degrees)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img_str,M,(cols,rows))
        cv2.imshow("steering wheel", dst)
    i+=1
cv2.destroyAllWindows()