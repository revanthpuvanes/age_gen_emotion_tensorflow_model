import numpy as np
import cv2
from keras.preprocessing import image
import time
import math
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

pTime = 0
#-----------------------------
#face expression recognizer initialization
input_shape = (64, 64, 1)
num_classes = 7
l2_regularization=0.01
# model parameters
regularization = l2(l2_regularization)
def mini_XCEPTION():
    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    
    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    
    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)
    model = Model(img_input, output)
    return model
# def emotionModel():
# 	model = mini_XCEPTION()
	
# 	model.load_weights("./models/emotion_model_weights.h5")
	
# 	return model

model = mini_XCEPTION()
model.load_weights('./models/_mini_XCEPTION.86-0.65.hdf5') #load weights

#-----------------------------

emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')

while(True):
	ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 7)

	#print(faces) #locations of detected faces

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(24,242,17),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (64, 64))
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion = emotions[max_index]

		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 0, 255), 2)

		info_box_color = (24,242,17)
		#triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
		triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
		cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
		cv2.rectangle(img,(x+int(w/2)-80,y-20),(x+int(w/2)+80,y-90),info_box_color,cv2.FILLED)
		
		#write emotion text above rectangle
		cv2.putText(img, emotion, (x +int(w/2)-60, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		# +int(w/2)
		#process on detected face end
		#-------------------------

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()