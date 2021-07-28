import numpy as np
import cv2
from keras.preprocessing import image
import time
import math

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#del(camera)
pTime = 0
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("./models/facial_expression_model_structure.json", "r").read())
model.load_weights('./models/facial_expression_model_weights.h5') #load weights

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
		# for i in range(10):
		# 	cv2.imwrite('opencv'+str(i)+'.png', detected_face)
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
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