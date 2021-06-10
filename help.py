import cv2
import tensorflow as tf
import numpy as np
import pickle


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# pickle_in = open("model_train.p","rb")
# model = pickle.load(pickle_in)
model = tf.keras.models.load_model("trainned.model")
# model = tf.keras.models.load_weights('model.h5')

print("Model loaded")

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


img_path = './images/9.jpg'

img = cv2.imread(img_path)
workingImg = np.array(img)
workingImg = cv2.resize(workingImg,(32,32))
workingImg = preProcessing(workingImg)
workingImg = workingImg.reshape(1,32,32,1)
class_index = int(model.predict_classes(workingImg))
predictions = model.predict(workingImg)
proVal = np.amax(predictions)
img = cv2.resize(img,(320,320))

if proVal > 0.8:
        cv2.putText(img,str(class_index)+"   "+str(round(proVal,2))+"%",(35,35),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)

cv2.imshow('out',img)
cv2.waitKey(0)

# while True:
#     success,orgImg = cap.read()
#     h,w,c = orgImg.shape
#     cpimg = orgImg[0:h//2,0:w//2]
#     cv2.imshow('cut',cpimg)
#     img = np.array(cpimg)
#     img = cv2.resize(img,(32,32))
#     img = preProcessing(img)
#     img = img.reshape(1,32,32,1)
#     # print(img.shape)

#     class_index = int(model.predict_classes(img))
#     # print(class_index)

#     predictions = model.predict(img)
#     # print(predictions)

#     proVal = np.amax(predictions)
#     # print(class_index,proVal)

#     if proVal > 0.8:
#         cv2.putText(orgImg,str(class_index)+"   "+str(proVal)+"%",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)

#     cv2.imshow("Output",orgImg)
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break