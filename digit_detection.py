#importing All libraries


import numpy as np
import cv2
import os
from numpy.core.defchararray import mod
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


#library for save
import pickle



# from keras.optimizer_v2 import adam
# from keras.models import Sequential
# from keras.layers import Dense,Dropout,Flatten
# from keras.layers.convolutional import Conv2D,MaxPooling2D




######### Declearing Constant #########
folder_path = 'data'
test_ratio = 0.2
val_ratio = 0.2



#######################################
images = []
class_no = []
data_list = os.listdir(folder_path)     #getting data folder
class_num = len(data_list)              # detecting labels (class number)
print("Number of classes: "+str(class_num))

#load images
print("Images loading.....")
for x in range(0,class_num):
    image_path = os.listdir(folder_path+"/"+str(x))
    for y in image_path:
        img = cv2.imread(folder_path+"/"+str(x)+"/"+y)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        class_no.append(x)
    print(x, end=" ")

print()
print("Images Loaded: ",len(images))



#Checking images
# show = cv2.resize(images[6578],(300,300))
# cv2.putText(show,str(class_no[6578]),(30,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
# cv2.imshow("output", show)
# cv2.waitKey()

#converting images in numpy array
images = np.array(images)
class_no = np.array(class_no)

print(images.shape) #print the shape of input
print(class_no.shape)

#spliting the data.
X_train, X_test, y_train, y_test = train_test_split(images, class_no,test_size=test_ratio)
X_train, X_validatioin, y_train, y_validatioin = train_test_split(X_train, y_train,test_size=val_ratio)

print(X_train.shape)
print(X_test.shape)
print(X_validatioin.shape)


#checking the distribution of data
# print(len(np.where(y_train==1)[0]))

#plotting figure
# num_of_samples = []
# for x in range(0,class_num):
#     num_of_samples.append(len(np.where(y_train==x)[0]))

# # print(num_of_samples)

# plt.figure(figsize=(10,5))
# plt.bar(range(0,class_num),num_of_samples)
# plt.title("Number of images for each class")
# plt.xlabel("Class ID/labels")
# plt.ylabel("Number of images")
# plt.show()



# #preprecessing images
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# # img = preProcessing(X_train[654])
# # img = cv2.resize(img,(300,300))
# # cv2.imshow("Output",img)
# # cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validatioin = np.array(list(map(preProcessing,X_validatioin)))
# # print(X_train.shape)

# #reshap for CNN
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validatioin = X_validatioin.reshape(X_validatioin.shape[0],X_validatioin.shape[1],X_validatioin.shape[2],1)


# #Augmentation data
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

dataGen.fit(X_train) # by doing this dataGen will know something about data


# #OneHotEncoding for labels
y_train = to_categorical(y_train,class_num)
y_test = to_categorical(y_test,class_num)
y_validatioin = to_categorical(y_validatioin,class_num)


# ##creating model
def createModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(class_num,activation='softmax'))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False,label_smoothing=0,reduction="auto",name='categorical_crossentropy'),
              metrics=['accuracy'])
    return model

model = createModel()
# print(model.summary())  #printing the summary

history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=20),epochs=20,validation_data=(X_validatioin,y_validatioin),shuffle=1)

# history = model.fit(X_train,y_train,epochs=1,validation_data=(X_validatioin,y_validatioin))


# # ### plotting results
# #plot loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

# #plot accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')

plt.show()


# # #evaluate score
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score: ',score[0])
print('Test Accuracy: ',score[1])



# # #save the model

# pickle_out = open('model_train.pi','wb')
# pickle.dump(model,pickle_out)
# pickle_out.close()

model.save('trainned.model')

# model.save_weights('train_model.h5')