import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import cv2
from sklearn.utils import shuffle

#Swing Images
load_images = []
for img in range(0, 1000):
    image = cv2.imread('Dataset/SwingImages/swing_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))

#Palm images
for img in range(0, 1000):
    image = cv2.imread('Dataset/PalmImages/palm_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))
    
#Fist images
for img in range(0, 1000):
    image = cv2.imread('Dataset/FistImages/fist_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))
    

#Define Output vector 

vectors = []
for img in range(0, 1000):
    vectors.append([1, 0, 0])

for img in range(0, 1000):
    vectors.append([0, 1, 0])

for img in range(0, 1000):
    vectors.append([0, 0, 1])

test_images = []

#Swing images
for img in range(0, 100):
    image = cv2.imread('Dataset/SwingTest/swing_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))

#Palm images
for img in range(0, 100):
    image = cv2.imread('Dataset/PalmTest/palm_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))
    
#Fist images
for img in range(0, 100):
    image = cv2.imread('Dataset/FistTest/fist_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))

labels = []

for img in range(0, 100):
    labels.append([1, 0, 0])
    
for img in range(0, 100):
    labels.append([0, 1, 0])

for img in range(0, 100):
    labels.append([0, 0, 1])
    
# Define the CNN Model
tf.reset_default_graph()
conv_net=input_data(shape=[None,89,100,1],name='input')
conv_net=conv_2d(conv_net,32,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)
conv_net=conv_2d(conv_net,64,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=conv_2d(conv_net,128,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=conv_2d(conv_net,256,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=conv_2d(conv_net,256,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=conv_2d(conv_net,128,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=conv_2d(conv_net,64,2,activation='relu')
conv_net=max_pool_2d(conv_net,2)

conv_net=fully_connected(conv_net,1000,activation='relu')
conv_net=dropout(conv_net,0.75)

conv_net=fully_connected(conv_net,3,activation='softmax')

conv_net=regression(conv_net,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(conv_net,tensorboard_verbose=0)

# Shuffle Training Data
load_images, vectors = shuffle(load_images, vectors, random_state=0)

# Perform the model training
model.fit(load_images, vectors, n_epoch=50, validation_set = (test_images, labels), snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("Trained_Model/GestureRecogModel.tfl")
