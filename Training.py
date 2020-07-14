#import lib
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import cv2
from sklearn.utils import shuffle

#Loading Datas
#Swing Images
load_images = []
for img in range(0, 800):
    image = cv2.imread('Dataset/SwingImages/swing_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))

#Palm images
for img in range(0, 800):
    image = cv2.imread('Dataset/PalmImages/palm_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))
    
#Fist images
for img in range(0, 800):
    image = cv2.imread('Dataset/FistImages/fist_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    load_images.append(gray_img.reshape(89, 100, 1))
    

#Define Output vector 

vectors = []
for img in range(0, 800):
    vectors.append([1, 0, 0])

for img in range(0, 800):
    vectors.append([0, 1, 0])

for img in range(0, 800):
    vectors.append([0, 0, 1])

test_images = []

#Preprocessing the data
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
    
# Creating CNN Model
tf.reset_default_graph()

network=input_data(shape=[None,89,100,1],name='input')
# Convolution Layer
# Conv2D wrapper, with bias and relu activation
network=conv_2d(network,32,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,64,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,128,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,256,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,256,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,128,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,64,2,activation='relu')

# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=fully_connected(network,1000,activation='relu')

# Apply Dropout
network=dropout(network,0.75)

# Fully connected layer
network=fully_connected(network,3,activation='softmax')

# With TFLearn estimators
# Define loss and optimizer
network=regression(network,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

# Train using Classifier
model=tflearn.DNN(network,tensorboard_verbose=0)

# Shuffle Training Data
load_images, vectors = shuffle(load_images, vectors, random_state=0)

# Perform the model training
# Training the model
model.fit(load_images, vectors, n_epoch=50, validation_set = (test_images, labels), snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("Trained_Model/GestureRecogModel.tfl")
