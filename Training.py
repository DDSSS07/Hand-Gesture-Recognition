#import library
import tensorflow as tf         # for building CNN architechture
import tflearn                  
#TFLearn brings "layers" that represent an abstract set of operations to make building neural networks more convenient
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import cv2

#Shuffle arrays or sparse matrices in a consistent way
from sklearn.utils import shuffle

# Loading images from their respective directories and converting them to grayscale
# Training images
load_images = []

# Datasets downloaded from kaggle

#Palm images
for img in range(0, 1000):
    image = cv2.imread('Dataset/PalmImages/palm_' + str(img) + '.png')      #function cv2.imread() to read an image.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      #For BGR -> Gray conversion
    # 1 channel as we converted the image to grayscale
    load_images.append(gray_img.reshape(89, 100, 1))                        #The reshape() function is used to give a new shape to an array without changing its data and
    # to match the convolutional layer, we used a 2D convolution, so we reshaped it into three-dimensional format)
    
#Fist images
for img in range(0, 1000):
    image = cv2.imread('Dataset/FistImages/fist_' + str(img) + '.png')      #function cv2.imread() to read an image.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      #For BGR -> Gray conversion
    # 1 channel as we converted the image to grayscale
    load_images.append(gray_img.reshape(89, 100, 1))                        #The reshape() function is used to give a new shape to an array without changing its data and
    # to match the convolutional layer, we used a 2D convolution, so we reshaped it into three-dimensional format)

# Swing Images
# We are using opencv to read and convert the image to grayscale
for img in range(0, 1000):
    image = cv2.imread('Dataset/SwingImages/swing_' + str(img) + '.png')    #function cv2.imread() to read an image.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      #For BGR -> Gray conversion
    # 1 channel as we converted the image to grayscale
    load_images.append(gray_img.reshape(89, 100, 1))                        #The reshape() function is used to give a new shape to an array without changing its data and
    # to match the convolutional layer, we used a 2D convolution, so we reshaped it into three-dimensional format)

# Define Result vector 
# We have 1000 images for training and 100 for testing. 
# Performing One-hot encoding for training images
vectors = []
for img in range(0, 800):       
    vectors.append([1, 0, 0])

for img in range(0, 800):
    vectors.append([0, 1, 0])

for img in range(0, 800):
    vectors.append([0, 0, 1])

# Testing images
# Loading images from their respective directories and converting them to grayscale
# All the images now become an array of pixels of size(89,100,1)


test_images = []
#Swing images

# We are using opencv to read and convert the image to grayscale
for img in range(0, 100):
    image = cv2.imread('Dataset/SwingTest/swing_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))                    # 1 denotes that we converted the image to grayscale

#Palm images
for img in range(0, 100):
    image = cv2.imread('Dataset/PalmTest/palm_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))                    # 1 denotes that we converted the image to grayscale
    
#Fist images
for img in range(0, 100):
    image = cv2.imread('Dataset/FistTest/fist_' + str(img) + '.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_images.append(gray_img.reshape(89, 100, 1))                    # 1 denotes that we converted the image to grayscale

labels = []
# Performing One-hot encoding for test images
for img in range(0, 100):
    labels.append([1, 0, 0])
    
for img in range(0, 100):
    labels.append([0, 1, 0])

for img in range(0, 100):
    labels.append([0, 0, 1])
    
# Creating CNN Model
tf.reset_default_graph()      # Clears the default graph stack 

# The input_data is a layer that will be used as the input layer to our network. 
network=input_data(shape=[None,89,100,1],name='input')

# Creating Convolution Layer

# Designing a Conv2D wrapper consisting of bias along with relu activation
# relu removes all the negative values from the convolution and all the positive values remain the same but all the negative values get changed to zero
# Max Pooling returns the maximum value from the portion of the image covered by the Kernel and performs as a Noise Suppressant. 
# It discards the noisy activations and also performs dimensionality reduction.

# Building 2D convolutional layer into architecture of CNN.
network=conv_2d(network,32,2,activation='relu')

# Creating Pooling layer
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,64,2,activation='relu')

# repeating additional convolution and pooling layers to decrease the loss
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

# Creating Fully connected layer
network=fully_connected(network,1000,activation='relu')

# Apply Dropout
network=dropout(network,0.75)
network=fully_connected(network,3,activation='softmax')

# With TFLearn estimators
# Define loss and optimizer
network=regression(network,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

# Train the Network using Classifier
model=tflearn.DNN(network,tensorboard_verbose=0)

# Shuffle Training Data
load_images, vectors = shuffle(load_images, vectors, random_state=0)

# Perform the model training
# Training the model
model.fit(load_images, vectors, n_epoch=50, validation_set = (test_images, labels), snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("TrainedModel/GestureRecogModel.tfl")
