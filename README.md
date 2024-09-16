# Implementation-of-CNN
 
## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries for handling data and building the model.

### STEP 2:
Download and load the MNIST dataset, which includes images and labels.
### STEP 3:
Examine and record the shape of the data to understand its structure.
### STEP 4:
Reshape: Adjust the image dimensions to include a color channel.
Normalize: Scale the pixel values to a range between 0 and 1.
### STEP 5:
Set Up Early Stopping
###
## PROGRAM

### Name: SHARMITHA V
### Register Number: 212223110048
### PROGRAM
~~~
import numpy as np
import tensorflow as tf
# Provide path to get the full path
data_path ='/content/mnist.npz.zip'

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
def reshape_and_normalize(images):
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    images = images.astype('float32') / 255.0

    return images
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name:SHARMITHA V         RegisterNumber: 212223110048         \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        
       
        if logs.get('accuracy') >= 0.995:
            
          
            self.model.stop_training = True

            print("\nReached 99.5% accuracy so cancelling training!")
import tensorflow as tf
def convolutional_model():
    model = tf.keras.models.Sequential([ 
        
      
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        
        
        tf.keras.layers.MaxPooling2D(2, 2),
        
     
        tf.keras.layers.Flatten(),
        
     
        tf.keras.layers.Dense(128, activation='relu'),
        
      
        tf.keras.layers.Dense(10, activation='softmax')
    ]) 

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
          
    return model
model = convolutional_model()
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])

~~~

## OUTPUT

### Reshape and Normalize output

![image](https://github.com/user-attachments/assets/2371e89a-29b6-45d7-9880-bdf4cb405f28)

### Training the model output

![image](https://github.com/user-attachments/assets/7cddde1a-6358-459a-bd7b-28d86100cd52)

## RESULT
Include your result here.
