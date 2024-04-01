# Tensorflow / Keras
import tensorflow as tf # used to access argmax function
from tensorflow import keras # for building Neural Networks
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout # for adding Concolutional and densely-connected NN layers.


# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

# Sklearn
import sklearn # for model evaluation
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding labels

# Visualization
import cv2 # for ingesting images
print('OpenCV: %s' % cv2.__version__) # print version
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt # for showing images
print('matplotlib: %s' % matplotlib.__version__) # print version

# Other utilities
import sys
import os

# Assign main directory to a variable
main_dir=os.path.dirname("/Users/bogdanoancea/OneDrive/Scoala/BazeleInformaticii/MachineLearning/Cursuri/")
print(main_dir)


# Specify the location of images after you have downloaded them
ImgLocation = main_dir + "/data/Caltech-101/101_ObjectCategories/"

# List image categories we are interested in
LABELS = set(["dalmatian", "hedgehog", "llama", "panda"])

# Create two lists to contain image paths and image labels
ImagePaths = []
ListLabels = []
for label in LABELS:
    for image in list(os.listdir(ImgLocation + label)):
        ImagePaths = ImagePaths + [ImgLocation + label + "/" + image]
        ListLabels = ListLabels + [label]

# Load images and resize to be a fixed 128x128 pixels, ignoring original aspect ratio
data = []
for img in ImagePaths:
    image = cv2.imread(img)
    image = cv2.resize(image, (128, 128))
    data.append(image)

# Convert image data to numpy array and standardize values
# (divide by 255 since RGB values ranges from 0 to 255)
data = np.array(data, dtype="float") / 255.0

# Show data shape
print("Shape of whole data: ", data.shape)

# Convert Labels list to numpy array
LabelsArray = np.array(ListLabels)

# Encode labels
enc = OrdinalEncoder()
y = enc.fit_transform(LabelsArray.reshape(-1, 1))

# ---- Create training and testing samples ---
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Print shapes
# Note, model input must have a four-dimensional shape [samples, rows, columns, channels]
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

# Display images of 9 animals in the training set and their true labels
fig, axs = plt.subplots(3, 3, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,3):
    for j in range(0,3):
        axs[i,j].matshow(X_train[n])
        axs[i,j].set(title=enc.inverse_transform(y_train)[n])
        n=n+1
plt.show()

##### Step 1 - Specify the structure of a Neural Network
# --- Define a Model
model = Sequential(name="DCN")  # Model

# --- Input Layer
# Specify input shape [rows, columns, channels]
# Input Layer - need to speicfy the shape of inputs
model.add(Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), name='Input-Layer'))

# --- First Set of Convolution, Max Pooling and Droput Layers (all parameters shown)
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='relu',
                 name='2D-Convolutional-Layer-1')
          )  # Convolutional Layer, relu activation used

model.add(MaxPool2D(pool_size=(2, 2),
                    strides=(2, 2),
                    name='2D-MaxPool-Layer-1')
          )  # Max Pooling Layer,

model.add(Dropout(0.1, name='Dropout-Layer-1'))  # Dropout Layer

# --- Second Set of Convolution, Max Pooling and Droput Layers
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',name='2D-Convolutional-Layer-2'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='2D-MaxPool-Layer-2'))
model.add(Dropout(0.1, name='Dropout-Layer-2'))  # Dropout Layer

# --- Third Set of Convolution, Max Pooling and Droput Layers
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                 name='2D-Convolutional-Layer-3'))  # Convolutional Layer
# Second Max Pooling Layer,
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='2D-MaxPool-Layer-3'))
model.add(Dropout(0.1, name='Dropout-Layer-3'))  # Dropout Layer

# --- Feed-Forward Densely Connected Layer and Output Layer
#  flattening is required to convert from 2D to 1D shape
# Flatten the shape so we can feed it into a regular densely connected layer
model.add(Flatten(name='Flatten-Layer'))
model.add(Dense(16, activation='relu', name='Hidden-Layer-1',
                kernel_initializer='HeNormal'))  # Hidden Layer, relu(x) = max(x, 0)
model.add(Dense(4, activation='softmax', name='Output-Layer'))

##### Step 2 - Compile keras model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              # Loss function to be optimized,
              metrics=['Accuracy'],
             )

##### Step 3 - Fit keras model on the dataset
history = model.fit(X_train,  # input data
                    y_train,  # target data
                    batch_size=1,
                    # Number of samples per gradient update.
                    epochs=20,
                    # default=1,
                    verbose=1,
                    )

##### Step 4 - Use model to make predictions
# We need to pass model output through argmax to convert from probability to label
# We convert output from tensor back to numpy array

# Predict class labels on training data
pred_labels_tr = np.array(tf.math.argmax(model.predict(X_train), axis=1))
# Predict class labels on a test data
pred_labels_te = np.array(tf.math.argmax(model.predict(X_test), axis=1))

##### Step 5 - Model Performance Summary
print('------------------------- Model Summary -------------------------')
model.summary()  # print model summary

print('------------------------- Encoded Names -------------------------')
for i in range(0, len(enc.categories_[0])):
    print(i, ": ", enc.categories_[0][i])


print('------------------ Evaluation on Training Data ------------------')
# Print classification report
print(classification_report(y_train, pred_labels_tr))


print('-------------------- Evaluation on Test Data --------------------')
print(classification_report(y_test, pred_labels_te))


# Read in the image
mydog = cv2.imread(main_dir+"/data/mydog.JPG")

# Display the image
plt.matshow(mydog)
plt.show()

# Resize
mydog = cv2.resize(mydog, (128, 128))

# Standardize (divide by 255 since RGB values ranges from 0 to 255)
mydog = mydog / 255.0

# The current shape of mydog array is [rows, columns, channels].
# Add extra dimension to make it [samples, rows, columns, channels]
# that is required by the model
mydog = mydog[np.newaxis, ...]

# Print shape
print("Shape of the input: ", mydog.shape)


#----- Predict label of mydog image -----
# We need to pass model output through argmax to convert from probability to label
# We convert output from tensor to numpy array
# We do inverse transform to convert from encoded value to categorical label
pred_mydog = enc.inverse_transform(np.array(tf.math.argmax(model.predict(mydog),axis=1)).reshape(-1, 1))
print("DCN model prediction: ", pred_mydog)


#----- Show Probabilities of each prediction -----
pred_probs=model.predict(mydog)

# Print in a format with label and probability next to each other
print("Probabilities for each category:")
for i in range(0,len(enc.categories_[0])):
    print(enc.categories_[0][i], " : ", pred_probs[0][i])