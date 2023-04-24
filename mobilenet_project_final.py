#%%
# Import needed libraries
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2 

from tensorflow import keras
from keras.applications.mobilenet import MobileNet, decode_predictions, preprocess_input
from keras import preprocessing
from tensorflow.keras.preprocessing import image
import keras.backend as K
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,InputLayer,Dropout
#%%
class_names = ['apple', 'bottle', 'bodycream', 'yoda']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (224,224)
#%%
# Loading the data
def load_data(): 
    DIRECTORY = r'/Users/CristaVillatoro/Desktop/tahini-tensor-student-code/week9/imageclassifier/data/'
    CATEGORY = ['train', 'test' ]
    
    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []

        print('Loaded {}'.format(category))

        for folder in os.listdir(path):
            if folder.startswith('.'):
                continue
            label = class_names_label[folder]

            # Iterate through each image in the folder
            for file in os.listdir(os.path.join(path,folder)):

                # Get the path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)
    
                
                # Open and resize the img
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype= 'int32')

        output.append((images, labels))
    return output

# %%
(Xtrain , ytrain),(Xtest, ytest) = load_data()
# %%
# Checking shapes:
print('Xtrain shape:', Xtrain.shape)
print(Xtrain.shape[0], 'train samples')
print(Xtest.shape[0], 'test samples')
print(Xtrain[0].shape, 'image shape')
print('ytrain shape:', ytrain.shape)

# %%
#To better train the model, the train dataset was shuffle
 
Xtrain , ytrain = shuffle(Xtrain, ytrain , random_state = 25)
# %%
# define the preprocessing function that should be applied to all images
data_gen = preprocessing.image.ImageDataGenerator(   # loads data in batches from disk
    preprocessing_function=preprocess_input,
    # fill_mode='nearest',
    rotation_range=20,                               # rotate image by a random degree between -20 and 20
    # width_shift_range=0.2,                         # shift image horizontally 
    # height_shift_range=0.2,                        # shift image vertically 
    # horizontal_flip=True,                          # randomly flip image horizontally
    zoom_range=0.5,                                  # apply zoom transformation using zoom factor between 0.5 and 1.5
    # shear_range=0.2                                # shear rotates pics, but makes them be in trapezoids (as opposed to squares)
    validation_split=0.2
)
# %%
DIRECTORY_TRAIN = r'/Users/CristaVillatoro/Desktop/tahini-tensor-student-code/week9/imageclassifier/data/train'
#%%
# a generator that returns batches of X and y arrays
train_data_gen = data_gen.flow_from_directory(      # points to dir where data lives
        directory=DIRECTORY_TRAIN,
        class_mode="categorical",
        classes=class_names,
        batch_size=150,
        target_size=(224, 224),
    subset='training'
)

# %%
DIRECTORY_VAL = r'/Users/CristaVillatoro/Desktop/tahini-tensor-student-code/week9/imageclassifier/data/test'
#%%
val_data_gen = data_gen.flow_from_directory(
        directory=DIRECTORY_VAL,
        class_mode="categorical",
        classes=class_names,
        batch_size=150,
        target_size=(224, 224),
    subset='validation'
)
#%%
train_data_gen.class_indices
# %%
class_names
# %%
K.clear_session()
base_model = MobileNet(
    weights='imagenet',
    include_top=False,                          # keep convolutional layers only
    input_shape=(224, 224, 3)
)
# %%
base_model.summary() 
# %%
base_model.trainable = False  # we don't want to train the base model, since this would destroy filters
# %%
len(class_names)
# %%
model = keras.Sequential()
model.add(base_model)
model.add(Flatten())  
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(class_names), activation='softmax')) # TODO; Final layer with a length of 2, and softmax activation
# %%
model.summary() 
# %%
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy, #TODO: why not binary x-entropy?
              metrics=[keras.metrics.categorical_accuracy])

# observe the validation loss and stop when it does not improve after 3 iterations
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.05,     # the minimum expected change in the metric used in order to be seen as an improvement
    patience=3,         # number of epochs with no improvement needed for the model to stop
    restore_best_weights=True,
    mode='min'
    )
# %%
history = model.fit(train_data_gen,
          verbose=2, 
          callbacks=[callback],
          epochs=20,
          validation_data=val_data_gen
          )
# %%
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
# %%

# Prediction
img = image.load_img('/Users/CristaVillatoro/Desktop/tahini-tensor-student-code/week9/imageclassifier/data/test/bottle/bottle1.png',target_size=(224,224))

# %%
plt.imshow(img);

# %%
img.size
# %%
x = np.array(img)
# %%
X = np.array([x]) 
X.shape 
# %%
X_preprocess = preprocess_input(X)
# %%
pred = model.predict(X_preprocess)
pred
# %%
plt.bar(x = class_names, height = pred[0]);
# %%
model.save('model_mobilenet.h5')

# %%
