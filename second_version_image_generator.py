#%%
# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam,SGD
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
# Prediction libraries
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

import pandas as pd
 
from keras.models import load_model

img_width, img_height = 224, 224
#%%
train_data_dir = './data/train'
validation_data_dir = './data/test'
nb_train_samples =120
nb_validation_samples = 40
epochs = 10
batch_size = 16
#%%
# Checking format of the image
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#%%

# # Building a model in Keras
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Conv2D is the layer to convolve the image into multiple images 
# Activation is the activation function. 
# MaxPooling2D is used to max pool the value from the given size matrix and same is used for the next 2 layers. 
# Flatten is used to flatten the dimensions of the image obtained after convolving it. 
# Dense is used to make this a fully connected model and is the hidden layer. 
# Dropout is used to avoid overfitting on the dataset. 
# Dense is the output layer contains only one neuron which decide to which category image belongs.
#%%
# Compile

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#model.compile(optimizer=Adam(learning_rate=0.1),loss='binary_crossentropy',metrics='accuracy')
# %%
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# ImageDataGenerator that rescales the image, applies shear in some range, zooms the image and does horizontal flipping with the image. 
# This ImageDataGenerator includes all possible orientation of the image. 
# train_datagen.flow_from_directory is the function that is used to prepare data from the train_dataset directory Target_size specifies the target size of the image. 
# test_datagen.flow_from_directory is used to prepare test data for the model and all is similar as above. 
# fit_generator is used to fit the data into the model made above, other factors used are steps_per_epochs tells us about the number of times the model will execute for the training data. 
# epochs tells us the number of times model will be trained in forward and backward pass. 
# validation_data is used to feed the validation/test data into the model. 
# validation_steps denotes the number of validation/test samples.
# %%

model.save_weights('model_saved.h5')

# %%
# Prediction
 
#model = load_model('model_saved.h5')

image = load_img('./data/test/bottle/bottle1.png', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - Apples , 1- Bottle): ", label[0][0])
# %%

# Train a Baseline Model

# Flatten images
X_train = []
y_train = []
for class_name in ['apple', 'bottle']:
    for i in range(nb_train_samples//2):
        img_path = f'./data/train/{class_name}/{class_name}{i+1}.png'
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array.reshape(-1)
        X_train.append(img_array)
        y_train.append(0 if class_name == 'apple' else 1)

X_test = []
y_test = []
for class_name in ['apple', 'bottle']:
    for i in range(nb_validation_samples//2):
        img_path = f'./data/test/{class_name}/{class_name}{i+1}.png'
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = img_array.reshape(-1)
        X_test.append(img_array)
        y_test.append(0 if class_name == 'apple' else 1)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Split data into training and testing sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# %%
print(X_train)
#%%

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Calculate training and test score
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print('Training score:', train_score)
print('Test score:', test_score)


# %%

# Plot a histogram

# Get the initial weights before training
initial_weights = model.get_weights()


# Get the final weights after training
final_weights = model.get_weights()

# Plot the histograms of the weights
fig, axs = plt.subplots(2, len(initial_weights), figsize=(20, 8))

for i, (initial_w, final_w) in enumerate(zip(initial_weights, final_weights)):
    axs[0, i].hist(initial_w.flatten(), bins=50)
    axs[1, i].hist(final_w.flatten(), bins=50)
    axs[0, i].set_title(f'Layer {i+1} Initial Weights')
    axs[1, i].set_title(f'Layer {i+1} Final Weights')

plt.show()

# %%


plot_model(model,
   to_file='model.png',
   show_shapes=False,
   show_layer_names=True,
   rankdir='TB', expand_nested=False, dpi=96
)
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_accuracy']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
logs = pd.DataFrame(history.history)
# %%
history.history
# %%
logs
# %%
logs.plot()
# %%
logs[['loss','val_loss']].plot()
# %%
