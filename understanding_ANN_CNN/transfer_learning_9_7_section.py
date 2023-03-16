#%%
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras_preprocessing.image import load_img
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
np.random.seed(0)
#%%

class_names = ['apple', 'bottle', 'bodycream', 'yoda']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (224,224)

# %%
def load_data():
    DIRECTORY = r'/Users/CristaVillatoro/Desktop/tahini-tensor-student-code/week9/imageclassifier/data/'
    CATEGORY = ['train', 'test' ]

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)

        if category == 'train':
            # Use data augmentation for the training set
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()

        # Load images and labels using the data generator
        images = []
        labels = []
        for folder in os.listdir(path):
            if folder.startswith('.'):
                continue
            label = class_names_label[folder]
            img_dir = os.path.join(path, folder)
            img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if not f.startswith('.')]
            for img_file in img_files:
                if not img_file.endswith('.jpg') and not img_file.endswith('.png'):
                    continue
                img = load_img(img_file, target_size=IMAGE_SIZE)
                x = np.array(img)
                x = x.reshape((1,) + x.shape)
                x = datagen.flow(x, batch_size=1)[0]
                images.append(x)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

# %%
(Xtrain , ytrain),(Xtest, ytest) = load_data()
# %%
Xtrain = np.squeeze(Xtrain, axis=1)
Xtest = np.squeeze(Xtest, axis=1)

# %%
# One-hot encode y
ytrain_onehot = tf.keras.utils.to_categorical(ytrain, nb_classes)
ytest_onehot = tf.keras.utils.to_categorical(ytest, nb_classes)
#%%
#ANN with a pre-trained model in the base
# Load the pre-trained VGG16 model
# Fine-tune the pre-trained VGG16 model
base_model = VGG16(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')

# Freeze all but the last few layers in the base model
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Add a new output layer
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(nb_classes, activation='softmax')(x)

# Create the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

#%%
print(base_model.output_shape)

#%%
# Adjust the learning rate
opt = tf.keras.optimizers.Adam(lr=0.0001)

# Compile the model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#%%
# Train the model for more epochs
history = model.fit(Xtrain, ytrain_onehot, epochs=20, validation_split=0.2)

# %%
# Inspect the learning curves to see 
# if the model needs to be optimized further

# Save the model
model.save('my_model.h5')
#%%
# Plot the learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %%
