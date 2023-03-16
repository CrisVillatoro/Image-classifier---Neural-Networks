#%%
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#%%
# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')
#%%
# Define the predict_frame function
def predict_frame(image):
    # Reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to (224, 224)
    image = cv2.resize(image, (224, 224))

    # Convert the image to an array of shape (1, 224, 224, 3)
    image_array = image.reshape((1, 224, 224, 3))

    # Apply pre-processing
    image_array = preprocess_input(image_array)

    # Make a prediction and decode it
    preds = model.predict(image_array)
    preds_decoded = decode_predictions(preds, top=1)[0]

    # Return the prediction
    return preds_decoded[0][1]

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        continue

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the space key was pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        # Make a prediction on the current frame
        prediction = predict_frame(frame)

        # Print the prediction
        print(prediction)

#%%
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# %%
