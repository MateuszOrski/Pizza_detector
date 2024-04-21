import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
from PIL import Image
import time
import gradio as gr

start = time.time()

# Load the dataset and create data iterators
data = tf.keras.utils.image_dataset_from_directory('pizza_not_pizza')
data = data.map(lambda x, y: (x / 255.0, y))  # Normalize images

# Split the dataset into train, validation, and test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Define the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train, epochs=20, validation_data=val)

def classify_image(img):
    # Convert the input image to PIL format, then resize it
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((256, 256))  # Resize to match the model's expected input dimensions

    # Convert back to numpy array and normalize
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)
    print(prediction[0])
    return {"Pizza": float(prediction[0]), "Not Pizza": float(1 - prediction[0])}

# Setup the Gradio interface
end = time.time()
print('Elapsed time:', end - start)
iface = gr.Interface(fn=classify_image, inputs="image", outputs="label")
iface.launch()
