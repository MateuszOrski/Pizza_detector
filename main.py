import time
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predict_pizza(img):
    # Ensure the input image is in the correct format
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((224, 224))  # Resize the image to match model's expected input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape
    preprocessed_img = preprocess_input(img_array)  # Preprocess the input for the model

    # Predict the class of the image
    prediction = model.predict(preprocessed_img)
    class_names = ['not_pizza', 'pizza']
    # Return the class name with the highest predicted probability
    return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

# Data augmentation setup and dataset preparation
start = time.time()
datagen = ImageDataGenerator(
    horizontal_flip=True,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=preprocess_input,
    dtype='float32',
    rotation_range=25
)

df_train = datagen.flow_from_directory(
    directory="C:/Users/mikol/ClionProjects/Pizza_detector2/pizza_not_pizza",
    target_size=(224, 224),
    subset='training',
    class_mode='sparse'
)

df_val = datagen.flow_from_directory(
    directory="C:/Users/mikol/ClionProjects/Pizza_detector2/pizza_not_pizza",
    target_size=(224, 224),
    subset='validation',
    class_mode='sparse'
)

IMG_SHAPE = (224, 224, 3)
base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Define and compile the model
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(min_delta=0.01, patience=1, restore_best_weights=True)
history = model.fit(df_train, epochs=15, validation_data=df_val, callbacks=[early_stopping])
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()

# Set up Gradio interface
end = time.time()
print(f"Elapsed time: {end - start} seconds")
demo = gr.Interface(fn=predict_pizza,inputs=gr.Image(),outputs=gr.Label(num_top_classes=2))

if __name__ == "__main__":
    demo.launch()
