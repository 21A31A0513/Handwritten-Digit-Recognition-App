import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28 pixels (MNIST standard size)
    image = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize the image
    img_array = img_array / 255.0
    # Reshape for the model input (28x28x1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make a prediction
def predict_digit(model, image):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Get the prediction
    prediction = model.predict(img_array)
    # Return the predicted digit
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Streamlit UI
st.title('Handwritten Digit Recognition App')

st.write("Draw a digit in the box below:")

# Create a canvas for drawing (using streamlit-drawing)
canvas_result = st_canvas(
    fill_color="white",  # Background color
    stroke_width=25,     # Stroke width for drawing
    stroke_color="white",  # Stroke color
    width=280,  # Canvas width
    height=280,  # Canvas height
    drawing_mode='freedraw',  # Drawing mode
    key="canvas"
)

if canvas_result.image_data is not None:
    # Get the image data from the canvas
    image = Image.fromarray(canvas_result.image_data.astype('uint8'))
    
    # Predict the digit
    if st.button("Predict"):
        predicted_digit = predict_digit(model, image)
        st.write(f"Predicted digit: {predicted_digit}")
        
