from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('../digit_classifier.h5')

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).astype('float32') / 255  # Normalize
    image_array = image_array.reshape(1, 784)  # Flatten for model
    return image_array

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Load and preprocess the image
        image = Image.open(file)
        image_array = preprocess_image(image)
        
        # Model prediction
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions)

        # Convert the image to base64 to display
        img_str = image_to_base64(image)

        return render_template('result.html', prediction=predicted_digit, img_data=img_str)

if __name__ == '__main__':
    app.run(debug=True)
