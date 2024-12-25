
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model.keras')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Model configuration constants
BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 20

# Function to preprocess the input image and make a prediction
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to an array
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)  # Predict using the model

    # Extract the predicted class and confidence score
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Route for the home page and handling image uploads
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']  # Get the uploaded file

        # Validate if a file was selected
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Validate file extension and save if valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Secure the filename
            filepath = os.path.join('static', filename)  # Define the file path
            file.save(filepath)  # Save the file to the server

            # Load and preprocess the uploaded image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Get the model's prediction
            predicted_class, confidence = predict(img)

            # Render the result page with predictions and image details
            return render_template('index.html', image_path=filepath, actual_label=predicted_class, predicted_label=predicted_class, confidence=confidence)

    # Render the default home page for GET requests
    return render_template('index.html', message='Upload an image')

# Function to validate the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
