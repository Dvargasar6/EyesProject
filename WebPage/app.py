from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os, uuid

app = Flask(__name__)  #Create the app.

# Load the pre-trained model.
model_path = os.path.join(app.root_path, 'models/modelo3.h5')
model = load_model(model_path)

# Folder path to save uploaded images.
STATIC_FOLDER = os.path.join(app.root_path, 'static')
# Create the static folder if it doesn't exist.
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']


def preprocess(img_path):

    # Load image and resize to 128x128
    img = image.load_img(img_path, target_size=(128, 128))

    # Normalize the image (divide by 255.0)
    img_array = image.img_to_array(img) / 255.0

    # Expand dimensions to be compatible with the model.
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.route('/')   #Create the route for the index page.
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  #Create the route for the predict page.

def predict():   # Define the predict function.

    # Verify if a file was uploaded.
    if 'file' not in request.files:
        return 'File not was uploaded.'
    
    # Get the uploaded file.
    file = request.files['file']
    if file.filename == '':
        return 'File not selected for uploading.'

    # Generate a unique filename to avoid conflicts.
    ext = os.path.splitext(file.filename)[1]  # Keep original file extension.
    unique_filename = f"{uuid.uuid4()}{ext}"  # Generate unique filename.
    img_path = os.path.join(STATIC_FOLDER, unique_filename) # Path to save the image.

    # Save the uploaded file to the static folder.
    file.save(img_path)

    # Preprocess the image.
    img_array = preprocess(img_path)

    # Make prediction.
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Render the result template with prediction and image path.
    return render_template('index.html', prediction=predicted_class, image_path=f'/static/{unique_filename}')

#Run the program.
if __name__ == '__main__':         
    app.run(host='0.0.0.0', port=5000, debug=True)