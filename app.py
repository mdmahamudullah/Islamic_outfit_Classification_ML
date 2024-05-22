from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the three ResNet50 models
resnet50_male_female = load_model("D:/Islamic_outfit_Classification_ML/ResNet50MF.h5")
resnet50_male_islamic = load_model("D:/Islamic_outfit_Classification_ML/ResNet50M.h5")
resnet50_female_islamic = load_model("D:/Islamic_outfit_Classification_ML/ResNet50F.h5")

# Folder path containing the test images
test_folder = "static"  # Assuming "static" is the static files directory

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming input size is (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions using ResNet50MF.h5
def predict_male_female(img_path):
    img_array = preprocess_image(img_path)
    predictions = resnet50_male_female.predict(img_array)
    return "female" if predictions[0][0] > 0.5 else "male"

# Function to make predictions using ResNet50M.h5
def predict_male_islamic(img_path):
    img_array = preprocess_image(img_path)
    predictions = resnet50_male_islamic.predict(img_array)
    return "Islamic" if predictions[0][0] > 0.5 else "Non-Islamic"

# Function to make predictions using ResNet50F.h5
def predict_female_islamic(img_path):
    img_array = preprocess_image(img_path)
    predictions = resnet50_female_islamic.predict(img_array)
    return "Islamic" if predictions[0][0] > 0.5 else "Non-Islamic"

# Create a list to store predicted rows
predicted_rows = {'male_islamic': [], 'male_non_islamic': [], 'female_islamic': [], 'female_non_islamic': []}

# Function to update the predicted rows
# Function to update the predicted rows
def update_predicted_rows():
    global predicted_rows

    # Clear the predicted_rows dictionary
    predicted_rows = {'male_islamic': [], 'male_non_islamic': [], 'female_islamic': [], 'female_non_islamic': []}

    # Iterate through test images and populate the dictionary
    for filename in os.listdir(test_folder):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(test_folder, filename)

            # Step 1: Male/Female Classification
            gender = predict_male_female(img_path)

            # Step 2: If Male, check for Islamic
            if gender == "male":
                islamic_result = predict_male_islamic(img_path)
                if islamic_result == "Islamic":
                    predicted_rows['male_islamic'].append([filename])
                else:
                    predicted_rows['male_non_islamic'].append([filename])
            # If Female, check for Islamic
            else:
                islamic_result = predict_female_islamic(img_path)
                if islamic_result == "Islamic":
                    predicted_rows['female_islamic'].append([filename])
                else:
                    predicted_rows['female_non_islamic'].append([filename])

# Route for the home page
@app.route('/')
def home():
    update_predicted_rows()
    return render_template('home.html', rows=predicted_rows)

# Route for the add product page
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        # Handle the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file to the test folder
            file_path = os.path.join(test_folder, uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Update predicted rows
            update_predicted_rows()

            return redirect(url_for('home'))
    return render_template('add_product.html')

if __name__ == '__main__':
    app.run(debug=True)
