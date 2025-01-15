from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize the Flask app
app = Flask(__name__)
model = load_model('rice_leaf_model.h5', compile=False)

# Save in SavedModel format  
model.save('saved_model/')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Main page route
@app.route('/main')
def main():
    return render_template('main.html')

# Prediction page route (GET request)
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Prediction page route (POST request for processing uploaded image)
@app.route('/predict', methods=['POST'])
def predictDisease():
    # Retrieve the uploaded image file
    imagefile = request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)
    
    # Load the trained model with compile=False to bypass optimizer-related issues
    loaded_model = load_model('rice_leaf_model.h5', compile=False)
    
    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)  # Convert image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the class with the highest probability
    
    # Class names (corresponding to the model's output)
    class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name
    
    print("Predicted class:", predicted_class_name)
    
    # Return the prediction and display the uploaded image
    return render_template('predict.html', image_path=image_path, prediction=predicted_class_name)

# Analysis page routed
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# Charts page route
@app.route('/charts')
def charts():
    return render_template('charts.html')

# Contact page route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Main entry point
if __name__ == "__main__":
    # Run the Flask development server 
    app.run(debug=True)
    
