from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model("DL_Model_DenseNet121.h5")

# Define a function for image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust to match your model's input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/")
def index():
    return render_template("index.html")  # Load HTML page

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read()))  # Open image
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    img = preprocess_image(img)

    try:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get class with highest probability
        return jsonify({"prediction": int(predicted_class)})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
