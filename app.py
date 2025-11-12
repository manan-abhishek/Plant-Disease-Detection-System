from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained Keras model
print("✅ Loading model...")
model = tf.keras.models.load_model("plant_disease_model_clean.keras")
print("✅ Model loaded successfully!")

# Load class labels
labels = open("labels.txt").read().splitlines()
print(f"✅ Labels loaded: {len(labels)} classes found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']

    # Save file in static folder
    if not os.path.exists('static'):
        os.makedirs('static')
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    result = labels[pred_idx]
    confidence = float(np.max(preds))

    return render_template('index.html',
                           prediction=result,
                           confidence=round(confidence * 100, 2),
                           image_path=file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
