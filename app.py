from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

print("✅ Starting Flask app...")

app = Flask(__name__)

print("✅ Loading model...")
model = tf.keras.models.load_model("plant_disease_model_clean.keras")

print("✅ Model loaded successfully!")

print("✅ Loading labels...")
labels = open("labels.txt").read().splitlines()
print(f"✅ Labels loaded: {len(labels)} classes found.")

@app.route('/')
def home():
    print("✅ Home route accessed.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("✅ Predict route triggered.")
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    print(f"✅ File saved at {file_path}")

    # preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    print("✅ Image preprocessed successfully.")

    # make prediction
    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    result = labels[pred_idx]
    confidence = float(np.max(preds))
    print(f"✅ Prediction complete: {result} ({confidence*100:.2f}%)")

    return render_template('index.html',
                           prediction=result,
                           confidence=round(confidence * 100, 2),
                           image_path=file_path)

if __name__ == '__main__':
    print("✅ Starting server...")

    # Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

    app.run(debug=True, port=8080)
