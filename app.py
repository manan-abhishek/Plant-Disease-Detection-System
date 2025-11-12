from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os
from tflite_runtime.interpreter import Interpreter

print("✅ Starting Flask app...")

app = Flask(__name__)

print("✅ Loading lightweight TFLite model...")
interpreter = Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded successfully (lite version)!")

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

    # Ensure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    print(f"✅ File saved at {file_path}")

    # Preprocess image
    img = Image.open(file_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

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

    # Ensure static folder exists before starting
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(host='0.0.0.0', port=10000, debug=False)
