
from flask import Flask, render_template, request, jsonify
import os
import cv2
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "models/leafguard_model.keras"
CLASS_PATH = "models/class_names.json"

# Load trained model and class names
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("LeafGuard AI model loaded successfully!")

# Pretrained ImageNet model for basic object checking
object_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

def is_plant_image(image):

    resized = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    image_data = np.expand_dims(
        rgb.astype(np.float32), axis=0
    )

    image_data = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_data
    )

    predictions = object_model.predict(image_data, verbose=0)

    labels = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions, top=5
    )[0]
    print("IMAGE CHECKER RESULTS:", labels)

    plant_words = [
        "leaf", "plant", "tree", "flower", "fruit",
        "vegetable", "corn", "maize", "cabbage",
        "broccoli", "potato", "strawberry", "orange",
        "grape", "vine", "daisy", "rose", "pepper",
        "mushroom", "fig", "acorn", "artichoke"
    ]

    for prediction in labels:
        label = prediction[1].lower()

        if any(word in label for word in plant_words):
            return True

    return False

print("Number of trained classes:", len(class_names))


def get_treatment(disease):

    treatments = {
        "healthy": "The leaf appears healthy. Continue proper watering and regular plant care.",
        "Early_blight": "Remove affected leaves. Avoid overhead watering and consult an agricultural expert about suitable treatment.",
        "Late_blight": "Remove infected plant parts. Avoid wetting leaves and seek expert advice.",
        "Black_rot": "Remove infected leaves and plant debris. Improve airflow.",
        "Esca": "Remove affected plant material and consult an agricultural expert.",
        "Leaf_Mold": "Improve ventilation and reduce leaf wetness.",
        "Bacterial_spot": "Remove infected plant debris and avoid overhead watering.",
        "Septoria": "Remove affected leaves and improve airflow.",
        "Spider_mites": "Inspect the undersides of leaves and seek advice about suitable mite control.",
        "Target_Spot": "Remove infected leaves and improve airflow.",
        "Tomato_Yellow_Leaf_Curl_Virus": "Consult an agricultural expert about infected plants and whitefly control.",
        "Tomato_mosaic_virus": "Remove infected plants where appropriate and disinfect gardening tools.",
        "Leaf_blight": "Remove affected leaves and reduce prolonged leaf wetness."
    }

    for key, advice in treatments.items():
        if key.lower() in disease.lower():
            return advice

    return "Consult a local agricultural expert for suitable treatment."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "leafImage" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    file = request.files["leafImage"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    # Image quality check
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 50:
        return jsonify({
            "error": "Image is too blurry. Please upload a clearer image."
        }), 400
    
    # Reject unrelated objects before disease prediction
   

    # Prepare image for the trained model
    resized = cv2.resize(image, (160, 160))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_image = np.array(rgb_image, dtype=np.float32)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict
    predictions = model.predict(input_image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    predicted_class = class_names[predicted_index]

    # Separate plant and disease
    if "___" not in predicted_class:
        return jsonify({
            "error": "Model class information is incorrect."
        }), 500

    plant, disease = predicted_class.split("___", 1)

    plant = plant.replace("_", " ").replace(",", "").strip()
    disease_name = disease.replace("_", " ").strip()

    # Basic uncertainty check
    if confidence < 0.50:
        return jsonify({
            "status": "uncertain",
            "plant": None,
            "disease": None,
            "confidence": round(confidence * 100, 2),
            "advice": "The image could not be identified confidently. Upload a clear Tomato, Potato or Grape leaf."
        })

    is_healthy = disease.lower() == "healthy"

    return jsonify({
        "status": "success",
        "plant": plant,
        "disease": "Healthy" if is_healthy else disease_name,
        "confidence": round(confidence * 100, 2),
        "advice": get_treatment(disease),
        "blur_score": round(float(blur_score), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
