from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model
model = load_model("my_model.keras")

# Class labels (CHANGE if your model uses different order)
CLASS_NAMES = ["General", "Compostable", "Recyclable"]

def preprocess_image(image):
    image = image.resize((128, 128))   # match model input size
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            processed = preprocess_image(image)
            preds = model.predict(processed)
            prediction = CLASS_NAMES[np.argmax(preds)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

