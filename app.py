import base64
import numpy as np
import io
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

classes = ["sad", "fearful", "angry", "happy", "surprised", "neutral"]


def get_model():
    global model
    model = load_model("exp_det.h5")
    print("Model Loaded")


print("loading model...")
get_model()


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
