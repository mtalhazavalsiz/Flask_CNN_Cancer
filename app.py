from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from skimage import transform
import tensorflow

app = Flask(__name__)


model = load_model("sonmodel.h5",compile=False)

def model_predict(img_path, model):

    img = image.load_img(img_path,(224,224))

    i = image.img_to_array(img)/255
    input_arr = np.array([i])

    preds = np.argmax(model.predict(input_arr))
    preds = (model.predict(input_arr) > 0.5).astype("int32")
    if preds == 0:
        print("The CT Image includes Pancreatic Tumor")
    else : 
        print("The CT Image is  Healthy Pancreas")
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/hakkında")
def hakkında():
    return "Pankreas Kanser Tespiti"


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        os.remove(file_path)

        isim1 = 'Cancer'
        isim2 = 'Healthy'
        if preds == 0:
            return isim1
        else:
            return isim2
    return None

if __name__ == '__main__':
    app.run(debug=True)
    
