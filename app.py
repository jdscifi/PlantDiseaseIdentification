import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model =tf.keras.models.load_model('PlantDNet.h5',compile=False)

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(144, 144))
    show_img = image.load_img(img_path, grayscale=False, target_size=(144, 144))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

@app.route ('/')
def home():
	return render_template(".html", )

 
 if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port = port)
