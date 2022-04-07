import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from flask import Flask, render_template, request,  flash, redirect

tf.get_logger().setLevel('ERROR')
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

json_path = "models/final_model.json"
h5_path = "models/final_model.h5"

IMAGE_INPUT_SIZE = (144, 144)

with open("infection_info.json", "r") as ii:
    infection_info = json.load(ii)


def load_model():
    try:
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})
        loaded_model.load_weights(h5_path)
        loaded_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                             loss='categorical_crossentropy', metrics=['accuracy'])
        return loaded_model
    except Exception:
        return False


model = load_model()


def file_upload(request):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        file = request.files['file']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return os.path.join(app.config['UPLOAD_FOLDER'], filename)


def model_predict(file_name):
    global model
    image_input = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(file_name, grayscale=False, target_size=IMAGE_INPUT_SIZE))
    image_input = np.array(np.expand_dims(image_input, axis=0), 'float32') / 255
    return model.predict(image_input, verbose=0)


@app.route('/')
def home():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    score = np.argmax(model_predict(file_upload(request)), axis=1)
    return render_template("output.html", infection=infection_info[score[0]]["infection"],
                           solution=infection_info[score[0]]["solution"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, host="0.0.0.0")
