import pathlib
from urllib import request

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template
from flask import request as flask_request
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

application = Flask(__name__)

@application.route('/')
def my_form():
    return render_template('predict.html')

def get_model():
    global model, data_dir, interpreter, class_names
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print(class_names)

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()  # Needed before execution!

def preprocess_image(url, target_size):
    # Open the link and save the image to res
    res = request.urlopen(url)
    # Read the res object and convert it to an array
    img = np.asarray(bytearray(res.read()), dtype='uint8')
    # Add the color variable
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(url+" Successfully Loaded")
    #Convert to RGB image to resize
    img = Image.fromarray(np.uint8(img)) 
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img, dtype='uint8')
    img = np.expand_dims(img, axis=0)

    return img

def load_classes(filename=r"Birds_Classifier_API\data\class_dict.csv"):
    global list_class_names
    
    dict_class_names = pd.read_csv(filename)
    list_class_names = dict_class_names["class"].to_list()
    
@application.route("/predict", methods=["POST"])
def predict():
    print("predict")
    url = flask_request.get_data(as_text=True)
    processed_image = preprocess_image(url, target_size=(224, 224))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], processed_image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        print(class_names[i] + ": " + str(results[i]/np.sum(results)*100) )

    # Dismiss results with a below 50% score
    if results[top_k[0]]/np.sum(results)*100 < 50:
        winning_class = "Inconclusive"
        confidence = "Inconclusive"
    else:
        winning_class = class_names[top_k[0]]
        confidence = str("{:.2%}".format(results[top_k[0]]/np.sum(results)))
    response = {
        'prediction': {
            'winning_class': winning_class,
            'confidence': confidence
        }
    }
    return jsonify(response)

def init_app():
    get_model()
    # load trained tensorflow model
    print(" * Loading Keras model...")

if __name__ == "__main__":
    init_app()

    # start flask app
    application.run(host='0.0.0.0')    
