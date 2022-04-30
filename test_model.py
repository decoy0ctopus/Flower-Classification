import base64
import csv
import os
import pathlib
from urllib import request

import cv2
import mtcnn
import numpy as np
import pandas as pd
import PIL
import tensorflow_addons as tfa
from flask import Flask, jsonify, render_template
from flask import request as flask_request
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from io import BytesIO
import requests


import tensorflow as tf
import numpy as np
from PIL import Image

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
print("Interpreter loaded successfully")
interpreter.allocate_tensors()  # Needed before execution!

print("== Input details ==")
print("name:", interpreter.get_input_details()[0]['name'])
print("shape:", interpreter.get_input_details()[0]['shape'])
print("type:", interpreter.get_input_details()[0]['dtype'])

print("\nDUMP INPUT")
print(interpreter.get_input_details()[0])

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image, dtype='uint8')
    image = np.expand_dims(image, axis=0)

    return image

url="https://t4.ftcdn.net/jpg/04/20/61/17/360_F_420611788_QSE62leN5bjS7NJjMXtefZRZCrARBZxS.jpg"
# Open the link and save the image to res
res = request.urlopen(url)

# Read the res object and convert it to an array
img = np.asarray(bytearray(res.read()), dtype='uint8')
# Add the color variable
img = cv2.imdecode(img, cv2.IMREAD_COLOR)
print(url+" Successfully Loaded")
img = Image.fromarray(np.uint8(img))  # RGB image
processed_image = preprocess_image(img, target_size=(224, 224))


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

interpreter.set_tensor(input_details[0]['index'], processed_image)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
results = np.squeeze(output_data)
print(results)

top_k = results.argsort()[-5:][::-1]
for i in top_k:
    print(class_names[i] + ": " + str(results[i]/np.sum(results)*100) )
