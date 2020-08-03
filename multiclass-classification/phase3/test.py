import os
import sys
import json

# Flask


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image







MODEL_PATH = '../phase1/model.h5'
MODEL_JSON = 'model.json'
# model = model_from_json("model.json")
model = load_model(MODEL_PATH, custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()
classes = []

