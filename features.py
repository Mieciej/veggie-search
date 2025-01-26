import glob
import numpy as np
import os.path
import sys
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import keras

MODELS = ["MACIEK1.keras", "KUBA1.keras"]

model_features = {}
model_imagenames = {}
models = {}

def load_features():
    cache = set()

    def add_features(model_name, imagename, features):
        model_features.setdefault(model_name, []).append(features)
        model_imagenames.setdefault(model_name, []).append(imagename)
        cache.add((model_name, imagename))

    if os.path.isfile("cache.ini"):
        with open("cache.ini", "r") as file:
            curr_model = None
            n = 0
            for line in file.readlines():
                if line[0] == "[" and line[len(line)-2] == "]":
                    curr_model = line.removesuffix("]\n").removeprefix("[")
                else:
                    assert curr_model != None
                    imagename, text_arr = line.split("=")
                    text_arr = text_arr.removeprefix("[").removesuffix("]\n")
                    text_arr = text_arr.split(" ")
                    arr = np.zeros((1024,))
                    for i, value in enumerate(text_arr):
                        arr[i] = float(value)
                    add_features(curr_model, imagename, arr)
                    n += 1
        print(f"Loaded {n} image features from cache")

    for model_name in MODELS:
        model = tf.keras.models.load_model("./models/"+model_name)
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('features').output
        )
        models[model_name] = feature_extractor
        print("Loaded model:", model_name)

    new_featues = False
    i = 0
    print("Looking for new images...")
    for filename in glob.glob("veggie-images/validation/*/*"):
        try:
            img_array = None
            for model_name, model in models.items():
                if (model_name, filename) in cache:
                    continue
                if not img_array:
                    img = image.load_img(filename, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = keras.ops.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array).reshape((1024, ))
                add_features(model_name, filename, prediction)
                new_featues = True
                i+=1
        except KeyboardInterrupt:
            break
    print(f"Read features of {i} new images")

    if new_featues:
        # print whole length of array to file
        with open("cache.ini", "w+") as cache_file:
            for model_name, features in model_features.items():
                cache_file.write(f"[{model_name}]\n")
                image_names = model_imagenames[model_name]
                for i, image_features in enumerate(features):
                    cache_file.write(f"{image_names[i]}=")
                    # because built-in numpy methods are disappointingly bad
                    str_arr="["
                    for n in image_features:
                        str_arr += str(n) + " "
                    cache_file.write(str_arr[:-1]+"]\n")


    for model in model_features:
        model_features[model]=np.array(model_features[model])


def get_image_order(query_filename, model):
    img = image.load_img(query_filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, axis=0)
    q = models[model].predict(img_array).reshape((1024,))
    f = model_features[model]
    fl = np.linalg.norm(f, axis=1)
    ql = np.linalg.norm(q)
    similarity = np.sum(q * f, axis=1) / (fl * ql)
    print(similarity)
    order = sorted(range(len(similarity)), key = lambda x : similarity[x], reverse=True)
    print("Most similar images")
    for i in range(5):
        print(model_imagenames[model][order[i]], similarity[order[i]])
    return order
