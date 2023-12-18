import hashlib
import io
import os
import time  # Import the time module

import cv2
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.utils import img_to_array, load_img
from PIL import Image


# Load the model and labels
model = load_model('87.h5')
labels = {
    0: "apple",
    1: "avocado",
    2: "banana",
    3: "cucumber",
    4: "dragonfruit",
    5: "durian",
    6: "grape",
    7: "guava",
    8: "kiwi",
    9: "lemon",
    10: "lychee",
    11: "mango",
    12: "orange",
    13: "papaya",
    14: "pear",
    15: "pineapple",
    16: "pomegranate",
    17: "strawberry",
    18: "tomato",
    19: "watermelon",
}


def resize_image(img_path, size=(224, 224)):
    """This function resize the image to square shape and save it to the same path."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = (
        cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
    )

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]
    mask = cv2.resize(mask, size, interpolation)
    cv2.imwrite(img_path, mask)


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    # Load the image
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the image
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    predicted_label = labels[predicted_class]
    return predicted_label, prediction[0][predicted_class] * 100


def run():
    st.sidebar.title("Image Upload")
    img_file = st.sidebar.file_uploader("Choose an Image", type=["jpg", "png", "webp"])
    st.title("Fruit Recognition")
    if img_file is not None:
        current_time = time.strftime("%Y%m%d%H%M%S")  # Add a timestamp
        img_content = img_file.read()  # Store image content
        img_extension = os.path.splitext(img_file.name)[1].lower()

        if img_extension == ".webp":
            img = Image.open(io.BytesIO(img_content)).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_content))

        st.image(img, use_column_width=False, width=500)

        img_hash = hashlib.md5(img_content).hexdigest()
        img_file.seek(0)  # Reset file pointer

        save_image_path = os.path.join('images', f'{current_time}_{img_hash}.png')
        with open(save_image_path, "wb") as f:
            f.write(img_content)

        # Resize the image
        resize_image(save_image_path)

        result, percentage = processed_img(save_image_path)
        print(result)

        # Display the result
        st.success("**Predicted : " + result + '**')

        # Display the accuracy
        st.info('**Accuracy : ' + str(round(percentage, 2)) + '%**')

        # Display the calories
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + ' (in 100 grams)**')


run()
