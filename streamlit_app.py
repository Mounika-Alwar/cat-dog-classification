import streamlit as st
import cv2
import numpy as np

# Preprocessing function
def preprocess_image(img, img_size=(64, 64)):
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = img.reshape(-1, img_size[0] * img_size[1] * 3).T
    return img

# Forward propagation functions
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = np.maximum(0, Z)
        activation_cache = Z

    elif activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = 1 / (1 + np.exp(-Z))
        activation_cache = Z

    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

# Function to make predictions on a single image
def predict_image(img, parameters):
    # Perform forward propagation
    AL, _ = L_model_forward(img, parameters)
    # Interpret the result
    prediction = 1 if AL > 0.5 else 0
    return prediction

# Load trained parameters (ensure the parameters file is in the same directory)
def load_parameters(file_name):
    data = np.load(file_name + '.npz')
    parameters = {key: data[key] for key in data}
    print(f"Parameters loaded from {file_name}.npz")
    return parameters

# Streamlit UI
st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = preprocess_image(image)

    # Load trained parameters (ensure the parameters file is in the same directory)
    parameters = load_parameters('trained_parameters')

    # Make prediction
    prediction = predict_image(img, parameters)

    # Display result
    class_name = 'Dog' if prediction == 1 else 'Cat'
    st.write(f"The predicted class for the image is: {class_name}")
