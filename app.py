import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("imageclassifiers1s.h5")
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the web app
st.markdown('<h1 class="centered">Image Classifier (Happy vs Sad)</h1>', unsafe_allow_html=True)

# Add some blank lines for spacing
st.text("\n\n")

# Image uploader widget
uploaded_file = st.sidebar.file_uploader("Choose an image (Happy/Sad)", type=["jpg", "jpeg", "png","bmp"])

# Check if an image has been uploaded and the model is loaded
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction button
    if st.button("Predict"):
        # Predict the class of the uploaded image
        yhat = model.predict(img_array)
        if yhat > 0.5:
            prediction = 'Predicted class is Sad'
        else:
            prediction = 'Predicted class is Happy'
        st.title(f"{prediction}")

