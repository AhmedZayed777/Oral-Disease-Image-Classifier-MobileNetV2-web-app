import streamlit as st # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image
import numpy as np

# Load model
model = load_model('mobilenetv2_model.h5')  


class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title("Oral Disease Classification App")
st.write("Upload an oral image and the model will predict the disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Predicted Class: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2%}")
    st.subheader("Class Probabilities")
    probs_dict = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
    st.bar_chart(probs_dict)


