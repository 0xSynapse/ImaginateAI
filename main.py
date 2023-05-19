import streamlit as st
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel



loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()


def predict(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


st.title("Image Captioning")

# Select image from local file
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preds = predict(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted Caption:", preds)

# Provide image URL
image_url = st.text_input("Enter Image URL")
if st.button("Submit") and image_url:
    try:
        response = requests.get(image_url, stream=True)
        image = Image.open(BytesIO(response.content))
        preds = predict(image)
        st.image(image, caption="Image from URL", use_column_width=True)
        st.write("Predicted Caption:", preds)
    except:
        st.write("Error: Invalid URL or unable to fetch image.")
        
        



