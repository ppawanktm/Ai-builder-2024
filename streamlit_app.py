import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

# Load model
@st.cache_resource
def get_model(model_name: str):
    return pipeline("image-classification", model=model_name)

# Streamlit app
def main():
    st.title("Image Classification for Eye Diseases")
    st.write("Upload an image for classification.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file)
        image = Image.open(uploaded_file)
        with st.spinner('Classifying...'):
            outputs = model(image)

            # Process the output to get a DataFrame
            labels = [output['label'] for output in outputs]
            scores = [output['score'] for output in outputs]
            df = pd.DataFrame({
                'Label': labels,
                'Score': scores
            })

            # Display the DataFrame
            st.write("Predicted Classes:")
            st.dataframe(df)

if __name__ == "__main__":
    model_name = "ttangmo24/vit-base-classification-Eye-Diseases"
    model = get_model(model_name)
    main()
