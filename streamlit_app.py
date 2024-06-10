import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

# Load model
@st.cache(allow_output_mutation=True)
def get_model(model_name: str):
    return pipeline("image-classification", model=model_name)

# Custom CSS to increase the size of the DataFrame container
def add_custom_css():
    st.markdown(
        """
        <style>
        .dataframe-container {
            max-width: 2000px;
            margin: 0 auto;  /* Center align the container */
        }
        .dataframe-container .stDataFrame {
            font-size: 30px;  /* Increase the font size */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit app
def main():
    st.title("Image Classification for Eye Diseases")
    st.write("Upload an image for classification.")

    # Add custom CSS
    add_custom_css()

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

            # Display the DataFrame in a container with custom CSS
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df.style.set_properties(**{
                'text-align': 'left'
            }))
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    model_name = "ttangmo24/vit-base-classification-Eye-Diseases"
    model = get_model(model_name)
    main()
