import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model
st.cache()
def get_processor(model_name:str):
    return AutoImageProcessor.from_pretrained(model_name)

st.cache()
def get_model(model_name:str):
    return AutoModelForImageClassification.from_pretrained(model_name)

# Streamlit app
def main():
    st.title("Image Classification with Hugging Face")
    st.write("Upload an image for classification.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Process uploaded image
        input_image = processor(uploaded_file)
        
        # Perform classification
        with st.spinner('Classifying...'):
            outputs = model(input_image.pixel_values)
            predicted_class_idx = outputs.logits.argmax().item()

        # Display results
        st.write("Predicted Class:", predicted_class_idx)

if __name__ == "__main__":
    model_name = "ttangmo24/vit-base-classification-Eye-Diseases"
    processor = get_processor(model_name)
    model = get_model(model_name)
    main()
