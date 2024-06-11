import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import plotly.express as px

# Load model
@st.cache(allow_output_mutation=True)
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

            # Display the scores in a custom colored bar chart
            st.write("Classification Scores:")
            fig = px.bar(df, x='Label', y='Score', title="Classification Scores",
                         color='Score', color_continuous_scale='Viridis')
            st.plotly_chart(fig)

            # Display the top score and corresponding label with increased font size and bold
            top_score_idx = df['Score'].idxmax()
            top_label = df.loc[top_score_idx, 'Label']
            top_score = df.loc[top_score_idx, 'Score']
            st.markdown(f"<h2><b>Top Prediction: {top_label} with a score of {top_score:.2f}</b></h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    model_name = "ttangmo24/vit-base-classification-Eye-Diseases"
    model = get_model(model_name)
    main()
