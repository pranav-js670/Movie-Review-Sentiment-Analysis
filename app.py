import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('model.pkl','rb'))
scalar = pk.load(open('scalar.pkl','rb'))

st.title('Movie Review Sentiment Analysis')
st.header("This app uses machine learning to classify movie reviews as **positive** or **negative** sentiment!")
review = st.text_area('Enter your review here - ',height=100)

if st.button('Predict'):
    if not review:
        st.warning('Please enter a review first!')
    else: 
        review_scalar = scalar.transform([review]).toarray()
        result = model.predict(review_scalar)
        st.subheader('Prediction Result:')
        prediction_labels = ['Negative', 'Positive']
        prediction_icons = ['👎', '👍']
        prediction_index = result[0]
        prediction_label = prediction_labels[prediction_index]
        prediction_icon = prediction_icons[prediction_index]
        st.markdown(f'**Sentiment:** {prediction_label} {prediction_icon}')
        if prediction_index == 0:
            st.write('The model predicts a negative sentiment for the given review.')
        elif prediction_index == 1:
            st.write('The model predicts a positive sentiment for the given review.')
      


    
