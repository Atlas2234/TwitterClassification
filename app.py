'''
Building a streamlit application with Model for Prediction
'''

# To run the app use the following command in the terminal of this the app.py directory: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

from transformers import pipeline

st.title("Fine Tuning BERT for Twitter Teets for Mutli Class Sentiment Classification")

# Load the model
classifer = pipeline('text-classification', model = 'bert-based-uncased-sentiment-model') # Create the classifier and load the model

text = st.text_area("Enter your tweet here") # Create a text area for user input

if st.button("Predict"): # Create a button to predict the sentiment
    result = classifer(text) # Predict the sentiment of the tweet
    st.write(result) # Display the result

