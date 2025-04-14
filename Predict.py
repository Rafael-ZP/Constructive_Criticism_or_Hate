import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the fine-tuned BERT model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("/Users/rafaelzieganpalg/Projects/SRP_Lab/Main_Proj/deberta_model")
tokenizer = AutoTokenizer.from_pretrained("/Users/rafaelzieganpalg/Projects/SRP_Lab/Main_Proj/deberta_model")

st.title("Constructive Criticism or Hate? An NLP Approach for Movie Reviews")

# Input field for the user to enter a review
input_review = st.text_area("Enter a movie review:", "")

if input_review:
    # Tokenize the input review with truncation and padding
    inputs = tokenizer(input_review, return_tensors="pt", truncation=True, padding=True)
    
    # Set model to evaluation mode and perform prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = int(torch.argmax(logits, dim=1))
    
    # Decode the prediction
    label_map = {0: "Constructive Criticism", 1: "Hate Speech"}
    
    st.subheader("Given Input:")
    st.write(input_review)
    st.subheader("Predicted Class:")
    st.write(label_map[prediction])
