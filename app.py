import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load trained model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('trained_bert_model.pth', map_location=device))
model.to(device)
model.eval()

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Title
st.title("Health-related Fake News Detection")

# Input box for users 
user_input = st.text_area("Enter the text to be analyzed:", height=200)

# Button 
if st.button("Check if the text is real or fake"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        # Display the result
        if prediction == 1:
            st.success("✅ The text is classified as **REAL**.")
        else:
            st.error("❌ The text is classified as **FAKE**.")
