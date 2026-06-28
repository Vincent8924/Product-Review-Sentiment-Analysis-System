import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
import plotly.express as px
from deep_translator import GoogleTranslator
import time
import re

# --- Page Configuration ---
st.set_page_config(page_title="Sentiment Analysis System", layout="wide")
st.title("🛍️ Product Review Sentiment Analysis System")
st.write("Upload an Excel file containing product reviews, and our AI model will automatically classify and analyze the sentiments.")

# --- 1. Load Model (Using cache to prevent reloading on every interaction) ---
@st.cache_resource
def load_model():
    # Path to your extracted fine-tuned model directory
    model_path = "./my_fine_tuned_model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

with st.spinner('Loading AI Model. Please wait...'):
    tokenizer, model = load_model()

# --- 2. Text Preprocessing Function ---
LOCAL_DICT = {
    "brg": "barang",
    "sdap": "sedap",
    "tq": "thank you",
    "gud": "good",
    "trbaik": "terbaik",
    "protin": "protein",
    "bkn": "bukan",
    "x": "tak",
    "mcm": "macam"
}

def preprocess_and_translate(text):
    """
    Advanced Data Pre-processing & Translation Pipeline.
    Cleans the text and translates it to English for the model.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Slang & Abbreviation Normalization
    for slang, correct_word in LOCAL_DICT.items():
        text = re.sub(rf'\b{slang}\b', correct_word, text)
        
    # 4. Remove excessive punctuation (e.g., "!!!!" -> "!")
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    # 5. Fix word lengthening (e.g., "goooooood" -> "good")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 6. Clean up whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Translate to English
    try:
        if len(text) < 2:
            return text
        # Auto-detect source language, translate to English
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        # Fallback to original text if translation fails (e.g., network issue)
        return text

# --- 3. Model Prediction Function ---
def predict_sentiment(text):
    # Tokenize the input text for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Disable gradient calculation for faster inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class ID with the highest score
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    # Class mapping (Modify this based on the label order during your model training)
    # Typically 0: Negative, 1: Neutral, 2: Positive
    label_map = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}
    return label_map.get(predicted_class_id, "Unknown")

# --- 4. File Upload and Main Analysis Logic ---
uploaded_file = st.file_uploader("📂 Please upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview:")
    st.dataframe(df.head())
    
    # Allow the user to select the column containing the reviews
    text_column = st.selectbox("Select the column containing the review text:", df.columns)
    
    if st.button("🚀 Start AI Analysis"):
        st.warning("⏳ Calling translation engine and AI model. Please wait...")
        
        # Initialize Progress Bar and Status Text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        cleaned_texts = []
        sentiment_results = []
        
        # Iterate through the DataFrame row by row
        for i, row in df.iterrows():
            original_text = row[text_column]
            
            # Update status text
            status_text.text(f"Processing row {i+1} of {total_rows}...")
            
            # 1. Preprocess & Translate
            cleaned = preprocess_and_translate(original_text)
            cleaned_texts.append(cleaned)
            
            # 2. Predict Sentiment 
            sentiment = predict_sentiment(cleaned)
            sentiment_results.append(sentiment)
            
            # Update progress bar
            progress = int(((i + 1) / total_rows) * 100)
            progress_bar.progress(progress)
            
            # Add a small delay to prevent triggering API rate limits
            time.sleep(0.1) 
            
        # Append the new columns to the DataFrame
        df['Translated_Text'] = cleaned_texts
        df['Sentiment_Result'] = sentiment_results
        
        # Final success message
        status_text.text("✅ All data processed successfully!")
        st.success("🎉 Analysis Completed!")
        
        # --- 5. Results Visualization ---
        st.write("### 📊 Sentiment Analysis Statistics")
        col1, col2 = st.columns(2)
        
        # Calculate sentiment counts
        sentiment_counts = df['Sentiment_Result'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        with col1:
            st.dataframe(sentiment_counts)
            
        with col2:
            # Generate a pie chart using Plotly
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                         title='Sentiment Distribution', 
                         color='Sentiment',
                         color_discrete_map={
                             "Positive 🟢": "green", 
                             "Neutral 🟡": "gold", 
                             "Negative 🔴": "red"
                         })
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📝 Detailed Classification Results")
        # 注意这里已经改成了 Translated_Text，与上面的数据匹配
        st.dataframe(df[[text_column, 'Translated_Text', 'Sentiment_Result']])
        
        # --- 6. Export Functionality (Excel Download) ---
        st.write("### 📥 Download Analysis Report")
        
        # Convert DataFrame to an Excel binary stream
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis_Result')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📁 Download Complete Analysis Results (Excel)",
            data=excel_data,
            file_name="Sentiment_Analysis_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Note regarding PDF printing
        st.info("🖨️ **How to save the report as a PDF?** \nGenerating a PDF directly with interactive web charts can be complex. We recommend using your browser's built-in print feature: **Press `Ctrl + P` (or `Cmd + P`), and select 'Save as PDF' as the destination printer.** This will perfectly save the report along with all charts.")