import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, RobertaModel
import re
from deep_translator import GoogleTranslator
import time
import io
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURATION & MAPPINGS
# ==========================================
st.set_page_config(page_title="Review Sentiment & Aspect Analyzer", layout="wide")

TOKENIZER_PATH = "Model_Sentiment_Aspect"
MODEL_WEIGHTS_PATH = "Model_Sentiment_Aspect/mtl_roberta_weights.pt" 

SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
ASPECT_LABELS = {0: "quality", 1: "General", 2: "expiry", 3: "packaging", 4: "effectiveness", 5: "taste", 6: "price"}

# ==========================================
# 2. MODEL ARCHITECTURE DEFINITION
# ==========================================
class CustomMTLRoberta(nn.Module):
    def __init__(self, num_sentiments=len(SENTIMENT_LABELS), num_aspects=len(ASPECT_LABELS)):
        super(CustomMTLRoberta, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        
        self.sentiment_classifier = nn.Linear(self.roberta.config.hidden_size, num_sentiments)
        self.aspect_classifier = nn.Linear(self.roberta.config.hidden_size, num_aspects)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        sentiment_logits = self.sentiment_classifier(pooled_output)
        aspect_logits = self.aspect_classifier(pooled_output) 
        
        return sentiment_logits, aspect_logits

# ==========================================
# 3. LOAD MODEL & TOKENIZER (Cached)
# ==========================================
@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomMTLRoberta()
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model, device

with st.spinner('Loading AI Model. Please wait...'):
    tokenizer, model, device = load_assets()

# ==========================================
# 4. PREPROCESSING
# ==========================================
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
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    for slang, correct_word in LOCAL_DICT.items():
        text = re.sub(rf'\b{slang}\b', correct_word, text)
        
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        if len(text) < 2:
            return text
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        return text

# ==========================================
# 5. INFERENCE FUNCTION
# ==========================================
def predict_sentiment_and_aspect(cleaned_text):
    if not cleaned_text or cleaned_text.strip() == "":
        return "Neutral", "General"
        
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        sentiment_logits, aspect_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    aspect_pred = torch.argmax(aspect_logits, dim=1).item() 
    
    return SENTIMENT_LABELS[sentiment_pred], ASPECT_LABELS[aspect_pred]

# ==========================================
# 6. STREAMLIT UI - MAIN WORKFLOW
# ==========================================
st.title("🛍️ Product Review Sentiment & Aspect Analyzer")
st.write("Upload a CSV or Excel file containing product reviews, select the text column, and our AI model will automatically extract Aspects and classify Sentiments.")

uploaded_file = st.file_uploader("📂 Please upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Safely load the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='gbk')
    else:
        df = pd.read_excel(uploaded_file)
        
    st.write("### 🔍 Data Preview:")
    st.dataframe(df.head())
    
    # Let the user select which column contains the text
    text_column = st.selectbox("Select the column containing the review text:", df.columns)
    
    if st.button("🚀 Start AI Analysis", type="primary"):
        st.warning("⏳ Calling translation engine and AI model. Please wait...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        cleaned_texts = []
        sentiment_results = []
        aspect_results = []
        
        # Iterate row by row to process and update the progress bar
        for i, row in df.iterrows():
            original_text = row[text_column]
            status_text.text(f"Processing row {i+1} of {total_rows}...")
            
            # 1. Preprocess & Translate
            cleaned = preprocess_and_translate(original_text)
            cleaned_texts.append(cleaned)
            
            # 2. Predict Sentiment & Aspect
            sentiment, aspect = predict_sentiment_and_aspect(cleaned)
            sentiment_results.append(sentiment)
            aspect_results.append(aspect)
            
            # Update Progress Bar
            progress = int(((i + 1) / total_rows) * 100)
            progress_bar.progress(progress)
            
        # Append predictions to dataframe
        df['Translated_Text'] = cleaned_texts
        df['Predicted_Sentiment'] = sentiment_results
        df['Predicted_Aspect'] = aspect_results
        
        status_text.text("✅ All data processed successfully!")
        st.success("🎉 Analysis Completed!")
        
        # --- 7. VISUALIZATIONS ---
        st.divider()
        st.subheader("📈 Analytics Dashboard")
        
        # Stacked Bar Chart for Overall Insights
        st.markdown("#### Overall Sentiment Distribution across all Aspects")
        aspect_sentiment_counts = df.groupby(['Predicted_Aspect', 'Predicted_Sentiment']).size().reset_index(name='Count')
        fig_bar = px.bar(aspect_sentiment_counts, x="Predicted_Aspect", y="Count", color="Predicted_Sentiment", 
                         title="Sentiment Breakdown by Aspect", barmode='stack',
                         color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # Dynamic Pie Chart by selected Aspect
        st.markdown("#### Filter Sentiment by Specific Aspect")
        available_aspects = df['Predicted_Aspect'].unique()
        selected_aspect = st.selectbox("Select an Aspect to view its Sentiment distribution:", available_aspects)
        
        if selected_aspect:
            filtered_df = df[df['Predicted_Aspect'] == selected_aspect]
            sentiment_counts = filtered_df['Predicted_Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             title=f"Sentiment Distribution for: {selected_aspect}",
                             color='Sentiment',
                             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- 8. RESULTS PREVIEW & EXPORT ---
        st.write("### 📝 Detailed Classification Results")
        st.dataframe(df[[text_column, 'Translated_Text', 'Predicted_Aspect', 'Predicted_Sentiment']])
        
        st.write("### 📥 Download Analysis Report")
        
        # Download as CSV
        st.download_button(
            label="📁 Download Complete Analysis Results (Excel)",
            data=excel_data,
            file_name="Sentiment_Aspect_Analysis_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Print/Save as PDF Button
        st.write("### 🖨️ Download Page Report")
        st.write("Click the button below to save the entire analysis dashboard (including charts) as a PDF.")
        components.html(
            """
            <div style="text-align: left; padding-top: 5px;">
                <button onclick="window.parent.print()" 
                        style="
                            background-color: white; 
                            color: rgb(49, 51, 63); 
                            border: 1px solid rgba(49, 51, 63, 0.2); 
                            border-radius: 8px; 
                            padding: 8px 16px; 
                            font-size: 16px; 
                            cursor: pointer;
                            font-family: 'Source Sans Pro', sans-serif;
                            transition: all 0.2s ease;
                        "
                        onmouseover="this.style.borderColor='#FF4B4B'; this.style.color='#FF4B4B';"
                        onmouseout="this.style.borderColor='rgba(49, 51, 63, 0.2)'; this.style.color='rgb(49, 51, 63)';">
                    📄 Save Page as PDF
                </button>
            </div>
            """,
            height=60
        )