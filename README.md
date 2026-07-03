# Product Review Sentiment Analysis System 🛒📊

## 📌 Project Overview
This project presents an advanced **Product Review Sentiment Analysis System** tailored for the Malaysian Endurance Sports Nutrition Market. By analyzing customer reviews from Shopee Malaysia, the system assists retailers, distributors, and product managers in evaluating product assortment expansion opportunities. 

The system leverages a fine-tuned **RoBERTa deep learning model** to accurately classify customer sentiment (Positive, Neutral, Negative) from highly unstructured, multilingual e-commerce reviews containing local Malaysian slang and emojis. 

---

## 🚀 Direct Access & Live Demo

The application is deployed and fully operational online. **No local installation or environment setup is required** to test or use the system.

* **🔗 Live Web Application Link:** `[https://hz2thxteydeat4aegaia9o.streamlit.app/]`
* **📄 Sample Test Data:** A sample dataset `[test_reviews_20.csv]` is provided in this repository for quick system verification. Download this file to immediately test the live application's functionality.

---

## ✨ Key Features
* **Robust Data Preprocessing:** Automatically cleans text, normalizes Malaysian internet slang, extracts emoji sentiment scores, and translates multilingual reviews (Malay/Chinese) into English using the Google Translator API.
* **Advanced Transformer Model:** Utilizes a fine-tuned RoBERTa model that achieved **91.00% accuracy**, significantly outperforming traditional baseline models like Logistic Regression (51%) and Multinomial Naive Bayes (54%).
* **Interactive Web Dashboard:** A user-friendly Streamlit interface that requires no coding experience to operate.
* **Automated Insights & Reporting:** Generates visual sentiment distributions (pie charts) and allows users to export results as downloadable Excel or PDF reports.

---

## 💻 How to Test the System
1. Open the **Live Web Application Link** in any modern browser.
2. Download the provided sample review Excel file from this repository.
3. Drag and drop the sample Excel file into the dashboard's uploader component.
4. Select the column containing the review text (e.g., `Review_Text`).
5. Click **"Start AI Analysis"** to see real-time preprocessing, language translation, and sentiment prediction.
6. Explore the interactive visual charts and download the final comprehensive **Excel** or **PDF** report.

---

## 📂 Repository Structure

```text
├── app.py                             # Main Streamlit web application script
├── TNL_Preprocessing.ipynb            # Jupyter Notebook for data cleaning and translation pipeline
├── Model_Training.ipynb               # Baseline traditional ML models training (Logistic Regression, Naive Bayes)
├── Transformer_Final_Training.ipynb   # RoBERTa deep learning model fine-tuning and evaluation
├── Model_Sentiment_ONLY/              # Directory containing the production-ready RoBERTa model and tokenizer
├── Model_Sentiment_Aspect/            # Experimental Multi-Task Learning (MTL) architecture for aspect-based analysis
├── requirements.txt                   # Python dependencies (for local running reference)
└── .gitignore                         # Git ignore file
```
---

## 📊 Model Performance
The system was evaluated against traditional machine learning benchmarks using a stratified testing dataset:
|Model Architecture                 |Test Accuracy     |Macro F1-Score      |
|-----------------------------------|------------------|--------------------|
|Logistic Regression (C=1)          |51.00%            |0.515               |
|Multinomial Naive Bayes (α=2.0)    |54.00%            |0.531               |
|**RoBERTa (Fine-Tuned)**           |91.00%            |0.910               |

(Note: The experimental Multi-Task Learning model for Aspect-Based analysis is available in the Model_Sentiment_Aspect folder but was excluded from the final deployment due to dataset class imbalance and task interference).

---

## 👥 Authors
* Lew Zixin
* Kuek Zheng Kang
* Lo Jin Kai
* Vincent Tay Yong Jun

Multimedia University (MMU), Faculty of Information Science and Technology.