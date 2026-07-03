# Product Review Sentiment Analysis System 🛒📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![License](https://img.shields.io/badge/License-MIT-green)

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

## 🗄️ Datasets & Processing Pipeline

The system includes a rigorous, multi-stage data processing pipeline designed to handle the complexities of local e-commerce feedback. 

**1. Raw Datasets**
Raw customer reviews were scraped across various endurance sports nutrition products. The `DATASET/` directory contains the initial JSON extractions for brands and products such as CLIF Bar, HIGH5 Energy Gel, Hammer Energy Gel, N8 Refuel, Nature's Key Electrolyte, Nuun Sport Tablets, and SiS products.

**2. The Preprocessing Pipeline**
The `PREPROCESS DATASET/` folder demonstrates the step-by-step transformation of the raw data into a structured format ready for machine learning:
* **Translation:** Raw sentences (`01_raw_sentences`) undergo automated and manual translation bridging (`02_sentence_translated`, `03_need.manual_translation_review`, `04_sentence_translated_completed`) guided by custom configuration files (`translation_glossary.json`) located in the `CONFIG/` folder.
* **Emoji Integration:** Emoji sentiment is calculated and scored (`05_emoji_scored`) using dictionaries stored in the `EMOJI DICTIONARIES/` directory.
* **Text Normalization:** The text is cleaned (`06_sentence_clean`) and standardized using a local slang mapping file (`normalization.json`) from the `CONFIG/` folder.
* **Data Refinement:** Reviews are tagged and filtered to ensure only relevant feedback is processed (`07_relevance_full_tagged`, `08_relevant_filtered`)[cite: 7].
* **Linguistic Structuring:** The pipeline finalizes the text by removing stopwords (`09_stopwords_removed`), extracting key attributes (`10_aspects_extracted`), and applying lemmatization (`11_lemmatized`).

---

## 📂 Repository Structure

```text
├── app.py                             # Main Streamlit web application script
├── requirements.txt                   # Python dependencies (for local running reference)
├── CONFIG/                            # Configuration files including slang normalization and translation glossaries
├── DATASET/                           # Raw collected JSON datasets for various sports nutrition brands
├── EMOJI DICTIONARIES/                # CSV and JSON dictionaries for emoji sentiment scoring
├── PREPROCESS DATASET/                # Step-by-step intermediate datasets from the data preprocessing pipeline
├── Model Training/                    # Jupyter Notebooks for baseline ML and RoBERTa deep learning model training
├── Model_Sentiment_ONLY/              # Directory containing the production-ready RoBERTa model and tokenizer
├── Model_Sentiment_Aspect/            # Experimental Multi-Task Learning (MTL) architecture weights
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