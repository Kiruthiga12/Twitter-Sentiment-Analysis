# 😊😢😐Twitter Sentiment Analysis using Logistic Regression

## 📘 Project Overview
This project classifies tweets as **positive** or **negative** based on their text content.  
It uses **Natural Language Processing (NLP)** techniques for text cleaning and **Logistic Regression** as the machine learning model.  
The dataset contains labeled tweets (positive = 4, negative = 0), which were preprocessed, vectorized, trained, and evaluated.

---

## 🔍 Objective
To build a simple and effective sentiment analysis model that can understand the emotional tone of tweets using classical machine learning methods.

---

## 🧠 Key Steps Performed

### 1. Data Loading
- Loaded **`twitter_dataset.csv`** using pandas with encoding `'ISO-8859-1'`.  
- Added column names: `['target', 'id', 'date', 'flag', 'user', 'text']`.

### 2. Data Preprocessing
- Checked dataset size and null values.  
- Replaced target value `4` with `1` to make binary labels (0 = Negative, 1 = Positive).  
- Used **Regular Expressions (`re`)** to clean unwanted symbols and numbers.  
- Converted all tweets to **lowercase**.  
- Removed **stopwords** using NLTK’s English stopword list.  
- Applied **Porter Stemming** to reduce words to their root form (e.g., *loving → love*).  
- Created a new column `stemmed_content` for cleaned text.

### 3. Feature Extraction
- Converted cleaned tweets into numerical form using **TF-IDF Vectorizer**.  
- TF-IDF (Term Frequency – Inverse Document Frequency) assigns higher weights to informative words.

### 4. Model Building
- Used **Logistic Regression** as the classification algorithm.  
- Split data into **80% training** and **20% testing** using `train_test_split` with stratification to maintain class balance.  
- Trained the model with `max_iter=1000` for better convergence.

### 5. Model Evaluation
- Predicted sentiment on both training and testing datasets.  
- Calculated **accuracy score** using `accuracy_score()` from scikit-learn.  
- Compared training and testing accuracy to check model performance.

### 6. Model Saving and Loading
- Saved the trained model using **`pickle`** (`trained_model.sav`).  
- Reloaded the saved model to verify persistence and reuse.  

### 7. Sentiment Prediction
- Tested the model by predicting sentiment for a few tweets from the test set.  
- Displayed output as:  
  - **Positive Tweet** if prediction = 1  
  - **Negative Tweet** if prediction = 0  

---

## 🧩 Tools & Libraries Used
- **Python**
- **pandas** – for data loading and analysis  
- **numpy** – for numerical operations  
- **re (regex)** – for text cleaning  
- **nltk** – for stopword removal and stemming  
- **scikit-learn** – for TF-IDF vectorization, model training, and evaluation  
- **pickle** – for saving and loading the model

---

## ⚙️ Workflow Summary
1. Import required libraries.  
2. Load the dataset.
3. Handle missing values and standardize target labels.  
4. Clean and preprocess the text data.  
5. Convert text into TF-IDF numerical vectors.  
6. Split data into train/test sets.  
7. Train Logistic Regression model.  
8. Evaluate accuracy on train and test data.  
9. Save and reload the trained model using pickle.  
10. Predict sentiment for individual tweets.

---

## 📊 Results
| Dataset | Accuracy |
|----------|-----------|
| Training Data | *79%* |
| Testing Data | *77%* |

---
