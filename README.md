# Sentiment_analysis 

## Sentiment analysis with BERT
This project implements a sentiment analysis pipeline using BERT (Bidirectional Encoder Representations from Transformers). The pipeline includes data preprocessing, fine-tuning a pretrained BERT model, and evaluating its performance on movie reviews. The goal is to classify reviews as either positive or negative.

## Project Overview

### 1:- Data Preprocessing:

The raw IMDB dataset is cleaned and transformed into a format suitable for training.
Sentiments are mapped (positive → 1, negative → 0).
The cleaned data is saved as cleaned_data.csv.

### 2:- Model Fine-Tuning:

A pretrained BERT model (distilbert-base-uncased) is fine-tuned for binary sentiment classification.
Hugging Face's Trainer API is used for efficient training and evaluation.

### 3:- Evaluation and Prediction:

The trained model is evaluated on a test dataset.
A prediction function allows users to input custom reviews and get sentiment predictions.

### Skills Used:

- Python
- Hugging Face Transformers Library
- PyTorch
- Pandas
- Datasets Library
- Scikit-learn

### Example Usage
Input:
Enter a movie review: I absolutely loved this movie! It was fantastic.

Output:
positive
