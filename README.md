# 🧠 BERT-based Movie Review Sentiment Classifier

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

> Fine-tuned **DistilBERT** on the IMDB dataset to classify movie reviews as positive or negative. Built a complete NLP pipeline: data preprocessing → fine-tuning → evaluation.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Model | `distilbert-base-uncased` |
| Dataset | IMDB (50,000 reviews) |
| Task | Binary Sentiment Classification |
| Accuracy | ~93% on test set |

---

## 🏗️ Pipeline Overview

```
Raw IMDB Dataset
      ↓
Data Preprocessing  (data_preprocessing.ipynb)
  • Map labels: positive → 1, negative → 0
  • Tokenize with DistilBERT tokenizer
  • Save as cleaned_data.csv
      ↓
Fine-Tuning  (fine_tuning_bert.ipynb)
  • Load pretrained distilbert-base-uncased
  • Fine-tune using HuggingFace Trainer API
  • Train/validation split: 80/20
      ↓
Evaluation  (evalution_with_bert.ipynb)
  • Evaluate on held-out test set
  • Run custom review predictions
```

---

## 🗂️ Project Structure

```
Sentiment_analysis/
├── data_preprocessing.ipynb   # Data cleaning and tokenization
├── fine_tuning_bert.ipynb     # Model fine-tuning with HuggingFace Trainer
├── evalution_with_bert.ipynb  # Evaluation and custom predictions
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Saketh-0/Sentiment_analysis.git
cd Sentiment_analysis
```

### 2. Install dependencies
```bash
pip install torch transformers datasets pandas scikit-learn
```

### 3. Run notebooks in order
```
1. data_preprocessing.ipynb
2. fine_tuning_bert.ipynb
3. evalution_with_bert.ipynb
```

---

## 🧪 Example Prediction

```python
review = "I absolutely loved this movie! The performances were outstanding."
# Output → Positive ✅

review = "Terrible plot, bad acting, complete waste of time."
# Output → Negative ❌
```

---

## 🛠️ Tech Stack

- **Model:** DistilBERT (`distilbert-base-uncased`) via HuggingFace Transformers
- **Framework:** PyTorch + HuggingFace Trainer API
- **Dataset:** IMDB Movie Reviews (50k samples)
- **Libraries:** `transformers`, `datasets`, `torch`, `pandas`, `scikit-learn`

---

## 📚 Key Learnings

- Fine-tuning pretrained transformer models for downstream classification tasks
- Using HuggingFace `Trainer` API for efficient training loops
- Tokenization strategies for BERT-family models
- Evaluating NLP models with accuracy and classification reports

---

*Part of my AI/ML portfolio — [View more projects](https://github.com/Saketh-0)*
