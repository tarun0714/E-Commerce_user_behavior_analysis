# 🛒 E-Commerce User Behavior Analysis and Sequence Prediction

This project demonstrates a complete end-to-end data science workflow for large-scale e-commerce clickstream data. It includes memory-efficient data sampling, exploratory data analysis (EDA), user segmentation with RFM analysis and K-Means clustering, and sequential behavior prediction using deep learning models (LSTM and RNN with PyTorch).

---

## 📦 Project Structure

```
USER_PREFERENCE_ANALYSIS/
│
├── .gitignore                  # Git ignore rules
├── behaviour_prediction.ipynb # Notebook for LSTM/RNN-based sequence modeling
├── best_lstm_model.pth        # Saved LSTM model weights
├── best_rnn_model.pth         # Saved RNN model weights
├── README.md                  # Project documentation (you’re here!)
├── requirement.txt            # Python dependencies
├── UserBehavior.csv           # Original raw dataset (~7GB if full)
└── UserBehavior_5M.csv        # 5% sampled dataset (memory-efficient)
```

---

## 📊 Overview

**Goal:** Understand and predict user behavior in an e-commerce context.

### 🔍 Steps:

1. Efficiently sample 5% of users from a massive dataset without memory issues.
2. Perform EDA to uncover user/product trends.
3. Segment users via **RFM (Recency, Frequency, Monetary)** and cluster with **K-Means** (optional).
4. Train sequential models (**LSTM, RNN**) to predict future user actions using PyTorch.

---

## 📂 Dataset

- **Source:** [Alibaba UserBehavior.csv (Tianchi)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
- Each row records a user event: `page view`, `add to cart`, `purchase`, or `favorite` with a timestamp.
- **Original file size:** ~7GB (not included in repo).

> To run this project, download the original dataset and save it as `UserBehavior.csv` in the project directory. Then run the sampling step to generate `UserBehavior_5M.csv`.

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch yellowbrick tqdm
```

### 2. Sample 5% of Users (Optional Preprocessing)

You can use the following script to reduce memory usage if you're starting with the full `UserBehavior.csv`.

```python
import pandas as pd
import numpy as np

column_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
chunk_size = 10**6

user_ids = set()
for chunk in pd.read_csv("UserBehavior.csv", names=column_names, chunksize=chunk_size):
    user_ids.update(chunk["User_ID"].unique())

user_ids = list(user_ids)
np.random.seed(42)
sampled_user_ids = np.random.choice(user_ids, size=int(0.05 * len(user_ids)), replace=False)
sampled_user_ids_set = set(sampled_user_ids)

sampled_chunks = []
for chunk in pd.read_csv("UserBehavior.csv", names=column_names, chunksize=chunk_size):
    sampled_chunk = chunk[chunk["User_ID"].isin(sampled_user_ids_set)]
    sampled_chunks.append(sampled_chunk)

df_5M = pd.concat(sampled_chunks, ignore_index=True)
df_5M.to_csv("UserBehavior_5M.csv", index=False)
```

### 3. Run Behavior Prediction

Open the notebook:

```bash
behaviour_prediction.ipynb
```

Inside it, you’ll find code for:

- Preprocessing the sampled dataset (`UserBehavior_5M.csv`)
- Preparing sequences
- Training LSTM and RNN models
- Evaluating prediction accuracy

---

## 📈 Features

- ✅ Handles large datasets using chunked reading.
- ✅ Implements sequence modeling via **PyTorch** (LSTM & RNN).
- ✅ Saved model checkpoints: `best_lstm_model.pth`, `best_rnn_model.pth`.
- ✅ Fully-commented and easy-to-understand notebook.
- ✅ EDA and modeling in a single notebook.

---

## 📘 References

- [Alibaba UserBehavior Dataset - Tianchi](https://tianchi.aliyun.com/dataset/649​)

---

## ✍️ License

This project is for **educational and personal use only**.  
Please respect Alibaba’s dataset license terms if redistributing.

---

## ❓ Questions or Feedback?

Open an issue or start a discussion in this repository.

---

**Happy Analyzing! 🚀**
