# 🛒 E-Commerce User Behavior Analysis and Sequence Prediction

This project demonstrates a complete end-to-end data science workflow for large-scale e-commerce clickstream data. It includes memory-efficient data sampling, exploratory data analysis (EDA), user segmentation with RFM analysis and K-Means clustering, and sequential behavior prediction using deep learning models (LSTM and RNN with PyTorch).

---

## 📦 Project Structure

```
project/
│
├── sample_users.py          # Efficient data sampler for large datasets
├── data_cleaning.py         # Data cleaning and formatting
├── eda.ipynb                # Exploratory Data Analysis notebook
├── customer_segmentation.py # RFM segmentation and K-Means clustering
├── sequence_modeling.py     # LSTM/RNN behavior prediction
├── requirements.txt         # List of dependencies
├── README.md                # You are here!
└── UserBehavior_5M.csv      # Sampled 5% user data (generated locally)
```

---

## 📊 Overview

**Goal:** Understand and predict user behavior in an e-commerce context.

### 🔍 Steps:

1. Efficiently sample 5% of users from a massive dataset (>1GB) without memory issues.
2. Perform EDA to uncover user/product trends.
3. Segment users via **RFM (Recency, Frequency, Monetary)** and cluster with **K-Means**.
4. Train sequential models (**LSTM, RNN**) to predict future user actions.

---

## 📂 Dataset

- **Source:** [Alibaba UserBehavior.csv (Tianchi)](https://tianchi.aliyun.com/dataset/649)
- Each row records a user event: `page view`, `add to cart`, `purchase`, `favorite` with a timestamp.
- **Original file size:** ~4GB (not included in repo).

> Create a free account on Alibaba Tianchi to access the dataset.  
> Place `UserBehavior.csv` in your project directory.

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch yellowbrick tqdm
```

### 2. Sample 5% of Users

Run the following script to create a lighter dataset:

```python
# sample_users.py
import pandas as pd
import numpy as np

column_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
chunk_size = 10**6

# Step 1: Get all unique user IDs
user_ids = set()
for chunk in pd.read_csv("UserBehavior.csv", names=column_names, chunksize=chunk_size):
    user_ids.update(chunk["User_ID"].unique())

user_ids = list(user_ids)
np.random.seed(42)
sampled_user_ids = np.random.choice(user_ids, size=int(0.05 * len(user_ids)), replace=False)
sampled_user_ids_set = set(sampled_user_ids)

# Step 2: Filter records for sampled users
sampled_chunks = []
for chunk in pd.read_csv("UserBehavior.csv", names=column_names, chunksize=chunk_size):
    sampled_chunk = chunk[chunk["User_ID"].isin(sampled_user_ids_set)]
    sampled_chunks.append(sampled_chunk)

df_5M = pd.concat(sampled_chunks, ignore_index=True)
df_5M.to_csv("UserBehavior_5M.csv", index=False)
```

⏳ This may take some time but works on machines with limited memory.

### 3. Run Analysis and Modeling

Once `UserBehavior_5M.csv` is created:

- Run **`eda.ipynb`** to visualize insights.
- Use **`customer_segmentation.py`** for RFM and K-Means clustering.
- Run **`sequence_modeling.py`** to train LSTM/RNN models.

---

## 📈 Features

- ✅ Memory-safe chunked reading for large datasets.
- ✅ Modular, reusable components: EDA, segmentation, modeling.
- ✅ Fully-commented, clean Python code and Jupyter notebooks.
- ✅ Predictive modeling using PyTorch with real behavioral sequences.

---

## 📘 References

- [Alibaba UserBehavior Dataset - Tianchi Competition](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)

---

## ✍️ License

This project is for educational/personal use only.  
Please respect the dataset’s license if you republish or reuse results.

---

## ❓ Questions or Issues?

Open an issue or start a discussion in the repository.

---

**Happy Analyzing! 🚀**
