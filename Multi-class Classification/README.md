# 🧠 Obesity Level Prediction — Multi-Class Classification Project

## 📍 Overview

This project focuses on predicting the **obesity level** of individuals using various health, lifestyle, and demographic features.  
We approach this as a **multi-class classification** problem using logistic regression with both **One-vs-All (OvA)** and **One-vs-One (OvO)** strategies.

---

## 📂 Dataset

**Filename**: `Obesity_level_prediction_dataset.csv`  
- The dataset contains features like `Age`, `Weight`, `Height`, `FCVC` (food intake), `TUE` (time using technology), and more.
- The target variable is `NObeyesdad`, a multi-class label indicating obesity type (e.g., Normal_Weight, Obesity_Type_I, etc.).

---

## 🛠️ Key Steps

### 1. Data Preprocessing
- Loaded the dataset using `pandas`
- One-hot encoded categorical features
- Scaled continuous features using `StandardScaler`
- Encoded the target class (`NObeyesdad`) using `astype('category').cat.codes`

### 2. Model Training
- Split data into training and test sets using `train_test_split` with stratification
- Trained two models using `LogisticRegression`:
  - `multi_class='ovr'` for **One-vs-All**
  - `multi_class='ovo'` for **One-vs-One**

### 3. Evaluation
- Evaluated both models using:
  - Accuracy score
  - Confusion matrix
  - Feature importance from model coefficients

---

## 📊 Results

| Strategy   | Accuracy (%) |
|------------|--------------|
| One-vs-All | ~76%         |
| One-vs-One | Similar range (may vary slightly) |

Confusion matrices and bar charts of feature importance were used to further interpret model performance.

---

## 🔍 Notebooks

- [`mulit_class_classification.ipynb`](./mulit_class_classification.ipynb)

---

## 💡 Highlights

- Demonstrates clear preprocessing, encoding, and scaling steps
- Compares OvA vs OvO multi-class strategies
- Explores interpretability through coefficient-based feature importance

---

## 🚀 Tools Used

- Python 🐍
- Pandas, NumPy
- Scikit-learn (Logistic Regression, preprocessing, metrics)
- Matplotlib / Seaborn (for visualization)

---

## 📁 How to Run

1. Clone the repo or open the notebook in VS Code or Jupyter
2. Make sure `Obesity_level_prediction_dataset.csv` is in the same directory
3. Run all cells in `mulit_class_classification.ipynb`


