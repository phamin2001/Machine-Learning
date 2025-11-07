# 🚢 Titanic Survival Prediction with ML Pipelines

A comprehensive machine learning project demonstrating end-to-end pipeline construction, hyperparameter tuning, and model comparison for classification tasks.

## 📊 Project Overview

This project builds a classification system to predict Titanic passenger survival using scikit-learn pipelines. The implementation showcases best practices for:
- Automated preprocessing of mixed data types
- Stratified cross-validation for imbalanced datasets
- Hyperparameter optimization with GridSearchCV
- Model comparison and feature interpretation

## 🎯 Key Features

### 🔧 **Automated Preprocessing**
- **Numerical features**: Median imputation + standard scaling
- **Categorical features**: Mode imputation + one-hot encoding
- **Column transformer**: Applies appropriate preprocessing to each column type

### 🔍 **Model Training & Tuning**
- **Random Forest Classifier**: Ensemble method with tuned tree parameters
- **Logistic Regression**: Linear model with L1/L2 regularization
- **GridSearchCV**: Exhaustive hyperparameter search with 5-fold stratified CV

### 📈 **Evaluation & Interpretation**
- Classification reports with precision, recall, and F1-scores
- Confusion matrices visualized as heatmaps
- Feature importance analysis (Gini for RF, coefficients for LR)
- Test set accuracy comparison between models

## 📂 Dataset

**Source:** Seaborn's built-in Titanic dataset  
**Size:** 891 passengers  
**Target:** Binary survival outcome (0 = Did not survive, 1 = Survived)  
**Class distribution:** ~38% survived, ~62% did not survive

### Selected Features

| Feature | Type | Description |
|:--------|:-----|:------------|
| `pclass` | Numerical | Ticket class (1st, 2nd, 3rd) |
| `sex` | Categorical | Gender |
| `age` | Numerical | Age in years |
| `sibsp` | Numerical | # of siblings/spouses aboard |
| `parch` | Numerical | # of parents/children aboard |
| `fare` | Numerical | Passenger fare |
| `class` | Categorical | Ticket class (object format) |
| `who` | Categorical | Man, woman, or child |
| `adult_male` | Boolean | True/False |
| `alone` | Boolean | Traveling alone? |

## 🛠️ Technologies Used

- **Python 3.11+**
- **Core Libraries:**
  - `pandas` — Data manipulation
  - `numpy` — Numerical operations
  - `matplotlib` & `seaborn` — Visualization
  - `scikit-learn` — ML algorithms and pipelines

## ⚙️ Installation & Setup

1. **Clone or download this folder**

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open `titanic_survival_pipeline.ipynb`**

## 📊 Methodology

### 1. **Data Preparation**
- Load Titanic dataset from seaborn
- Select relevant features and target
- Check class balance (38% survival rate)
- Split into 80/20 train/test with stratification

### 2. **Pipeline Construction**
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([SimpleImputer, StandardScaler]), numerical_cols),
        ('cat', Pipeline([SimpleImputer, OneHotEncoder]), categorical_cols)
    ])),
    ('classifier', RandomForestClassifier())
])
```

### 3. **Hyperparameter Tuning**
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`
- **Logistic Regression:** `solver`, `penalty`, `C`, `class_weight`
- **Cross-validation:** 5-fold stratified to preserve class distribution

### 4. **Model Comparison**
- Evaluate both models on held-out test set
- Compare accuracy, precision, recall, F1-score
- Analyze feature importances vs coefficients

## 📈 Results Summary

| Model | CV Accuracy | Test Accuracy |
|:------|:------------|:--------------|
| Random Forest | ~82% | ~80% |
| Logistic Regression | ~81% | ~79% |

**Key Insights:**
- Both models achieve similar overall accuracy
- Random Forest shows better recall for minority class (survivors)
- Feature importance rankings differ between models, suggesting multiple perspectives on predictive patterns
- `sex_male`, `pclass`, and `fare` emerge as strong predictors across both approaches

## 🔍 Feature Importance Highlights

### Random Forest (Gini Importance)
1. `sex_male` — Gender has strong predictive power
2. `fare` — Higher fare correlates with survival
3. `age` — Age influences survival likelihood

### Logistic Regression (Coefficients)
1. `sex_male` (negative) — Being male decreases survival odds
2. `pclass` (negative) — Higher class number (lower class) decreases odds
3. `class_First` (positive) — First-class ticket increases survival odds

## 🧠 Key Learnings

### Best Practices Demonstrated:
1. **Pipeline encapsulation**: Prevents data leakage by applying transformations consistently across CV folds
2. **Stratified splitting**: Maintains class proportions in imbalanced datasets
3. **Separate preprocessing**: Different strategies for numerical vs categorical features
4. **GridSearchCV automation**: Systematically explores hyperparameter space
5. **Model comparison**: Multiple algorithms provide complementary insights

### Considerations:
- **Feature correlation**: Variables like `sex_male`, `who_man`, and `adult_male` likely share information
- **Missing data**: `age` has ~20% missing values; imputation strategy matters
- **Interpretability trade-off**: Random Forest accuracy vs Logistic Regression interpretability

## 🚀 Next Steps

### Potential Enhancements:
1. **Feature engineering**: Create `family_size` = `sibsp` + `parch` + 1
2. **Advanced imputation**: Use KNN or iterative imputation for age
3. **Ensemble methods**: Try Gradient Boosting or XGBoost
4. **Imbalance handling**: Experiment with SMOTE or class weights
5. **Feature selection**: Use recursive feature elimination (RFE)
6. **Correlation analysis**: Investigate multicollinearity among predictors

## 📄 Notebook Structure

1. **Data loading & exploration** — Understand dataset characteristics
2. **Feature selection & target definition** — Choose relevant predictors
3. **Class balance check** — Identify imbalance and plan mitigation
4. **Preprocessing setup** — Define numerical and categorical transformers
5. **Pipeline construction** — Combine preprocessing with classifiers
6. **Random Forest tuning** — GridSearchCV with tree hyperparameters
7. **Model evaluation** — Classification report and confusion matrix
8. **Feature importance analysis** — Interpret Random Forest decisions
9. **Logistic Regression comparison** — Alternative model perspective
10. **Coefficient analysis** — Linear model interpretability
11. **Final comparison** — Side-by-side performance metrics

## 🤝 Use Cases

This project serves as:
- **Educational template** for building ML pipelines
- **Portfolio demonstration** of classification skills
- **Code reference** for preprocessing mixed data types
- **Comparison framework** for model selection

## 📞 Questions or Feedback?

Feel free to explore the notebook, modify parameters, and experiment with different models. This project is designed to be a learning resource for understanding end-to-end classification workflows.

---

<div align="center">
  <strong>Happy Learning! 🎯📚</strong>
  <br>
  <em>Mastering ML pipelines through hands-on practice.</em>
</div>
