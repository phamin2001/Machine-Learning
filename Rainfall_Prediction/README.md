# 🌧️ Australian Weather - Rainfall Prediction Classifier

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Weather_icon_-_cloud_with_rain.svg/200px-Weather_icon_-_cloud_with_rain.svg.png" alt="Rain Icon" width="120"/>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
  [![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
</div>

## 📋 Project Overview

This project builds a **binary classification model** to predict whether it will rain in Melbourne, Australia, using historical weather data from 2008-2017. The project demonstrates comprehensive machine learning workflows including feature engineering, pipeline development, model optimization, and performance evaluation with imbalanced datasets.

### 🎯 Business Problem

**Goal**: Predict whether it will rain today in Melbourne based on yesterday's weather data.

**Real-world Applications**:
- 🏃 Planning outdoor activities and events
- 🌾 Agricultural decision-making and irrigation scheduling
- 🎪 Event management and logistics
- ☂️ Personal daily planning (umbrella decisions!)

**Key Challenge**: The dataset is highly imbalanced (~76% no-rain vs ~24% rain), requiring sophisticated evaluation beyond simple accuracy metrics.

---

## 📊 Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | Australian Bureau of Meteorology via Kaggle |
| **Time Period** | 2008 - 2017 (10 years) |
| **Original Size** | ~140,000 observations across Australia |
| **Filtered Size** | 7,557 observations (Melbourne region only) |
| **Target Variable** | RainToday (Binary: Yes/No) |
| **Features** | 22 weather-related features |

### 🌡️ Key Features

**Meteorological Measurements**:
- **Temperature**: Min/Max daily, readings at 9am and 3pm
- **Humidity**: Moisture levels at 9am and 3pm
- **Pressure**: Atmospheric pressure at 9am and 3pm
- **Wind**: Speed and direction measurements
- **Cloud Cover**: Sky obscuration levels
- **Precipitation**: Rainfall amounts, evaporation, sunshine hours

**Engineered Features**:
- **Season**: Australian seasons (Summer, Autumn, Winter, Spring)
- **Location**: Melbourne, Melbourne Airport, Watsonia (within 18km)

---

## 🏗️ Project Structure

```
Rainfall_Prediction/
│
├── rainfall_prediction_australia.ipynb    # Main notebook with full analysis
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```

---

## 🔬 Methodology

### 1. **Data Preprocessing**
- ✅ Handled 60%+ missing values by dropping incomplete rows
- ✅ Retained 56,000+ complete observations
- ✅ Addressed data leakage by reframing prediction target
- ✅ Filtered to Melbourne region for granular analysis

### 2. **Feature Engineering**
- 🍂 Created `Season` feature from dates (Australian seasons)
- 📍 Maintained `Location` as categorical for micro-variations
- 🔄 Renamed columns to reflect temporal shift (RainYesterday → RainToday)

### 3. **Data Leakage Prevention**

**Problem Identified**: Original target `RainTomorrow` using same-day features creates leakage
- Features like `Rainfall`, `Evaporation`, `Sunshine` aren't known until day ends

**Solution Implemented**: Reframe prediction task
- **Original**: Predict tomorrow's rain using today's data ❌
- **Revised**: Predict today's rain using yesterday's data ✅

This makes predictions **practical and actionable** for real-world use!

### 4. **Modeling Pipeline**

```python
Pipeline Architecture:
├── Preprocessing
│   ├── Numeric Features → StandardScaler (normalize to mean=0, std=1)
│   └── Categorical Features → OneHotEncoder (create binary columns)
└── Classification
    ├── Model 1: Random Forest Classifier
    └── Model 2: Logistic Regression
```

### 5. **Hyperparameter Optimization**

**Grid Search with 5-Fold Stratified Cross-Validation**:
- Maintains class distribution in each fold
- Tests multiple hyperparameter combinations
- Selects best model based on cross-validation performance

---

## 📈 Results & Performance

### Model Comparison

| Metric | Random Forest | Logistic Regression | Winner |
|--------|---------------|---------------------|--------|
| **Accuracy** | 84% | 83% | 🌲 RF |
| **Precision (Rain)** | 74% | 68% | 🌲 RF |
| **Recall (Rain)** | 51% | 51% | 🤝 Tie |
| **F1-Score (Rain)** | 0.60 | 0.58 | 🌲 RF |

### 🏆 Winner: **Random Forest Classifier**

**Advantages**:
- ✅ 1% higher overall accuracy
- ✅ 6% better precision (fewer false alarms)
- ✅ Better at capturing non-linear weather patterns
- ✅ Provides feature importance rankings

### ⚠️ Critical Limitation

**Both models achieve only 51% recall for rain** - meaning they miss approximately **half of all rainy days**!

**Why This Happens**:
- Class imbalance (76% no-rain vs 24% rain)
- Models biased toward predicting "no rain"
- Accuracy metric masks poor minority class performance

**Real-world Impact**: 
If you used these models to decide whether to bring an umbrella, you'd get caught in the rain **50% of the time**! 🌧️☂️

---

## 🌟 Key Insights

### Top 5 Most Important Features (Random Forest)

1. **Humidity3pm** - Afternoon moisture levels (strongest predictor)
2. **Sunshine** - Hours of sunshine (inversely related to rain)
3. **Cloud3pm** - Afternoon cloud cover
4. **Pressure3pm** - Atmospheric pressure
5. **RainYesterday** - Weather persistence pattern

**Insight**: Afternoon measurements (3pm) are more predictive than morning (9am) readings!

### Data Granularity Strategy

**Decision**: Focus on Melbourne region (3 nearby locations) rather than all of Australia

**Rationale**:
- Weather patterns vary drastically across Australia's geography
- Melbourne region has similar climate patterns
- More training data per pattern = better learning
- Results in a specialized, accurate model for one region

---

## 🎓 Learning Outcomes

This project demonstrates:

1. ✅ **End-to-end ML pipeline development** with scikit-learn
2. ✅ **Handling class imbalance** and understanding its impact
3. ✅ **Feature engineering** for temporal and categorical data
4. ✅ **Data leakage prevention** and temporal validation
5. ✅ **Model comparison** and proper evaluation techniques
6. ✅ **Why accuracy alone is misleading** for imbalanced data
7. ✅ **Production-ready code** with pipelines and best practices

---

## 🚀 Recommendations for Improvement

### 1. **Address Class Imbalance**
- Use SMOTE (Synthetic Minority Over-sampling Technique)
- Apply cost-sensitive learning with adjusted class weights
- Try under-sampling majority class strategically

### 2. **Optimize for Recall**
- Change grid search metric from accuracy to F1-score or recall
- Lower classification threshold below 0.5 to catch more rainy days
- Accept more false positives (false alarms) to reduce false negatives

### 3. **Advanced Feature Engineering**
- Create interaction features (e.g., Humidity × Pressure)
- Add rolling averages (3-day, 7-day weather trends)
- Include temporal features (day of year, days since last rain)
- Consider geographic features (distance from coast, elevation)

### 4. **Try Advanced Models**
- **XGBoost/LightGBM** with `scale_pos_weight` parameter
- **Neural Networks** for complex pattern recognition
- **Ensemble methods** combining multiple models
- **Focal Loss** for training with extreme imbalance

### 5. **Threshold Optimization**
- Use ROC curve analysis to find optimal threshold
- Implement custom threshold based on business requirements
- Balance precision vs recall based on use case

---

## 💡 Technical Highlights

### Code Quality Features
- ✅ Modular pipeline architecture
- ✅ Comprehensive documentation and comments
- ✅ Professional visualizations
- ✅ Reproducible results (fixed random seeds)
- ✅ Clean code following PEP 8 standards

### Best Practices Implemented
- ✅ Stratified train-test split
- ✅ Preprocessing within pipeline (prevents data leakage)
- ✅ Cross-validation for robust evaluation
- ✅ Multiple evaluation metrics (not just accuracy)
- ✅ Feature importance analysis

---

## 📦 Requirements

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone or download** the notebook
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Open Jupyter**: `jupyter notebook rainfall_prediction_australia.ipynb`
4. **Run all cells** sequentially (Kernel → Restart & Run All)
5. **Explore results** and modify hyperparameters

---

## 📚 References

- **Dataset**: [Kaggle - Weather Dataset Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Original Source**: [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/)
- **Scikit-learn Docs**: [Pipeline and GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
- **Class Imbalance**: [Imbalanced-learn Library](https://imbalanced-learn.org/stable/)

---

## 🎯 Future Work

- [ ] Implement SMOTE for better minority class handling
- [ ] Test XGBoost with optimized parameters
- [ ] Create web app for real-time predictions
- [ ] Expand to other Australian cities
- [ ] Add ensemble voting classifier
- [ ] Implement threshold optimization
- [ ] Create automated retraining pipeline


---


<div align="center">
  <p><strong>⭐ If you found this project helpful, please consider giving it a star!</strong></p>
  <p><em>Built with ❤️ using Python, scikit-learn, and pandas</em></p>
</div>
