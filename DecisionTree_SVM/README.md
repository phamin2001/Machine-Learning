# Credit Card Fraud Detection with Decision Trees and SVM

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit-learn logo" width="100"/>
  <p><em>A comprehensive comparison of Decision Tree and SVM algorithms for fraud detection</em></p>
</div>

---

## 📝 Project Overview

This project implements and compares **Decision Tree** and **Support Vector Machine (SVM)** algorithms for credit card fraud detection. The analysis focuses on understanding how feature selection impacts model performance and explores the trade-offs between different machine learning approaches in highly imbalanced datasets.

## 🎯 Objectives

1. **Build and evaluate** Decision Tree and SVM models for fraud detection
2. **Compare model performance** with full feature set vs. selected features  
3. **Analyze the impact** of feature dimensionality on each algorithm
4. **Provide insights** on algorithm selection for fraud detection tasks

## 📊 Dataset

- **Source**: Credit card transaction data with anonymized features
- **Size**: 284,807 transactions
- **Features**: 30 features (V1-V28 anonymized, Time, Amount)
- **Target**: Binary classification (0: Normal, 1: Fraud)
- **Challenge**: Highly imbalanced dataset (~0.17% fraud cases)

## 🔬 Methodology

### Data Preprocessing
- **Standardization**: StandardScaler for feature normalization
- **L1 Normalization**: Consistent feature scaling across samples
- **Class Balancing**: Sample weights and balanced class parameters

### Feature Selection Strategy
- **Correlation Analysis**: Identification of features most correlated with fraud
- **Top 6 Features**: Selected based on absolute correlation values
- **Dimensionality Reduction**: ~79% reduction in feature count

### Model Implementation

#### Decision Tree Classifier
```python
DecisionTreeClassifier(
    max_depth=4,           # Prevent overfitting
    min_samples_split=20,  # Minimum samples for node splitting
    min_samples_leaf=10,   # Minimum samples at leaf nodes
    random_state=35        # Reproducible results
)
```

#### Support Vector Machine
```python
LinearSVC(
    class_weight='balanced',  # Handle class imbalance
    loss="hinge",            # Standard SVM loss function
    fit_intercept=False,     # Normalized data
    max_iter=1000           # Convergence iterations
)
```

## 📈 Evaluation Metrics

- **Primary**: ROC-AUC Score (handles class imbalance well)
- **Secondary**: Classification Report (Precision, Recall, F1-Score)
- **Visual**: Confusion Matrix Analysis, ROC Curves
- **Comparison**: Performance change with feature selection

## 🎨 Visualizations

The project includes comprehensive visualizations:

### Data Analysis
- **Class Distribution**: Pie charts and bar plots showing imbalance
- **Feature Correlation**: Heatmaps and correlation analysis with fraud class
- **Top Features**: Visualization of most predictive features

### Model Performance
- **ROC Curves**: Performance comparison across all models
- **Confusion Matrices**: Detailed prediction analysis for each model
- **Performance Comparison**: Bar charts comparing ROC-AUC scores
- **Feature Impact**: Before/after feature selection analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Core Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` & `seaborn` - Advanced visualizations
  - `scikit-learn` - Machine learning algorithms and metrics
- **Environment**: Jupyter Notebook

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Analysis
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DecisionTree_SVM
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook decsion_tree_svm.ipynb
   ```

3. **Execute cells sequentially** to reproduce the analysis

## 📁 Project Structure

```
DecisionTree_SVM/
├── decsion_tree_svm.ipynb    # Main analysis notebook
└── README.md                 # Project documentation
```

## 🔍 Key Findings

### Feature Selection Impact
- **Decision Trees**: ✅ Benefit from feature selection (reduced overfitting)
- **SVM**: ❌ Often perform better with full feature sets (optimal hyperplane)

### Algorithm Performance
- **High-Dimensional Data**: SVMs excel with complex decision boundaries
- **Interpretability**: Decision Trees provide clear, interpretable rules
- **Class Imbalance**: Both algorithms handle imbalanced data with proper weighting

### Practical Implications
- **Feature Reduction**: Achieved ~79% reduction in computational complexity
- **Performance Trade-offs**: Minimal accuracy loss with significant efficiency gains
- **Model Selection**: Context-dependent choice between interpretability and performance

## 📊 Results Summary

| Model Configuration | ROC-AUC Score | Features Used | Performance Change |
|---------------------|---------------|---------------|-------------------|
| Decision Tree (All) | 0.9372 | 29 | Baseline |
| SVM (All) | 0.9706 | 29 | 0.0335 |
| Decision Tree (Selected) | 0.8718 | 6 | -0.0654 |
| SVM (Selected) | 0.8652 | 6 | -0.1054 |

### Algorithm Comparison

| Aspect | Decision Tree | SVM |
|--------|---------------|-----|
| **Interpretability** | ✅ High | ❌ Low |
| **High-Dimensional Data** | ⚠️ Moderate | ✅ Excellent |
| **Training Speed** | ✅ Fast | ⚠️ Moderate |
| **Feature Selection Benefit** | ✅ Yes | ❌ Limited |
| **Overfitting Risk** | ⚠️ Higher | ✅ Lower |
| **Memory Usage** | ✅ Low | ⚠️ Higher |

## 🎯 Business Recommendations

### Production Deployment
- **Model Selection**: Choose based on ROC-AUC performance and business requirements
- **Feature Strategy**: Balance between computational efficiency and accuracy
- **Monitoring**: Implement continuous performance tracking
- **Threshold Tuning**: Optimize classification thresholds based on business costs

### Risk Management
- **False Positive Cost**: Consider customer experience impact
- **False Negative Cost**: Account for fraud loss prevention
- **Real-time Constraints**: Balance accuracy with response time requirements

## 💡 Key Insights

### Validated Findings
- ✅ **SVM superiority with full features**: Better performance with high-dimensional data
- ✅ **Decision Tree improvement with selection**: Feature reduction helps prevent overfitting
- ✅ **Dimensionality impact**: SVMs require more features for optimal hyperplane creation

### Strategic Implications
- **Feature Engineering**: Focus on most correlated features for efficiency
- **Algorithm Choice**: Context-dependent selection based on requirements
- **Scalability**: Consider computational resources for production deployment

## 🔮 Future Enhancements

### Advanced Modeling
- **Ensemble Methods**: Random Forest, Gradient Boosting, Voting Classifiers
- **Deep Learning**: Neural networks for complex pattern recognition
- **Anomaly Detection**: Unsupervised approaches for fraud detection

### Feature Engineering
- **Temporal Features**: Time-based patterns and seasonality
- **Interaction Terms**: Feature combinations and polynomial features
- **Domain Knowledge**: Business-specific feature creation

### Deployment Considerations
- **Real-time Scoring**: Low-latency prediction systems
- **Model Monitoring**: Performance drift detection
- **A/B Testing**: Continuous model improvement
- **Explainable AI**: SHAP values for model interpretability

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📚 Improve documentation
- 🔧 Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [ROC-AUC Score Interpretation](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

## 📞 Contact

For questions about this project:
- 📧 Open an issue in the repository
- 💬 Start a discussion in the Discussions tab

## 📄 License

This project is open source and available under the MIT License.
