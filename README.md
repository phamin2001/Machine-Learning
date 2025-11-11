# 🧠 Machine Learning with Python

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit-learn logo" width="100"/>
  <p><em>A comprehensive collection of machine learning projects and implementations</em></p>
</div>

### 📝 **What You'll Find:**
- **📊 Regression Analysis**: Linear regression techniques for predictive modeling
- **🎯 Classification Methods**: Multi-class classification and fraud detection systems  
- **🌳 Ensemble Learning**: Random Forest and XGBoost comparative analysis
- **🔍 Clustering Algorithms**: K-Means, DBSCAN, and HDBSCAN for pattern discovery
- **🗺️ Geographic Analysis**: Spatial clustering and location intelligence
- **📉 Dimensionality Reduction**: PCA, t-SNE and UMAP for visualization and analysis
- **📊 Model Evaluation**: Comprehensive performance assessment and interpretation techniques
- **💼 Business Applications**: Real-world problem solving with actionable insights
- **📈 Advanced Visualizations**: From basic plots to interactive 3D and geographic analysis
- **🌧️ Weather Prediction**: Rainfall forecasting with imbalanced data handling
- **⚖️ Class Imbalance**: Techniques for handling imbalanced datasets effectively


## 📝 Overview

This repository contains a comprehensive collection of machine learning projects implemented in Python, covering both **supervised** and **unsupervised learning** algorithms. The projects span from fundamental regression techniques to advanced ensemble methods and customer segmentation analysis. Each project includes detailed analysis, professional visualizations, business insights, and educational content suitable for learning and portfolio demonstration.

## 📂 Project Structure

```
Machine Learning/
│
├── Linear Regression/
│   ├── Simple Linear Regression/
│   │   ├── simple_linear_regression.ipynb
│   │   ├── FuelConsumptionCo2.csv
│   │   └── README.md
│   │
│   └── Multiple Linear Regression/
│       ├── multiple_linear_regression.ipynb
│       ├── FuelConsumptionCo2.csv
│       └── README.md
│
├── Multi-class Classification/
│   ├── multi_class_classification.ipynb
│   └── Obesity_level_prediction_dataset.csv
│
├── DecisionTree_SVM/
│   ├── decsion_tree_svm.ipynb
│   └── README.md
│
├── Random Forest & XGBoost/
│   ├── random_forests_XGBoost.ipynb
│   └── README.md
│
├── Random Forest Evaluation/
│   ├── random_forest_evaluation.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── K-Means/
│   ├── K-Means_Customer_Segmentation_Personal.ipynb
│   └── README.md
│
├── DBSCAN_HDBSCAN_Clustering/
│   ├── dbscan_hdbscan_clustering.ipynb
│   ├── Canada.tif
│   ├── requirements.txt
│   └── README.md
│
├── PCA/
│   ├── pca_demo.ipynb
│   └── README.md
│
├── Pipelines_and_Model_Selection/
│   ├── pca_knn_pipeline_experiment.ipynb
│   └── README.md
│
├── Titanic_Survival_Prediction/
│   ├── titanic_survival_pipeline.ipynb
│   └── README.md
│
├── tSNE_UMAP_Dimension_Reduction/
│   ├── tSNE_UMAP_Dimension_Reduction.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── Evaluating_Classification_Models/
│   ├── Evaluating_Classification_Models.ipynb
│   ├── Evaluating_KMeans_Clustering.ipynb
│   ├── Regularization_in_Linear_Regression.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── Rainfall_Prediction/
│   ├── rainfall_prediction_australia.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── README.md (this file)
└── LICENSE
```

## 🔬 Current Projects

### 📊 Linear Regression
A comprehensive exploration of regression techniques for predicting CO2 emissions from vehicle characteristics.

#### Simple Linear Regression
- **Objective**: Predict CO2 emissions using single features
- **Key Features**:
  - Univariate analysis and visualization
  - Model evaluation with R², MSE, RMSE
  - Residual analysis and interpretation
  - Educational focus on fundamental concepts

#### Multiple Linear Regression
- **Objective**: Improve predictions using multiple features
- **Key Features**:
  - Feature correlation analysis and selection
  - Feature scaling and standardization
  - 3D visualization of regression plane
  - Coefficient interpretation in original scale
  - Advanced model evaluation techniques

### 🎯 Multi-class Classification
- **Objective**: Classify obesity levels using health and lifestyle data
- **Key Features**:
  - One-vs-All (OvA) and One-vs-One (OvO) strategies
  - Comprehensive data exploration and preprocessing
  - Feature importance analysis
  - Model comparison and evaluation
  - Detailed performance metrics and confusion matrices

### 🌳 Decision Tree vs SVM for Fraud Detection
- **Objective**: Compare Decision Tree and SVM algorithms for credit card fraud detection
- **Key Features**:
  - Comprehensive algorithm comparison on imbalanced dataset
  - Feature selection impact analysis (full vs. top 6 features)
  - Advanced evaluation metrics (ROC-AUC, classification reports)
  - Class imbalance handling with sample weights and balanced parameters
  - Professional visualizations: ROC curves, confusion matrices, correlation analysis
  - Business insights and production deployment recommendations

### 🚀 Random Forest vs XGBoost (California Housing)
- **Objective**: Compare RandomForestRegressor and XGBRegressor on the California Housing dataset
- **Key Features**:
  - Fair comparison with matching `n_estimators` and timed fit/predict
  - Test metrics: Mean Squared Error (MSE) and R²
  - Visualization: Predicted vs Actual scatter with ±1σ band for both models
  - Reproducibility with fixed random seeds; guidance if xgboost is missing
  - Notebook: `Random Forest & XGBoost/Random_Forests_XGBoost_personal.ipynb`

### 🌲 Random Forest Evaluation
- **Objective**: Comprehensive evaluation of Random Forest regression performance on California Housing data
- **Key Features**:
  - In-depth statistical metrics analysis (MAE, MSE, RMSE, R²) with business interpretation
  - Systematic residual analysis to identify model biases and limitations
  - Feature importance ranking with domain knowledge validation
  - Advanced visualization techniques for model assessment
  - Data quality impact analysis (skewed distributions, clipped values)
  - Educational focus on proper evaluation methodology beyond accuracy metrics
  - Notebook: `Random Forest Evaluation/random_forest_evaluation.ipynb`

### 🎯 K-Means Customer Segmentation
- **Objective**: Apply unsupervised learning to identify distinct customer segments for targeted marketing
- **Key Features**:
  - Comprehensive K-Means implementation with synthetic data validation
  - Real-world customer segmentation with business personas
  - Elbow method for optimal cluster selection and inertia analysis
  - Advanced 2D and interactive 3D visualizations
  - Customer personas with actionable marketing strategies
  - Business impact analysis and ROI estimation
  - Feature standardization and preprocessing for distance-based clustering

### 🗺️ DBSCAN vs HDBSCAN Geographic Clustering
- **Objective**: Compare density-based clustering algorithms on Canadian museum location data
- **Key Features**:
  - Comprehensive comparison of DBSCAN and HDBSCAN algorithms
  - Real-world geographic data from Statistics Canada (ODCAF)
  - Proper geographic coordinate scaling for distance calculations
  - Professional map visualizations with geopandas and contextily
  - Density-based clustering for varying geographic densities
  - Noise detection and outlier identification for rural locations
  - Educational content with detailed algorithm explanations
  - Practical applications for tourism, urban planning, and resource allocation

### 📉 PCA: Dimensionality Reduction & Visualization
- **Objective**: Build intuition for PCA with synthetic 2D data and reduce Iris (4D) to 2D
- **Key Features**:
  - Projection intuition: visualize projections onto PC1 and PC2
  - 2D visualization of Iris classes in PCA space
  - Explained variance and cumulative variance plots to choose number of PCs
  - Notebook: `PCA/pca_demo.ipynb`

### 🧪 Pipeline Diagnostics with PCA + k-NN
- **Objective**: Understand how PCA-driven feature transformations interact with k-NN classification inside a single evaluation pipeline
- **Key Features**:
  - Baseline vs PCA-enhanced pipeline comparison with cross-validation
  - Hold-out diagnostics including confusion matrix and classification report
  - Explained-variance curve to justify PCA component selection
  - Notebook: `Pipelines_and_Model_Selection/pca_knn_pipeline_experiment.ipynb`

### � Titanic Survival Prediction with ML Pipelines
- **Objective**: Build an end-to-end classification pipeline to predict passenger survival using mixed data types
- **Key Features**:
  - Automated preprocessing with ColumnTransformer for numerical and categorical features
  - Stratified cross-validation and GridSearchCV for imbalanced datasets
  - Model comparison: Random Forest vs Logistic Regression
  - Feature importance analysis and coefficient interpretation
  - Comprehensive confusion matrices and classification reports
  - Notebook: `Titanic_Survival_Prediction/titanic_survival_pipeline.ipynb`

### �🔄 t-SNE vs UMAP: Advanced Dimensionality Reduction
- **Objective**: Compare nonlinear dimensionality reduction techniques on synthetic 3D data
- **Key Features**:
  - Side-by-side comparison of PCA, t-SNE, and UMAP
  - Interactive 3D visualization of the original data
  - Parameter sensitivity analysis for t-SNE and UMAP
  - Educational explanations of algorithm differences and trade-offs
  - Standardization best practices for dimension reduction
  - Notebook: `tSNE_UMAP_Dimension_Reduction/tSNE_UMAP_Dimension_Reduction.ipynb`

### 📊 Evaluating Classification Models
- **Objective**: Train and compare classification models on the breast cancer dataset
- **Key Features**:
  - Data preprocessing with standardization and noise simulation
  - Implementation of KNN and SVM classifiers
  - Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
  - Confusion matrix visualization and interpretation
  - Special focus on medical context and error type implications
  - Notebook: `Evaluating_Classification_Models/Evaluating_Classification_Models.ipynb`
  - Companion practice notebook covering k-means cluster evaluation (silhouette, inertia, Davies–Bouldin, Voronoi diagnostics): `Evaluating_Classification_Models/Evaluating_KMeans_Clustering.ipynb`
  - Experimental regularization lab with Ridge, Lasso, Elastic Net comparisons and feature-selection workflow: `Evaluating_Classification_Models/Regularization_in_Linear_Regression.ipynb`

### 🌧️ Australian Weather - Rainfall Prediction Classifier
- **Objective**: Build a binary classifier to predict whether it will rain today in Melbourne using yesterday's weather data
- **Key Features**:
  - **Data Leakage Prevention**: Reframed prediction task to avoid using same-day features
  - **Feature Engineering**: Created seasonal features and handled temporal data
  - **Data Granularity Strategy**: Focused on Melbourne region (3 locations) for localized accuracy
  - **Pipeline Development**: Robust ML pipeline with preprocessing (StandardScaler + OneHotEncoder)
  - **Model Comparison**: Random Forest (84% accuracy) vs Logistic Regression (83% accuracy)
  - **Class Imbalance Analysis**: Deep dive into handling 76% no-rain vs 24% rain distribution
  - **Performance Metrics**: Comprehensive evaluation beyond accuracy (precision, recall, F1-score)
  - **Feature Importance**: Identified top predictors (Humidity3pm, Sunshine, Cloud3pm)
  - **Critical Insight**: Both models achieve only 51% recall for rain (miss ~50% of rainy days)
  - **Business Context**: Practical recommendations for improving minority class detection
  - **Educational Value**: Demonstrates why accuracy alone is misleading for imbalanced datasets
  - **Dataset**: 7,557 observations from Australian Bureau of Meteorology (2008-2017)
  - Notebook: `Rainfall_Prediction/rainfall_prediction_australia.ipynb`

## 🛠️ Technologies Used

- **Python 3.x**
- **Core Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `matplotlib`: Data visualization
  - `seaborn`: Statistical visualization and enhanced plots
  - `scipy`: Spatial analysis, distance metrics, and Voronoi diagrams
  - `scikit-learn`: Machine learning algorithms and tools
  - `xgboost`: Gradient-boosted trees implementation
  - `plotly`: Interactive 3D visualizations
  - `hdbscan`: Hierarchical density-based clustering
  - `geopandas`: Geographic data processing and visualization
  - `contextily`: Basemap integration for geographic plots
  - `umap-learn`: Uniform Manifold Approximation and Projection for dimension reduction
- **Development Environment**: Jupyter Notebook
- **Version Control**: Git

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Machine Learning with Python"
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly jupyter hdbscan geopandas contextily shapely requests umap-learn
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Navigate to any project folder and open the respective notebook**

## 📊 Key Learning Outcomes

### 📈 Regression Analysis
- Understanding linear relationships between variables
- Feature selection and engineering techniques
- Model evaluation and interpretation
- Handling multicollinearity and scaling issues

### 🎯 Classification Methods
- Multi-class classification strategies
- Binary classification for fraud detection
- Model comparison and selection techniques
- Feature importance and correlation analysis
- Performance evaluation metrics and ROC analysis
### 🌳 Advanced Classification Algorithms
- **Decision Trees**: Interpretable models with depth control and pruning
- **Support Vector Machines**: High-dimensional data classification with kernel methods
- **Feature Selection**: Impact analysis and dimensionality reduction strategies
- **Imbalanced Data Handling**: Sample weighting and balanced classification approaches

### 🎯 Unsupervised Learning & Clustering
- **K-Means Clustering**: Customer segmentation and pattern discovery
- **DBSCAN Clustering**: Density-based clustering with noise detection
- **HDBSCAN Clustering**: Hierarchical density-based clustering for varying densities
- **Geographic Clustering**: Spatial data analysis with proper coordinate scaling
- **Elbow Method**: Optimal cluster number selection techniques
- **Feature Standardization**: Preprocessing for distance-based algorithms
- **Business Translation**: Converting technical clusters into actionable insights
- **Customer Personas**: Marketing strategy development from clustering results

### 📋 General ML Skills
- Data preprocessing and exploration
- Visualization techniques for ML insights
- Model validation and testing methodologies
- Professional code documentation and analysis
- Business-oriented model interpretation and recommendations

## 🏆 Project Highlights

### 📊 **Comprehensive Analysis**
- Each project includes detailed exploratory data analysis
- Statistical insights and data quality assessments
- Professional visualizations and interpretations
- Algorithm comparison and performance benchmarking

### 🔍 **Educational Value**
- Step-by-step explanations of algorithms
- Code comments and detailed documentation
- Theoretical background and practical implementation
- Real-world problem-solving approaches

### 📈 **Professional Quality**
- Clean, well-organized code structure
- Reproducible results with fixed random seeds
- Industry-standard evaluation metrics
- Business insights and deployment considerations

### 🎯 **Advanced Techniques**
- Feature engineering and selection strategies
- Class imbalance handling methods
- Multiple algorithm comparison frameworks
- Production-ready model evaluation pipelines

## 🔍 Specialized Focus Areas

### 🚨 **Fraud Detection & Security**
- Imbalanced dataset handling techniques
- False positive/negative cost analysis
- Real-time scoring considerations
- Risk management and business impact assessment

### 📊 **Feature Engineering**
- Correlation analysis and feature selection
- Dimensionality reduction impact studies
- Feature scaling and normalization techniques
- Domain-specific feature creation strategies

### ⚖️ **Model Comparison Frameworks**
- Systematic algorithm evaluation methodologies
- Performance trade-off analysis
- Computational efficiency considerations
- Interpretability vs. accuracy balance

### 🎯 **Customer Analytics & Business Intelligence**
- Customer segmentation and persona development
- Marketing strategy optimization through clustering
- Business impact assessment and ROI calculation
- Data-driven customer relationship management

### 🗺️ **Geographic Data Analysis**
- Spatial clustering and geographic pattern recognition
- Coordinate system handling and scaling techniques
- Density-based clustering for irregular spatial distributions
- Geographic visualization with basemap integration
- Location intelligence for business and policy decisions

## ⚠️ Prerequisites

- Basic understanding of Python programming
- Familiarity with data science concepts
- Knowledge of basic statistics and linear algebra
- Understanding of machine learning fundamentals
- Basic knowledge of classification and regression concepts

## 🎓 Educational Use

This repository is designed for:
- **Students** learning machine learning concepts and implementations
- **Professionals** looking to understand practical algorithm comparisons
- **Data Scientists** seeking comprehensive evaluation frameworks
- **Instructors** needing detailed examples for teaching ML concepts
- **Portfolio development** for demonstrating ML skills to employers

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements or new features
- Improve documentation and explanations
- Add new machine learning projects or algorithms
- Enhance visualization and analysis techniques

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities, please feel free to reach out through the repository's issue tracker.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code with attribution.

---

<div align="center">
  <strong>Happy Learning! 🎯📚</strong>
  <br>
  <em>Building knowledge through practical machine learning implementation.</em>
</div>