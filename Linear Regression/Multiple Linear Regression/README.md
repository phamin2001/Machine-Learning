# 📊 Multiple Linear Regression Analysis

## 📝 Overview
This project demonstrates the implementation of Multiple Linear Regression to predict CO2 emissions from vehicles using multiple features simultaneously. The analysis extends simple linear regression by incorporating multiple predictors to improve prediction accuracy and model performance.

## 📂 Dataset
- **File**: `FuelConsumptionCo2.csv`
- **Description**: Contains vehicle specifications and their corresponding CO2 emissions
- **Selected Features**:
  - `ENGINESIZE`: Engine displacement in liters
  - `FUELCONSUMPTION_COMB_MPG`: Combined fuel consumption in MPG
  - `CO2EMISSIONS`: CO2 emissions in grams per kilometer (target variable)

## 🗂️ Project Structure
```
Multiple Linear Regression/
│
├── multiple_linear_regression.ipynb  # Main analysis notebook
├── FuelConsumptionCo2.csv            # Dataset
└── README.md                         # This file
```

## 🔬 Analysis Workflow

### 1️⃣ Data Exploration
- **Data Loading**: Import and examine the dataset structure
- **Statistical Summary**: Understand data distribution and characteristics
- **Correlation Analysis**: Identify relationships between features and avoid multicollinearity

### 2️⃣ Feature Engineering & Selection
- **Correlation Matrix**: Analyze feature relationships to select optimal predictors
- **Feature Selection**: Choose ENGINESIZE and FUELCONSUMPTION_COMB_MPG based on correlation analysis
- **Multicollinearity Avoidance**: Select features with high target correlation but low inter-correlation

### 3️⃣ Data Preprocessing
- **Feature Scaling**: Standardize features using StandardScaler for fair model training
- **Train-Test Split**: 80% training, 20% testing with reproducible random state
- **Data Visualization**: Scatter matrix to visualize feature relationships

### 4️⃣ Model Development
- **Multiple Linear Regression**: Train model using multiple features simultaneously
- **Coefficient Analysis**: Extract and interpret model parameters
- **Scale Transformation**: Convert coefficients back to original scale for interpretation

### 5️⃣ Model Evaluation & Visualization
- **Performance Metrics**:
  - 📉 Mean Squared Error (MSE)
  - 📉 Root Mean Squared Error (RMSE)
  - 📈 R² Score (Coefficient of Determination)
- **3D Visualization**: Interactive plot showing regression plane and data points
- **2D Plots**: Individual feature relationships with regression lines

## 🏆 Key Results

### Multiple Linear Regression Model
- **Linear Equation**: CO2 = β₀ + β₁ × Engine Size + β₂ × Fuel Consumption MPG
- **Interpretation**: Model considers both engine size (positive impact) and fuel efficiency (negative impact)
- **Advantage**: Better prediction accuracy compared to single-feature models

### Model Performance
- **R² Score**: Model explains variance in CO2 emissions using multiple predictors
- **RMSE**: Average prediction error in g/km
- **Improved Accuracy**: Multiple features provide better predictions than single-feature models

## 🛠️ Technologies Used
- **Python 3.x**
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `matplotlib`: Data visualization and 3D plotting
  - `scikit-learn`: Machine learning algorithms, preprocessing, and metrics

## ⚙️ Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Running the Analysis
1. 📥 Clone or download the project files
2. 📄 Ensure the dataset `FuelConsumptionCo2.csv` is in the same directory
3. 🚀 Open Jupyter Notebook:
   ```bash
   jupyter notebook multiple_linear_regression.ipynb
   ```
4. ▶️ Run all cells to reproduce the analysis

## 📊 Key Findings

### Statistical Insights
- **Multiple Features**: Using multiple predictors improves model performance
- **Feature Scaling**: Standardization ensures fair contribution from all features
- **Linear Relationships**: Multiple linear regression captures combined effects of features

### Advanced Visualizations
- **3D Regression Plane**: Visualizes how the model fits data in three-dimensional space
- **Scatter Matrix**: Shows pairwise relationships between all selected features
- **Individual Feature Plots**: Demonstrates each feature's contribution to predictions

## ⚠️ Model Limitations
- **Linear Assumptions**: Assumes linear relationships between features and target
- **Feature Selection**: Limited to selected features; other important variables may be excluded
- **Multicollinearity**: Requires careful feature selection to avoid correlated predictors

## 🚧 Future Enhancements
- **Polynomial Features**: Add polynomial terms to capture non-linear relationships
- **Regularization**: Implement Ridge/Lasso regression for better generalization
- **Feature Engineering**: Create interaction terms between features
- **Advanced Models**: Explore ensemble methods or neural networks
- **Cross-Validation**: Implement k-fold validation for robust model evaluation

## 🎓 Learning Objectives
This project demonstrates:
- Multiple linear regression implementation and interpretation
- Feature scaling and preprocessing techniques
- Advanced data visualization (3D plotting)
- Model evaluation with multiple metrics
- Coefficient interpretation in multiple regression context

## 🤝 Contributing
This is an educational project. Suggestions for improvements or additional analyses are welcome!

