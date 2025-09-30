# Evaluating Classification Models

This practice folder now contains two complementary notebooks: one focused on **classification model evaluation** and another that deepens your intuition for **k-means clustering**. Both notebooks include detailed commentary, helper functions, and reflection prompts so you can revisit the course material at your own pace.

## Contents
- `Evaluating_Classification_Models.ipynb` — Classification workflow using the Wisconsin breast cancer dataset, covering KNN and SVM with extensive evaluation
- `Evaluating_KMeans_Clustering.ipynb` — Guided exploration of k-means clustering, including silhouette analysis, metric comparisons, and edge cases for non-spherical data
- `README.md` — Documentation (this file)
- `requirements.txt` — Python dependencies for running the notebooks

## Features
- Data preprocessing and standardization for tabular classification tasks
- Noise simulation to stress-test KNN and SVM classifiers
- Comprehensive evaluation with accuracy, precision, recall, and F1-score, plus annotated confusion matrices
- End-to-end k-means walkthrough with reusable silhouette plotting helper
- Visual comparison of inertia, silhouette scores, and Davies–Bouldin index for multiple values of *k*
- Experiments showing the limits of k-means on non-spherical data (with Voronoi diagrams for intuition)

## How to use

1. Create a virtual environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch the notebooks:

```bash
jupyter notebook Evaluating_Classification_Models.ipynb
jupyter notebook Evaluating_KMeans_Clustering.ipynb
```

## Key Concepts Covered
- Binary classification for medical diagnosis
- Feature standardization and noise handling
- Model comparison and selection
- Error analysis with a focus on minimizing false negatives in medical applications
- Visualization of model performance
- Silhouette analysis, inertia, and Davies–Bouldin scores for clustering
- Assessing k-means stability and the impact of centroid initialisation
- Recognising when to switch from k-means to density-based clustering methods

## Requirements

The notebook requires the following Python packages:
- numpy: For numerical operations and array handling
- pandas: For data manipulation and DataFrame operations
- scikit-learn: For machine learning algorithms and evaluation metrics
- matplotlib: For creating plots and visualizations
- seaborn: For enhanced statistical visualizations
- scipy: For Voronoi diagrams in the k-means notebook

## Learning Outcomes
After working through this notebook, you should be able to:
1. Understand how to properly prepare data for classification tasks
2. Implement and compare different classification algorithms
3. Interpret various evaluation metrics in the context of the problem domain
4. Make informed decisions about model selection based on appropriate metrics
5. Understand the importance of different error types in medical applications
6. Diagnose k-means solutions using multiple qualitative and quantitative tools
7. Identify scenarios where k-means breaks down and propose suitable alternatives


## License
This project is available under the MIT License.
