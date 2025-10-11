# Pipeline Diagnostics: PCA + k-NN

This mini-project mirrors the workflow from my original PCA pipeline experiments and packages it as a personal learning lab. The notebook (`pca_knn_pipeline_experiment.ipynb`) investigates how principal component analysis (PCA) interacts with a k-nearest neighbors (k-NN) classifier when both steps live inside a scikit-learn pipeline.

## Why it matters
- **Transformer vs estimator clarity**: PCA reshapes the feature space, while k-NN performs the actual classification. Keeping both inside a single pipeline reinforces this separation of responsibilities.
- **Leakage-safe tuning**: By wrapping the pipeline in `GridSearchCV`, every fold refits PCA and k-NN, ensuring hyperparameters are tuned without data leakage.
- **Dimensionality trade-offs**: The Wine dataset offers enough features to show how reducing dimensionality can change neighborhood structure and model accuracy.

## Notebook highlights
1. **Baseline comparison**: Evaluate a scaled k-NN pipeline without PCA to establish a reference accuracy.
2. **Grid search for pipelines**: Tune PCA variance thresholds and k-NN hyperparameters simultaneously, capturing the best combination as `best_model`.
3. **Cross-validation deep dive**: Use 5-fold cross-validation inside `GridSearchCV` so every hyperparameter candidate is evaluated on held-out folds, reinforcing how pipelines prevent leakage during tuning.
4. **Hold-out diagnostics**: Inspect classification metrics and a confusion matrix generated from the tuned pipeline.
5. **Explained variance study**: Visualize how many principal components are needed to retain 95% of the variance.
6. **Reflection and next steps**: Summarize insights and outline future experiments (e.g., alternate estimators, higher-dimensional datasets).

## Running the notebook
1. Activate your Python environment with scikit-learn, pandas, numpy, matplotlib, and seaborn installed.
2. Launch Jupyter Notebook or JupyterLab in this directory.
3. Open `pca_knn_pipeline_experiment.ipynb` and run the cells sequentially.

Feel free to treat the notebook as a template: swap in different datasets, extend the hyperparameter grid, or integrate additional preprocessing steps (e.g., feature selection) to deepen your model selection practice.
