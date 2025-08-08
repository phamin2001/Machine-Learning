# Random Forest vs XGBoost (California Housing)

A concise, runnable comparison of RandomForestRegressor and XGBRegressor on the California Housing dataset. The notebook trains both models with comparable settings, times training/inference, reports metrics, and visualizes predictions vs. ground truth.

## Notebook
- File: `Random_Forests_XGBoost_personal.ipynb`
- What it does:
  - Loads the California Housing dataset (from scikit-learn)
  - Splits data into train/test
  - Trains Random Forest and XGBoost with the same `n_estimators`
  - Times fit/predict for each model
  - Reports Mean Squared Error (MSE) and R² on the test set
  - Plots predicted vs. actual prices with a ±1σ band

## Requirements
- Python 3.9+
- Jupyter (VS Code or JupyterLab/Notebook)
- Packages: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`

Quick install (optional):
```bash
# macOS (zsh)
python -m pip install -U scikit-learn xgboost pandas numpy matplotlib seaborn
```

## How to run
1. Open `Random_Forests_XGBoost_personal.ipynb` in VS Code or Jupyter.
2. Run all cells.
   - If `ModuleNotFoundError: No module named 'xgboost'` appears, install it (see Troubleshooting) and rerun the import cell.
3. Review the printed metrics and the two scatter plots comparing predictions vs. actual values.

## Outputs
- Console metrics per model:
  - Test MSE
  - Test R²
  - Fit and predict times (seconds)
- Plots:
  - Random Forest: Predicted vs Actual (with ±1σ band)
  - XGBoost: Predicted vs Actual (with ±1σ band)

## Notes & Settings
- Both models use the same `n_estimators` for a fair comparison (see the notebook cell where models are created).
- No scaling is required; tree-based methods are invariant to monotonic feature scaling.
- Random seeds are set for reproducibility where supported.

## Troubleshooting
- xgboost not installed:
  ```bash
  # Option A: pip
  python -m pip install -U xgboost

  # Option B: conda
  conda install -c conda-forge xgboost
  ```
  After installing, restart the Jupyter kernel and rerun the imports.

- Plot display issues in VS Code: ensure the Python extension is enabled and the notebook kernel matches your environment.

## License
This folder inherits the repository’s license.
