# PCA: Dimensionality Reduction & Visualization

This project demonstrates Principal Component Analysis (PCA) through two compact examples inside a Jupyter Notebook:

- Synthetic 2D data projected onto its two principal axes (projection intuition).
- Iris dataset reduced from 4 features to 2 principal components (2D visualization + explained variance).

## What you’ll learn
- How PCA finds directions of maximum variance (principal components).
- How to project data onto PCs and visualize those projections.
- How much variance is retained when reducing dimensions (explained variance and cumulative explained variance).

## Project structure
- `pca_demo.ipynb` — notebook with documented steps for both examples.
- `requirements.txt` — pinned dependencies for reproducibility.

## Quick start (notebook)

Create a virtual environment, install dependencies, then open and run the notebook:

```zsh
# Optional: create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open the notebook in VS Code or Jupyter and run all cells
# (VS Code) Use the Notebook UI to run cells
# (Jupyter) jupyter notebook pca_demo.ipynb
```

## Visualizations
The notebook renders the following visualizations inline:
- Synthetic 2D data with projections onto PC1 and PC2.
- Iris dataset projected to 2D (PC1 vs PC2), colored by species.
- Explained variance per component with cumulative explained variance.

## Notes
- Reducing Iris from 4D to 2D typically retains ~95–96% of total variance (PC1+PC2). 100% would require keeping all 4 components.
- Standardize features before PCA so large-scale features don’t dominate the components.

## License
This project is for educational use in your personal GitHub repo.
