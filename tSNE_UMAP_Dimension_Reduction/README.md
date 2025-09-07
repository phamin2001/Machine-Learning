# t-SNE vs UMAP vs PCA: Dimensionality Reduction on Synthetic Data

This mini-project demonstrates and explains how three popular dimensionality reduction techniques compare on a simple, synthetic dataset:

- PCA (linear, baseline)
- t-SNE (nonlinear, manifold learning)
- UMAP (nonlinear, manifold learning)

You’ll generate 3D Gaussian blobs, visualize them in 3D, then project to 2D with each method and discuss the results.

## Why this project
- Learn the strengths/limitations of each method with an easy visual example.
- Practice standardization and good plotting practices.
- Get a ready-to-run notebook you can share.

## Project structure

- `tSNE_UMAP_Dimension_Reduction.ipynb` — Educational notebook with step-by-step explanations.
- `requirements.txt` — Dependencies to run the notebook.

## Setup

Install dependencies (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n dr-demo python=3.10 -y
conda activate dr-demo
pip install -r requirements.txt
```

Note: If Plotly figures don’t render in your environment, ensure you’re running the notebook kernel that has these packages installed.

## Run

Open the notebook and run cells top-to-bottom:

```bash
code tSNE_UMAP_Dimension_Reduction.ipynb
```

Or with Jupyter:

```bash
jupyter notebook tSNE_UMAP_Dimension_Reduction.ipynb
```

## Notes and tips
- Standardize your features before t-SNE/UMAP/PCA (`StandardScaler`).
- t-SNE is sensitive to `perplexity` and often benefits from trying a few values (e.g., 5–50).
- UMAP’s `n_neighbors`, `min_dist`, and `spread` influence cluster tightness and separation.
- PCA is fast and often a strong baseline; non-linear methods aren’t always “better.”

## License
MIT
