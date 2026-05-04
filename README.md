# 04 — Dimensionality Reduction (PCA & t-SNE)

> Status: 🟡 Planned (skeleton only)

## Goal

Show what high-dimensional data actually looks like. Use PCA for a linear
view of variance and t-SNE / UMAP for a non-linear view of local structure,
then make the trade-off between the two explicit.

## Topics covered

- PCA: explained variance, scree plot, when it's appropriate
- t-SNE: perplexity, the "don't trust global distances" caveat
- UMAP as a faster, often more faithful alternative
- Visualizing class structure in unsupervised projection
- Using PCA as a denoising preprocessor for downstream models

## Dataset

**Self-generated synthetic data — no third-party / copyrighted datasets.**
A custom generator (`generate_data.py`) produces high-dimensional points
that lie on a known low-dimensional manifold (Swiss roll or S-curve),
with optional noise dimensions added to test whether each method ignores
them. Because the true 2D structure is known, projection quality is
quantitative — we can measure how well neighborhood relationships
survive, not just eyeball the scatter plot.

## Tech stack

- Python 3.10+
- numpy, scikit-learn, umap-learn
- matplotlib / plotly for 2D scatter

## Run

```bash
pip install -r requirements.txt
python project.py
```

## Results

_To be filled in after implementation._

## What I learned

_To be filled in after implementation._

---
*Part of the [machine-learning portfolio](../README.md).*
