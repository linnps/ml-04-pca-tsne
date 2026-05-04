<div align="center">

# Dimensionality Reduction — PCA vs. t-SNE on a Hidden Manifold

**Hide a 2-D manifold inside a 10-D space, then ask each method to find it.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Generate a Swiss roll — a 2-D surface curled up in 3-D — and embed it in 10-D by adding 7 noise dimensions. PCA, by definition, can only project. t-SNE can *unroll*. The 3-D plot below is the truth; the 4-panel comparison shows whether each algorithm recovered it.

<table>
<tr>
<td align="center" width="33%">
<sub>PCA (linear)</sub><br>
<b style="font-size:1.5em; color:#C04040;">trustworthiness 0.971</b><br>
<sub>preserves global, can't unroll</sub>
</td>
<td align="center" width="33%">
<sub>t-SNE (perp = 30)</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">trustworthiness 0.999</b><br>
<sub>local structure preserved beautifully</sub>
</td>
<td align="center" width="33%">
<sub>Noise rejection</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">PC4–PC10 ≈ 0.3%</b><br>
<sub>PCA correctly ignores all 7 noise dims</sub>
</td>
</tr>
</table>

| Method | Trustworthiness @k=10 | Captures global geometry? | Captures local geometry? |
|---|---:|:---:|:---:|
| PCA | 0.971 | ✅ | ⚠️  fails to unroll non-linear manifolds |
| t-SNE (perp=5) | 0.998 | ❌ (no notion of global) | ✅ |
| **t-SNE (perp=30)** | **0.999** | ❌ | **✅** |
| t-SNE (perp=80) | 0.999 | ❌ | ✅ (but smoother) |

<sub>**Headline finding:** these two methods don't compete for the same prize. PCA gives you a faithful global picture but can't unroll curvature. t-SNE gives you a perfect local picture but distorts global distances on purpose. Choose by the question, not by which gives the prettier plot.</sub>

---

## Dashboard

### 1. The truth — what we're trying to recover

![swiss roll 3D](assets/01_swiss_3d.png)

A 2-D surface curled into 3-D, then padded with 7 dimensions of pure Gaussian noise (not shown). Color encodes the 1-D position along the roll — the latent variable a good projection should preserve as a smooth color gradient.

### 2. Projections side-by-side

![projections](assets/02_projections.png)

This is the figure that answers the question.

- **PCA** finds the two directions of largest variance and projects onto them. Because the Swiss roll has a curved structure, the projection looks like a flattened spiral — points that are *far apart along the roll* (red and blue) end up next to each other in 2-D. PCA *cannot* fix this; it's a linear method, and unrolling is not a linear operation.
- **t-SNE at perplexity 5** keeps very local neighborhoods intact but breaks the manifold into disconnected island-shaped clumps. Useful if you only care about cluster membership — misleading if you want to see continuous structure.
- **t-SNE at perplexity 30** is the sweet spot here: a clean, continuous unrolling where the color gradient flows smoothly from one end of the embedding to the other.
- **t-SNE at perplexity 80** uses very large neighborhoods, so it preserves more global structure than perplexity 5 but at the cost of crispness — borders are softer.

The "right" perplexity isn't fixed; it depends on the data's intrinsic neighborhood scale. Try several. Always.

### 3. PCA scree plot — variance per component

![scree](assets/03_scree.png)

A nice diagnostic that PCA can do but t-SNE cannot: tell you *how many dimensions actually matter*. The first three components capture ~98% of the variance (the three informative dims of our Swiss roll), and components 4–10 each contribute ~0.3% (the 7 noise dimensions). The cumulative red curve flattens hard right after PC3 — the visual signature of "this dataset has 3 informative dimensions."

### 4. Trustworthiness — quantifying local fidelity

![trustworthiness](assets/04_trustworthiness.png)

Trustworthiness ∈ [0, 1] measures: *of the k nearest neighbors of point p in the projection, what fraction were actually neighbors of p in the original space?* All three t-SNE projections sit at ≥ 0.998 across every k tested. PCA stays at ~0.97 — high but not perfect, because folding a curved manifold flat is never quite faithful. Above ~0.95 the differences are small in human terms; the qualitative gap in panel 2 is the more important diagnostic.

---

## What's actually happening

### PCA — diagonalize the covariance matrix

Find the orthogonal directions in which the data has the most variance. Project onto the top `k` of them.

- **What it preserves**: variance, global pairwise distances, Euclidean angles.
- **What it cannot do**: unfold curvature. A line on a Swiss roll's surface, projected, becomes a chord of a circle — a different distance.
- **Bonus**: tells you *how much variance each dimension carries*, which doubles as an answer to "how many dimensions does this data really have?"

### t-SNE — match neighborhood probabilities

For each point, define a probability distribution over "who my neighbors are" in the high-D space (Gaussian-weighted). Define the same kind of distribution in low-D (using a heavy-tailed Student-t to allow more breathing room). Find a 2-D layout that minimizes the KL divergence between the two distributions.

- **What it preserves**: local neighborhood structure — who's near whom.
- **What it intentionally distorts**: global distances. Two clusters that look far apart in t-SNE may not be far apart in reality.
- **Knob that matters most**: perplexity. Roughly "how many neighbors define a neighborhood." Too small → islands. Too large → blob.
- **Reproducibility caveat**: t-SNE is non-convex and stochastic. Different seeds yield visually different layouts but qualitatively similar structure.

### Mental model

| If you want… | Use |
|---|---|
| A **denoising** preprocessor before a linear model | PCA |
| To know how many "real" dimensions your data has | PCA scree plot |
| A **2-D scatter for visual inspection** of clusters / structure | t-SNE (or UMAP) |
| **Distance-faithful** projections for downstream geometry | PCA — never t-SNE |

UMAP wasn't included here to keep the dependency footprint small, but it's the natural third member of this family: similar local fidelity to t-SNE, much faster, and with somewhat better global preservation.

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
```

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py):

```python
DataConfig(
    n_samples=1500,
    n_noise_dims=7,    # how many distractor dims wrap the manifold
    noise_std=0.6,     # how loud the distractors are
    seed=42,
)
```

Try `n_noise_dims=50` (PCA's noise rejection still works; t-SNE may need higher perplexity to ignore the noise) or change the manifold inside `generate()` to an S-curve or a sphere — both expose t-SNE's perplexity sensitivity in different ways.

---

## Project layout

```
04-pca-tsne/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← Swiss-roll-in-noise generator (deterministic)
├── train.py               ← PCA + t-SNE @ 3 perplexities + figures
├── assets/                ← rendered dashboard figures (4 PNGs)
└── results/metrics.json
```

---

## What I learned

- **PCA's "failure" on a Swiss roll isn't a bug — it's the definition of "linear method."** No clever choice of components will unroll a curved manifold; that's what non-linear methods are for.
- **t-SNE's perplexity is not a hyperparameter you set once.** It's a *lens choice*: small perplexity = microscope, large perplexity = wide-angle. You should look at multiple perplexities before believing any structure you see.
- **Trustworthiness numbers compress the story too much.** All three t-SNE settings score ≥ 0.998 here, but visually they look very different. The metric tells you "no major neighbors were lost," not "this is the right way to look at the data."
- **A scree plot doubles as a built-in noise detector.** When 7 of your 10 dimensions are pure noise, PC4–PC10 each capturing ~0.3% is the empirical signature of "those 7 dimensions are noise" — and it's the kind of diagnostic that synthetic data lets you trust completely.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
