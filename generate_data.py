"""
Synthetic high-dimensional data with a known low-dimensional manifold.

The signal lives on a 2-D manifold (a Swiss roll: 1-D position along the
roll, plus 1-D height) embedded in a 3-D ambient space, then padded with
extra Gaussian-noise dimensions to make the problem high-dimensional.

Because we generate the data ourselves, we keep:
  - `t`: the 1-D position-along-the-roll parameter (the "color" used to
         judge whether a projection unrolled the manifold faithfully)
  - the original 3-D Swiss roll coordinates (for ground-truth viz)

A good non-linear method (t-SNE / UMAP / Isomap) should produce a 2-D
embedding where points are ordered by `t`. PCA, being linear, *cannot*
unroll a Swiss roll — it can only project it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    n_samples: int = 1500
    n_noise_dims: int = 7
    noise_std: float = 0.6
    seed: int = 42


def generate(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X         : (n_samples, 3 + n_noise_dims)  ambient high-D coords
    swiss_3d  : (n_samples, 3)                 original Swiss-roll 3-D coords
    t         : (n_samples,)                   1-D position along the roll
    """
    rng = np.random.default_rng(cfg.seed)

    # `t` parameterizes the position along the rolled-up curve.
    t = 1.5 * np.pi * (1 + 2 * rng.random(cfg.n_samples))      # 1.5π → 4.5π
    h = 21 * rng.random(cfg.n_samples)                          # height

    x = t * np.cos(t)
    z = t * np.sin(t)
    y = h
    swiss_3d = np.column_stack([x, y, z])

    # Pad with extra Gaussian noise dimensions.
    if cfg.n_noise_dims > 0:
        noise = rng.normal(0.0, cfg.noise_std, size=(cfg.n_samples, cfg.n_noise_dims))
        X = np.hstack([swiss_3d, noise])
    else:
        X = swiss_3d

    return X, swiss_3d, t


def save(out_dir: Path, X: np.ndarray, swiss_3d: np.ndarray, t: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["x1", "x2", "x3"] + [f"x{i+4}_noise" for i in range(X.shape[1] - 3)]
    pd.DataFrame(X, columns=cols).to_csv(out_dir / "X.csv", index=False)
    pd.DataFrame(swiss_3d, columns=["sx", "sy", "sz"]).to_csv(out_dir / "swiss_3d.csv", index=False)
    pd.Series(t, name="t").to_csv(out_dir / "t.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Swiss-roll-in-noise data.")
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--n-noise-dims", type=int, default=7)
    p.add_argument("--noise-std", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(
        n_samples=args.n_samples,
        n_noise_dims=args.n_noise_dims,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    X, swiss_3d, t = generate(cfg)
    save(args.out_dir, X, swiss_3d, t)
    print(f"Generated {len(X)} points in {X.shape[1]}-D "
          f"(3 informative + {cfg.n_noise_dims} noise).")
    print(f"Saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
