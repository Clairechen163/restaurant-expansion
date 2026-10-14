"""Model evaluation: cross-validation, baseline, prediction intervals, residual diagnostics.

With only a few dozen cities a single train/test split is noisy, so k-fold
cross-validation is used. Every model is compared with a naive baseline that
always predicts the mean profit; a useful model must beat it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

from .model import LinearRegression


def kfold_cv(x, y, k: int = 5, seed: int = 0) -> dict:
    """K-fold cross-validation of the linear model vs. a mean-only baseline.

    Metrics are computed on out-of-fold predictions, i.e. on cities the model
    never saw while fitting.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= number of samples")

    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    pred = np.empty(n)
    base = np.empty(n)
    for fold in folds:
        train = np.setdiff1d(idx, fold)
        m = LinearRegression("normal").fit(x[train], y[train])
        pred[fold] = m.predict(x[fold])
        base[fold] = y[train].mean()

    def rmse(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "k": k,
        "n": n,
        "rmse": rmse(pred),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": 1 - float(np.sum((y - pred) ** 2)) / ss_tot,
        "baseline_rmse": rmse(base),
    }


def prediction_interval(model: LinearRegression, x_train, y_train, x_new,
                        level: float = 0.95, kind: str = "prediction"):
    """Classical interval for simple linear regression.

    kind="prediction": range for ONE new city's profit (includes city-to-city noise).
    kind="mean": range for the AVERAGE profit of cities with that population.
    Returns (lower, upper) arrays shaped like x_new.
    """
    if kind not in {"prediction", "mean"}:
        raise ValueError("kind must be 'prediction' or 'mean'")
    x = np.asarray(x_train, dtype=float).ravel()
    y = np.asarray(y_train, dtype=float).ravel()
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 points for an interval")
    x_new = np.asarray(x_new, dtype=float)

    resid = y - model.predict(x)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))          # residual standard error
    sxx = np.sum((x - x.mean()) ** 2)
    lever = 1.0 / n + (x_new - x.mean()) ** 2 / sxx
    se = s * np.sqrt(lever + (1.0 if kind == "prediction" else 0.0))
    t = stats.t.ppf(0.5 + level / 2, df=n - 2)
    center = model.predict(x_new)
    return center - t * se, center + t * se


def residual_diagnostics(model: LinearRegression, x, y) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    fitted = model.predict(x)
    resid = y - fitted
    shapiro_p = float(stats.shapiro(resid).pvalue) if resid.size >= 3 else float("nan")
    # Rough heteroscedasticity check: do residual magnitudes grow with population?
    rho, rho_p = stats.spearmanr(x, np.abs(resid))
    return {
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std(ddof=2)),
        "shapiro_p": shapiro_p,
        "abs_resid_vs_x_rho": float(rho),
        "abs_resid_vs_x_p": float(rho_p),
    }


def _thousands(value, _pos=None):
    return f"{value:,.0f}"


def plot_residuals(model: LinearRegression, x, y, path, target: str = "profit") -> None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    fitted = model.predict(x)
    resid = y - fitted

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(fitted, resid, c="tab:red", marker="x")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set(xlabel=f"Fitted {target} ($10,000s)", ylabel="Residual", title="Residuals vs. fitted")
    axes[1].hist(resid, bins=12, color="tab:blue", alpha=0.8)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q plot")
    for ax in (axes[0], axes[2]):
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[0].xaxis.set_major_locator(MaxNLocator(5))
    axes[0].xaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].xaxis.set_major_formatter(FuncFormatter(_thousands))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(train: pd.DataFrame, out_dir="outputs", k: int = 5, seed: int = 0,
                   target: str = "profit") -> dict:
    """Fit on all data, cross-validate, run diagnostics, and write a text report + plot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, y = train["population"].to_numpy(), train[target].to_numpy()
    model = LinearRegression("normal").fit(x, y)

    cv = kfold_cv(x, y, k=k, seed=seed)
    diag = residual_diagnostics(model, x, y)
    plot_residuals(model, x, y, out / "residuals.png", target)

    gain = 1 - cv["rmse"] / cv["baseline_rmse"]
    lines = [
        f"Evaluation ({cv['n']} cities, {k}-fold cross-validation, seed={seed})",
        f"Units: {target} in $10,000s",
        "",
        f"Model      RMSE = {cv['rmse']:.3f}   MAE = {cv['mae']:.3f}   out-of-fold R^2 = {cv['r2']:.3f}",
        f"Baseline   RMSE = {cv['baseline_rmse']:.3f}   (always predict the mean {target})",
        f"The model cuts the baseline error by {gain:.0%}.",
        "",
        "Residual diagnostics",
        f"  mean residual            {diag['resid_mean']:+.3f}   (should be ~0)",
        f"  Shapiro-Wilk p-value     {diag['shapiro_p']:.3f}   (p > 0.05: no evidence against normal errors)",
        f"  |resid| vs. population   Spearman rho = {diag['abs_resid_vs_x_rho']:+.2f}, "
        f"p = {diag['abs_resid_vs_x_p']:.3f}   (small p: spread changes with city size)",
    ]
    report = "\n".join(lines)
    (out / "evaluation.txt").write_text(report + "\n", encoding="utf-8")
    return {"cv": cv, "diagnostics": diag, "report": report}
