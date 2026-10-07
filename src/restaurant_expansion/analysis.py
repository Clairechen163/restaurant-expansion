"""Load data, fit the model, rank candidate cities, and make plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import LinearRegression

# Units follow the classic version of this problem:
#   population in 10,000s of people, profit in $10,000s.
POP_UNIT, PROFIT_UNIT = 10_000, 10_000


def load_data(restaurants_csv, candidates_csv):
    train = pd.read_csv(restaurants_csv)
    cand = pd.read_csv(candidates_csv)
    for df, cols in ((train, {"city", "population", "profit"}), (cand, {"city", "population"})):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
    return train, cand


def rank_candidates(model: LinearRegression, cand: pd.DataFrame) -> pd.DataFrame:
    out = cand.copy()
    out["predicted_profit"] = model.predict(out["population"])
    out = out.sort_values("predicted_profit", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["recommend"] = out["predicted_profit"] > 0
    return out


def plot_fit(train, cand_ranked, model, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(train["population"], train["profit"], marker="x", c="tab:red", label="Existing restaurants")
    xs = np.linspace(0, max(train["population"].max(), cand_ranked["population"].max()) * 1.05, 100)
    ax.plot(xs, model.predict(xs), label=f"Fit: {model.intercept_:.2f} + {model.coef_:.2f}·pop")
    ax.scatter(cand_ranked["population"], cand_ranked["predicted_profit"],
               c="tab:green", marker="o", label="Candidate cities (predicted)")
    for _, r in cand_ranked.iterrows():
        ax.annotate(r["city"], (r["population"], r["predicted_profit"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("City population (10,000s)")
    ax.set_ylabel("Profit ($10,000s)")
    ax.set_title("Profit vs. city population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(model, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.cost_history_)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J")
    ax.set_title("Gradient descent convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(restaurants_csv, candidates_csv, out_dir="outputs", method="normal"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, cand = load_data(restaurants_csv, candidates_csv)
    model = LinearRegression(method=method).fit(train["population"], train["profit"])
    ranked = rank_candidates(model, cand)
    ranked.to_csv(out / "ranked_candidates.csv", index=False)
    plot_fit(train, ranked, model, out / "fit.png")
    if method == "gd":
        plot_convergence(model, out / "convergence.png")
    return model, ranked, model.score(train["population"], train["profit"])
