import numpy as np
import pandas as pd
import pytest

from restaurant_expansion.evaluation import (
    kfold_cv,
    prediction_interval,
    residual_diagnostics,
    run_evaluation,
)
from restaurant_expansion.model import LinearRegression


def make_data(n=60, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(5, 23, n)
    return x, -3.9 + 1.19 * x + rng.normal(0, noise, n)


def test_cv_beats_mean_baseline():
    x, y = make_data()
    cv = kfold_cv(x, y)
    assert cv["rmse"] < cv["baseline_rmse"]
    assert 0 < cv["r2"] < 1


def test_cv_rmse_close_to_true_noise():
    x, y = make_data(n=400, noise=2.0, seed=1)
    assert kfold_cv(x, y, k=10)["rmse"] == pytest.approx(2.0, abs=0.25)


def test_cv_rejects_bad_k():
    x, y = make_data(n=10)
    with pytest.raises(ValueError):
        kfold_cv(x, y, k=1)
    with pytest.raises(ValueError):
        kfold_cv(x, y, k=11)


def test_prediction_interval_coverage_is_about_95_percent():
    rng = np.random.default_rng(2)
    hits, trials = 0, 600
    for _ in range(trials):
        x, y = make_data(n=40, noise=2.0, seed=int(rng.integers(1e9)))
        m = LinearRegression().fit(x, y)
        x0 = rng.uniform(5, 23)
        y0 = -3.9 + 1.19 * x0 + rng.normal(0, 2.0)
        lo, hi = prediction_interval(m, x, y, x0)
        hits += bool(lo <= y0 <= hi)
    assert 0.92 <= hits / trials <= 0.98


def test_interval_widens_away_from_the_mean_and_mean_interval_is_narrower():
    x, y = make_data()
    m = LinearRegression().fit(x, y)
    lo_c, hi_c = prediction_interval(m, x, y, x.mean())
    lo_f, hi_f = prediction_interval(m, x, y, 60.0)
    assert (hi_f - lo_f) > (hi_c - lo_c)
    lo_m, hi_m = prediction_interval(m, x, y, x.mean(), kind="mean")
    assert (hi_m - lo_m) < (hi_c - lo_c)


def test_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.linear_model")
    x, y = make_data()
    ours = LinearRegression("normal").fit(x, y)
    ref = sk.LinearRegression().fit(x.reshape(-1, 1), y)
    assert ours.coef_ == pytest.approx(ref.coef_[0])
    assert ours.intercept_ == pytest.approx(ref.intercept_)


def test_diagnostics_on_well_specified_data():
    x, y = make_data(n=200, seed=3)
    m = LinearRegression().fit(x, y)
    d = residual_diagnostics(m, x, y)
    assert abs(d["resid_mean"]) < 1e-8
    assert d["shapiro_p"] > 0.01


def test_run_evaluation_writes_outputs(tmp_path):
    x, y = make_data()
    df = pd.DataFrame({"city": range(len(x)), "population": x, "profit": y})
    res = run_evaluation(df, tmp_path)
    assert (tmp_path / "evaluation.txt").exists()
    assert (tmp_path / "residuals.png").exists()
    assert "Baseline" in res["report"]
