import numpy as np
import pandas as pd
import pytest

from restaurant_expansion.analysis import rank_candidates, run
from restaurant_expansion.evaluation import (
    cv_interval_coverage,
    holdout_check,
    kfold_cv,
    prediction_interval,
    residual_diagnostics,
    run_evaluation,
    summarize_holdout,
)
from restaurant_expansion.model import LinearRegression, LogLogRegression


def power_law(n=400, b=1.0, sigma=0.3, seed=0):
    """sales = 2000 * pop**b * exp(noise): spread proportional to size."""
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(2.8, 0.9, n)).clip(5, 300)
    y = 2000 * x ** b * np.exp(rng.normal(0, sigma, n))
    return x, y


def test_recovers_elasticity_and_scale():
    x, y = power_law(n=2000, b=1.2, sigma=0.2)
    m = LogLogRegression().fit(x, y)
    assert m.coef_ == pytest.approx(1.2, abs=0.03)
    assert m.intercept_ == pytest.approx(np.log(2000), abs=0.15)
    # smearing factor ~ exp(sigma^2 / 2) for lognormal noise
    assert m.smearing_ == pytest.approx(np.exp(0.2 ** 2 / 2), abs=0.01)


def test_prediction_is_unbiased_on_original_scale():
    x, y = power_law(n=3000, sigma=0.4)
    m = LogLogRegression().fit(x, y)
    assert m.predict(x).sum() == pytest.approx(y.sum(), rel=0.03)


def test_rejects_non_positive_values():
    with pytest.raises(ValueError):
        LogLogRegression().fit([1, 2, 0], [1, 2, 3])
    with pytest.raises(ValueError):
        LogLogRegression().fit([1, 2, 3], [1, -2, 3])


def test_log_model_fixes_the_spread_problem_that_linear_has():
    x, y = power_law()
    lin = residual_diagnostics(LinearRegression().fit(x, y), x, y)
    log = residual_diagnostics(LogLogRegression().fit(x, y), x, y)
    assert lin["abs_resid_vs_x_p"] < 0.01          # linear: spread grows with size
    assert log["abs_resid_vs_x_p"] > 0.01          # log: spread roughly constant
    assert abs(log["abs_resid_vs_x_rho"]) < abs(lin["abs_resid_vs_x_rho"])


def test_log_model_has_lower_percentage_error_on_proportional_noise():
    x, y = power_law(sigma=0.3)
    lin, log = kfold_cv(x, y, k=5), kfold_cv(x, y, k=5, log=True)
    assert log["median_ape"] < lin["median_ape"]


def test_percentage_error_is_skipped_when_target_can_be_negative():
    x = np.arange(1.0, 21.0)
    cv = kfold_cv(x, x - 5, k=4)
    assert cv["mape"] is None and cv["median_ape"] is None


def test_log_prediction_interval_covers_about_95_percent_and_is_multiplicative():
    rng = np.random.default_rng(1)
    hits, trials = 0, 1500
    for _ in range(trials):
        x, y = power_law(n=60, sigma=0.3, seed=int(rng.integers(1e9)))
        m = LogLogRegression().fit(x, y)
        x_new = np.array([20.0])
        y_new = 2000 * 20.0 * np.exp(rng.normal(0, 0.3))
        lo, hi = prediction_interval(m, x, y, x_new)
        hits += bool(lo[0] <= y_new <= hi[0])
    assert 0.92 <= hits / trials <= 0.98

    x, y = power_law()
    m = LogLogRegression().fit(x, y)
    lo, hi = prediction_interval(m, x, y, np.array([10.0, 100.0]))
    assert (hi / lo)[0] == pytest.approx((hi / lo)[1], rel=0.1)   # same relative width
    assert (hi - lo)[1] > 5 * (hi - lo)[0]                       # much wider in dollars


def test_run_and_evaluation_with_log(tmp_path):
    x, y = power_law(n=80)
    train = pd.DataFrame({"city": [f"c{i}" for i in range(80)], "population": x, "sales": y})
    cand = pd.DataFrame({"city": ["a", "b"], "population": [12.0, 90.0]})
    train.to_csv(tmp_path / "r.csv", index=False)
    cand.to_csv(tmp_path / "c.csv", index=False)

    model, ranked, r2 = run(tmp_path / "r.csv", tmp_path / "c.csv", tmp_path, target="sales", log=True)
    assert isinstance(model, LogLogRegression) and r2 > 0.5
    assert ranked.loc[0, "city"] == "b" and (tmp_path / "fit.png").exists()

    res = run_evaluation(train, tmp_path, target="sales", log=True)
    assert "log-log" in res["report"] and "% error" in res["report"]
    assert (tmp_path / "residuals.png").exists()

    with pytest.raises(ValueError):
        run(tmp_path / "r.csv", tmp_path / "c.csv", tmp_path, method="gd", target="sales", log=True)


def test_holdout_check_percent_errors():
    ranked = pd.DataFrame({"city": ["a", "b"], "population": [1, 2], "predicted_sales": [90.0, 150.0]})
    actuals = pd.DataFrame({"city": ["a", "b"], "actual_sales": [100.0, 100.0]})
    check = holdout_check(ranked, actuals, "sales")
    assert check["error_pct"].tolist() == pytest.approx([-10.0, 50.0])
    text = summarize_holdout(check)
    assert "30.0%" in text and "within 15% = 1/2" in text and "under-predicted = 1/2" in text
    with pytest.raises(ValueError):
        holdout_check(ranked, actuals.rename(columns={"actual_sales": "x"}), "sales")


def test_rank_candidates_works_with_log_model():
    x, y = power_law()
    m = LogLogRegression().fit(x, y)
    ranked = rank_candidates(m, pd.DataFrame({"city": ["s", "l"], "population": [8.0, 80.0]}), "sales")
    assert ranked["city"].tolist() == ["l", "s"] and ranked["recommend"].all()


def test_command_line_help_and_end_to_end(tmp_path, monkeypatch, capsys):
    """--help formats every option's help text (a bare % in one crashes Python 3.14 at startup)."""
    import sys

    from restaurant_expansion.__main__ import main

    monkeypatch.setattr(sys, "argv", ["restaurant_expansion", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "--log" in capsys.readouterr().out

    x, y = power_law(n=60)
    pd.DataFrame({"city": [f"c{i}" for i in range(60)], "population": x, "sales": y}).to_csv(
        tmp_path / "r.csv", index=False)
    pd.DataFrame({"city": ["a", "b"], "population": [12.0, 90.0]}).to_csv(tmp_path / "c.csv", index=False)
    pd.DataFrame({"city": ["a", "b"], "actual_sales": [30000.0, 190000.0]}).to_csv(
        tmp_path / "act.csv", index=False)
    monkeypatch.setattr(sys, "argv", [
        "restaurant_expansion", "--restaurants", str(tmp_path / "r.csv"),
        "--candidates", str(tmp_path / "c.csv"), "--target", "sales", "--evaluate", "--log",
        "--actuals", str(tmp_path / "act.csv"), "--out", str(tmp_path / "out")])
    main()
    out = capsys.readouterr().out
    assert "elasticity" in out and "Held-out cities" in out
    assert (tmp_path / "out" / "holdout_check.csv").exists()


def test_interval_coverage_is_about_95_percent_when_the_model_is_right():
    rng = np.random.default_rng(2)
    x = rng.uniform(5, 100, 900)
    y = 3 + 2 * x + rng.normal(0, 8, 900)
    cov = cv_interval_coverage(x, y, k=5)
    assert 0.92 <= cov["overall"] <= 0.98
    assert all(0.88 <= c <= 1.0 for c in cov["by_size"])
    assert sum(cov["n_by_size"]) == 900


def test_constant_spread_intervals_are_uneven_by_size_and_log_intervals_are_not():
    x, y = power_law(n=900, sigma=0.3, seed=4)
    lin, log = cv_interval_coverage(x, y), cv_interval_coverage(x, y, log=True)
    assert max(lin["by_size"]) - min(lin["by_size"]) > 0.05    # uneven across city sizes
    assert max(log["by_size"]) - min(log["by_size"]) < 0.06    # roughly even
    assert 0.92 <= log["overall"] <= 0.98
    with pytest.raises(ValueError):
        cv_interval_coverage(x, y, k=1)


def test_holdout_check_reports_whether_actual_is_inside_the_interval():
    ranked = pd.DataFrame({"city": ["a", "b"], "population": [1, 2], "predicted_sales": [90.0, 150.0]})
    actuals = pd.DataFrame({"city": ["a", "b"], "actual_sales": [100.0, 100.0]})
    check = holdout_check(ranked, actuals, "sales", lower=[80.0, 110.0], upper=[120.0, 190.0])
    assert check["inside_95"].tolist() == [True, False]
    assert "inside 95% interval = 1/2" in summarize_holdout(check)


def test_evaluation_report_includes_coverage(tmp_path):
    x, y = power_law(n=120)
    train = pd.DataFrame({"city": [f"c{i}" for i in range(120)], "population": x, "sales": y})
    res = run_evaluation(train, tmp_path, target="sales", log=True)
    assert "prediction-interval coverage" in res["report"]
    assert 0.8 <= res["coverage"]["overall"] <= 1.0
