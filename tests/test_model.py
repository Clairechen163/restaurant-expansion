import numpy as np
import pandas as pd
import pytest

from restaurant_expansion.analysis import rank_candidates
from restaurant_expansion.model import LinearRegression


def test_normal_equation_recovers_line():
    x = np.linspace(1, 20, 30)
    m = LinearRegression("normal").fit(x, 2 * x - 3)
    assert m.coef_ == pytest.approx(2) and m.intercept_ == pytest.approx(-3)


def test_gd_matches_normal_equation():
    rng = np.random.default_rng(0)
    x = rng.uniform(5, 23, 80)
    y = -3.9 + 1.19 * x + rng.normal(0, 2, 80)
    a = LinearRegression("normal").fit(x, y)
    b = LinearRegression("gd").fit(x, y)
    assert b.coef_ == pytest.approx(a.coef_, abs=1e-3)
    assert b.intercept_ == pytest.approx(a.intercept_, abs=1e-2)
    assert b.cost_history_[-1] <= b.cost_history_[0]


def test_ranking_sorted_descending():
    m = LinearRegression().fit([1, 2, 3, 4], [1, 2, 3, 4])
    r = rank_candidates(m, pd.DataFrame({"city": ["a", "b", "c"], "population": [2, 9, 5]}))
    assert list(r["city"]) == ["b", "c", "a"] and list(r["rank"]) == [1, 2, 3]


def test_bad_input():
    with pytest.raises(ValueError):
        LinearRegression().fit([1], [1])
