import pandas as pd
import pytest

from restaurant_expansion.census import GEO, build_dataset, split_holdout


def tables(n=60, suppress=(), seed=0):
    """Mock Census API responses in the real wire format: list of lists, strings, header first."""
    import numpy as np

    rng = np.random.default_rng(seed)
    codes = [f"{10000 + i * 20}" for i in range(n)]
    people = rng.uniform(60_000, 2_500_000, n).round()
    sales_k = 0.9 * people / 10 * 1.0 + rng.normal(0, 4_000, n)      # $1,000s; roughly proportional
    sales = [["NAICS2017", "RCPTOT", "RCPTOT_F", "NAICS2017", GEO]]
    pop = [["NAME", "B01003_001E", GEO]]
    for i, c in enumerate(codes):
        val = None if i in suppress else str(int(max(sales_k[i], 1_000)))
        sales.append(["7225", val, None, "7225", c])
        pop.append([f"City{i}, ST Metro Area", str(int(people[i])), c])
    return sales, pop, people


def test_build_dataset_units_and_names():
    sales, pop, people = tables()
    df, rep = build_dataset(sales, pop)
    assert list(df.columns) == ["city", "population", "sales"]
    assert df["city"].str.endswith("Metro Area").sum() == 0          # suffix stripped
    # population converted to 10,000s of people
    assert df["population"].max() == pytest.approx(people.max() / 10_000, rel=1e-3)
    assert rep["joined"] == 60


def test_suppressed_sales_are_dropped_and_counted():
    sales, pop, _ = tables(suppress={3, 7, 11})
    df, rep = build_dataset(sales, pop)
    assert rep["rows_with_published_sales"] == 57
    assert len(df) == 57


def test_population_filter():
    sales, pop, _ = tables()
    df, _ = build_dataset(sales, pop, min_pop=500_000, max_pop=1_000_000)
    assert df["population"].between(50, 100).all()


def test_missing_column_is_reported():
    sales, pop, _ = tables()
    sales[0][1] = "WRONG"
    with pytest.raises(ValueError, match="missing columns"):
        build_dataset(sales, pop)


def test_holdout_split_is_disjoint_and_reproducible():
    sales, pop, _ = tables()
    df, _ = build_dataset(sales, pop)
    tr1, ca1, ac1 = split_holdout(df, 8, seed=1)
    tr2, ca2, _ = split_holdout(df, 8, seed=1)
    assert len(tr1) + len(ca1) == len(df)
    assert not set(tr1["city"]) & set(ca1["city"])
    assert ca1.equals(ca2)
    assert list(ac1.columns) == ["city", "population", "actual_sales"]
    assert "sales" not in ca1.columns                                  # candidates must not leak the answer


def test_end_to_end_with_pipeline(tmp_path):
    from restaurant_expansion.analysis import run
    from restaurant_expansion.evaluation import run_evaluation

    sales, pop, _ = tables()
    df, _ = build_dataset(sales, pop)
    tr, ca, _ = split_holdout(df, 8)
    tr.to_csv(tmp_path / "r.csv", index=False)
    ca.to_csv(tmp_path / "c.csv", index=False)
    model, ranked, r2 = run(tmp_path / "r.csv", tmp_path / "c.csv", tmp_path, target="sales")
    assert "predicted_sales" in ranked.columns and r2 > 0.9
    res = run_evaluation(pd.read_csv(tmp_path / "r.csv"), tmp_path, target="sales")
    assert "sales" in res["report"]


# ---- network layer behaviour (no real network: urlopen is mocked) ----
class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status = body.encode(), status

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_api_requires_a_key():
    from restaurant_expansion import census

    with pytest.raises(RuntimeError, match="free key"):
        census._get("ecnbasic", {"get": "X"}, key=None)


def test_non_json_reply_shows_server_message_and_hides_key(monkeypatch):
    from restaurant_expansion import census

    html = "<html><body><h1>Missing Key</h1> A valid key must be included</body></html>"
    monkeypatch.setattr(census.urllib.request, "urlopen", lambda url, timeout=60: _Resp(html))
    with pytest.raises(RuntimeError) as err:
        census._get("ecnbasic", {"get": "X"}, key="SECRET123")
    msg = str(err.value)
    assert "Missing Key" in msg and "not JSON" in msg
    assert "SECRET123" not in msg and "key=***" in msg


def test_empty_reply_is_explained(monkeypatch):
    from restaurant_expansion import census

    monkeypatch.setattr(census.urllib.request, "urlopen", lambda url, timeout=60: _Resp("", 204))
    with pytest.raises(RuntimeError, match="no data for this query"):
        census._get("ecnbasic", {"get": "X"}, key="k")


def test_valid_json_passes_through(monkeypatch):
    from restaurant_expansion import census

    monkeypatch.setattr(census.urllib.request, "urlopen",
                        lambda url, timeout=60: _Resp('[["A","B"],["1","2"]]'))
    assert census._get("ecnbasic", {"get": "A,B"}, key="k") == [["A", "B"], ["1", "2"]]
