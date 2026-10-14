"""Build a real-data version of the project from public U.S. Census Bureau tables.

Data used (both are U.S. Census Bureau products; see the README for attribution):
  * 2017 Economic Census (API dataset ``ecnbasic``): restaurant SALES per metro area,
    variable RCPTOT = "Sales, value of shipments, or revenue ($1,000)".
  * 2017 American Community Survey 5-year estimates (``acs/acs5``): metro population,
    variable B01003_001E = total population.

NOTE: the Census Bureau does not publish PROFIT for individual metro areas, so the
real-data analysis predicts restaurant *sales* (revenue) from population.

Usage (needs internet and a FREE Census API key, https://api.census.gov/data/key_signup.html):
    python -m restaurant_expansion.census --probe --key YOUR_KEY          # which NAICS codes exist?
    python -m restaurant_expansion.census --out data/real --naics 7225 --key YOUR_KEY
    python -m restaurant_expansion --restaurants data/real/restaurants.csv \\
        --candidates data/real/candidate_cities.csv --target sales --evaluate --out outputs/real
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://api.census.gov/data/2017"
GEO = "metropolitan statistical area/micropolitan statistical area"
DISCLAIMER = "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."


# ---------------------------------------------------------------- network layer
KEY_URL = "https://api.census.gov/data/key_signup.html"


def _get(dataset: str, params: dict, key: str | None = None) -> list[list[str]]:
    """Call the Census API. Returns the raw JSON table (first row = column names).

    The API requires a (free) key. Non-JSON replies (for example the 'Missing Key' page)
    are reported with the server's own message instead of a JSON parsing error.
    """
    if not key:
        raise RuntimeError(
            "The Census API needs a free key. Request one at "
            f"{KEY_URL} (arrives by email in a minute), then run with --key YOUR_KEY "
            "or set the CENSUS_API_KEY environment variable."
        )
    q = dict(params)
    q["key"] = key
    url = f"{BASE}/{dataset}?{urllib.parse.urlencode(q, quote_via=urllib.parse.quote, safe=':*/,')}"
    shown = url.replace(key, "***")                      # never print the key
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
    except Exception as exc:                              # network error or HTTP 4xx/5xx
        detail = ""
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
        raise RuntimeError(f"Census API request failed: {exc}\n{detail}\nURL: {shown}") from None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        text = " ".join(body.split())[:300] or "(empty response: the API has no data for this query)"
        raise RuntimeError(f"Census API returned something that is not JSON (HTTP {status}):\n  {text}\nURL: {shown}") from None


def fetch_sales(naics: str, key: str | None = None) -> list[list[str]]:
    return _get("ecnbasic", {"get": "NAICS2017,RCPTOT,RCPTOT_F", "for": f"{GEO}:*", "NAICS2017": naics}, key)


def probe_naics(msa_code: str = "12580", key: str | None = None) -> pd.DataFrame:
    """List the NAICS codes (and their sales) the API serves for one metro area (default: Baltimore)."""
    t = _get("ecnbasic", {"get": "NAICS2017,NAICS2017_LABEL,RCPTOT", "for": f"{GEO}:{msa_code}"}, key)
    return _to_frame(t)[["NAICS2017", "NAICS2017_LABEL", "RCPTOT"]].drop_duplicates()


def fetch_population(key: str | None = None) -> list[list[str]]:
    return _get("acs/acs5", {"get": "NAME,B01003_001E", "for": f"{GEO}:*"}, key)


# ------------------------------------------------------------ pure data shaping
def _to_frame(table: list[list[str]]) -> pd.DataFrame:
    if not table or len(table) < 2:
        raise ValueError("API returned no data rows")
    return pd.DataFrame(table[1:], columns=table[0])


def build_dataset(sales_table, pop_table, min_pop: int = 50_000, max_pop: int = 3_000_000):
    """Join sales and population by metro-area code and convert to the project's units.

    Returns (frame, report). ``frame`` has columns city, population (10,000s of people),
    sales ($10,000s). Metro areas with suppressed/missing sales are dropped and counted.
    """
    sales = _to_frame(sales_table)
    pop = _to_frame(pop_table)
    for df, need in ((sales, {"RCPTOT", GEO}), (pop, {"NAME", "B01003_001E", GEO})):
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"API response is missing columns: {sorted(missing)}")

    sales["sales_k"] = pd.to_numeric(sales["RCPTOT"], errors="coerce")
    pop["people"] = pd.to_numeric(pop["B01003_001E"], errors="coerce")
    n_sales_rows, n_pop_rows = len(sales), len(pop)

    sales = sales.dropna(subset=["sales_k"])
    sales = sales[sales["sales_k"] > 0].drop_duplicates(GEO)
    n_with_sales = len(sales)

    df = pop[[GEO, "NAME", "people"]].merge(sales[[GEO, "sales_k"]], on=GEO, how="inner")
    n_joined = len(df)
    df = df[(df["people"] >= min_pop) & (df["people"] <= max_pop)]

    out = pd.DataFrame({
        "city": df["NAME"].str.replace(r"\s+(Metro|Micro) Area$", "", regex=True).str.strip(),
        "population": (df["people"] / 10_000).round(4),
        "sales": (df["sales_k"] / 10).round(3),   # $1,000 -> $10,000s
    }).sort_values("population").reset_index(drop=True)

    report = {
        "sales_rows": n_sales_rows,
        "population_rows": n_pop_rows,
        "rows_with_published_sales": n_with_sales,
        "joined": n_joined,
        "kept_after_population_filter": len(out),
        "min_pop": min_pop,
        "max_pop": max_pop,
    }
    return out, report


def split_holdout(frame: pd.DataFrame, n_holdout: int = 8, seed: int = 0):
    """Hold out some real metro areas to play the role of 'candidate cities'.

    Their true sales are known, so predictions can be checked against reality.
    Returns (train, candidates, actuals).
    """
    if not 0 < n_holdout < len(frame) - 10:
        raise ValueError("n_holdout must leave at least 10 metro areas for fitting")
    idx = np.random.default_rng(seed).permutation(len(frame))
    hold = frame.iloc[np.sort(idx[:n_holdout])]
    train = frame.drop(hold.index).reset_index(drop=True)
    cand = hold[["city", "population"]].reset_index(drop=True)
    actual = hold[["city", "population", "sales"]].rename(columns={"sales": "actual_sales"}).reset_index(drop=True)
    return train, cand, actual


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Prepare real Census data (restaurant sales vs. metro population).")
    p.add_argument("--out", default="data/real")
    p.add_argument("--naics", default="7225",
                   help="NAICS2017 code: 7225 = restaurants & other eating places (try 722 or 722511 if empty)")
    p.add_argument("--key", default=os.environ.get("CENSUS_API_KEY"), help="Census API key (required for API use; or env CENSUS_API_KEY)")
    p.add_argument("--min-pop", type=int, default=50_000)
    p.add_argument("--max-pop", type=int, default=3_000_000,
                   help="drop the very largest metros, which dominate a linear fit")
    p.add_argument("--holdout", type=int, default=8, help="metro areas held out as 'candidate cities'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", action="store_true",
                   help="just list the NAICS codes available for one metro area, to choose --naics")
    p.add_argument("--sales-csv", help="use a local CSV (columns incl. RCPTOT and the metro code) instead of the API")
    p.add_argument("--pop-csv", help="use a local CSV (columns incl. NAME, B01003_001E and the metro code)")
    a = p.parse_args(argv)

    if a.probe:
        try:
            print(probe_naics(key=a.key).to_string(index=False))
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        return

    if a.sales_csv and a.pop_csv:
        sales_t = pd.read_csv(a.sales_csv, dtype=str).pipe(lambda d: [list(d.columns)] + d.values.tolist())
        pop_t = pd.read_csv(a.pop_csv, dtype=str).pipe(lambda d: [list(d.columns)] + d.values.tolist())
    else:
        try:
            sales_t, pop_t = fetch_sales(a.naics, a.key), fetch_population(a.key)
        except RuntimeError as exc:
            raise SystemExit(str(exc))

    frame, rep = build_dataset(sales_t, pop_t, a.min_pop, a.max_pop)
    for k, v in rep.items():
        print(f"{k:32s} {v}")
    if len(frame) < 30:
        raise SystemExit("Fewer than 30 usable metro areas: try --naics 722 or a different code "
                         "(many detailed cells are suppressed to protect confidentiality).")

    train, cand, actual = split_holdout(frame, a.holdout, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "restaurants.csv", index=False)
    cand.to_csv(out / "candidate_cities.csv", index=False)
    actual.to_csv(out / "holdout_actuals.csv", index=False)
    print(f"\nWrote {len(train)} training metros, {len(cand)} held-out candidates to {out}/")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
