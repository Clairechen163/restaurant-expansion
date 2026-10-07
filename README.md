# Restaurant Expansion: Which City Next?
[![Python application](https://github.com/Clairechen163/restaurant-expansion/actions/workflows/python-app.yml/badge.svg)](https://github.com/Clairechen163/restaurant-expansion/actions/workflows/python-app.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Suppose you are the CEO of a restaurant franchise deciding where to open a new outlet.
You have **profit and population** for cities that already have a restaurant, and only
**population** for candidate cities. This project fits a linear regression
(`profit ≈ θ₀ + θ₁ · population`) on existing outlets and uses it to **rank candidate cities
by predicted profit**.

> ⚠️ `data/` contains **synthetic sample data** so the project runs out of the box.
> Replace it with your real data (same column names) before drawing conclusions.

## Quick start

```bash
git clone https://github.com/Clairechen163/restaurant-expansion.git
cd restaurant-expansion
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
pip install -e .

python -m restaurant_expansion              # closed-form (normal equation)
python -m restaurant_expansion --method gd  # gradient descent + convergence plot
pytest                                      # run tests
```

Outputs go to `outputs/`: `ranked_candidates.csv`, `fit.png`, and `convergence.png` (gd only).

## Data format

| File | Columns |
|------|---------|
| `data/restaurants.csv` | `city, population, profit` |
| `data/candidate_cities.csv` | `city, population` |

Units: population in **10,000s**, profit in **$10,000s** (e.g. `population=7.2` = 72,000 people,
`profit=4.0` = $40,000). Use your own units consistently.

## Method

- **Model:** univariate linear regression, written from scratch in NumPy (`src/restaurant_expansion/model.py`).
- **Solvers:** normal equation and batch gradient descent (with feature standardization). Tests check they agree.
- **Decision rule:** rank candidates by predicted profit; flag `recommend = True` where predicted profit > 0.

## Project layout

```
data/        sample CSVs
scripts/     make_sample_data.py (regenerates the synthetic data)
src/restaurant_expansion/
  model.py     LinearRegression
  analysis.py  load, rank, plot
  __main__.py  CLI
tests/       pytest suite
```

## Caveats and next steps

- Population alone is a weak predictor (sample R² ≈ 0.6). Real decisions should also consider
  rent, competition, income, foot traffic, and demographics.
- Predictions for populations outside the range seen in existing cities (extrapolation) are unreliable.
  Check `Metro City`-style outliers manually.
- Ideas: multivariate regression, prediction intervals, cross-validation, a Streamlit dashboard.

## License

MIT
