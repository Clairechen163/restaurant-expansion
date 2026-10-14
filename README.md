# Restaurant Expansion: Which City Next?

[![Python application](https://github.com/Clairechen163/restaurant-expansion/actions/workflows/python-app.yml/badge.svg)](https://github.com/Clairechen163/restaurant-expansion/actions/workflows/python-app.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://restaurant-expansion.streamlit.app)

<img src="docs/fit.png" width="600" alt="Profit vs. city population with fitted line, 95% prediction interval and candidate cities">

Suppose you are the CEO of a restaurant franchise deciding where to open a new outlet. You have **profit and population** for cities that already have a restaurant, and only **population** for candidate cities. This project fits a linear regression (`profit ≈ θ₀ + θ₁ · population`) on existing outlets and uses it to **rank candidate cities by predicted profit**.

> ⚠️ `data/` contains **synthetic sample data** so the project runs out of the box. Replace it with your real data (same column names) before drawing conclusions. Results on real U.S. Census data are shown [further down](#results-on-real-data-us-census).

## Quick start

```bash
git clone https://github.com/Clairechen163/restaurant-expansion.git
cd restaurant-expansion
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
pip install -e .

python -m restaurant_expansion                  # closed-form (normal equation)
python -m restaurant_expansion --method gd      # gradient descent + convergence plot
python -m restaurant_expansion --evaluate       # cross-validation + residual diagnostics
pytest                                          # run tests

pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py    # interactive dashboard
```

Outputs go to `outputs/`: `ranked_candidates.csv`, `fit.png`, `convergence.png` (gd only), and with `--evaluate`, `evaluation.txt` and `residuals.png`.

Options: `--target COLUMN` (what to predict, default `profit`), `--log` (log-log model, see below), `--actuals FILE` (compare predictions with known values for held-out cities), `--folds K`.

### Real data (U.S. Census)

Get a free Census API key at <https://api.census.gov/data/key_signup.html> and keep it in an environment variable. **Never put the key in a file or commit it.**

```powershell
# Windows PowerShell
$env:CENSUS_API_KEY = "YOUR_KEY_HERE"
python -m restaurant_expansion.census --out data/real --naics 7225
python -m restaurant_expansion `
  --restaurants data/real/restaurants.csv `
  --candidates data/real/candidate_cities.csv `
  --target sales --evaluate `
  --actuals data/real/holdout_actuals.csv `
  --out outputs/real

# same, with the log-log model
python -m restaurant_expansion `
  --restaurants data/real/restaurants.csv `
  --candidates data/real/candidate_cities.csv `
  --target sales --evaluate --log `
  --actuals data/real/holdout_actuals.csv `
  --out outputs/real_log
```

On Mac/Linux use `export CENSUS_API_KEY=YOUR_KEY_HERE` and `\` instead of the backtick at the end of each line. `data/real/` and `outputs/` are git-ignored.

## Data format

| File | Columns |
|---|---|
| `data/restaurants.csv` | `city`, `population`, `profit` |
| `data/candidate_cities.csv` | `city`, `population` |

Units: population in **10,000s**, profit in **$10,000s** (e.g. `population=7.2` = 72,000 people, `profit=4.0` = $40,000). Use your own units consistently.

## Method

- **Model:** univariate linear regression, written from scratch in NumPy (`src/restaurant_expansion/model.py`). A log-log variant (`LogLogRegression`, option `--log`) fits `log(y)` on `log(x)`, a power law.
- **Solvers:** normal equation and batch gradient descent (with feature standardization). Tests check they agree.
- **Decision rule:** rank candidates by predicted profit; flag `recommend = True` where predicted profit > 0.

## Project layout

```
app/         streamlit_app.py (interactive dashboard)
data/        sample CSVs
docs/        fit.png, residuals.png, real_fit.png, real_residuals.png, real_log_residuals.png
scripts/     make_sample_data.py (regenerates the synthetic data)
src/restaurant_expansion/
  model.py       LinearRegression, LogLogRegression
  evaluation.py  cross-validation, prediction intervals, coverage, diagnostics
  census.py      download and prepare real Census data
  analysis.py    load, rank, plot
  __main__.py    CLI
tests/       pytest suite
```

## Evaluation

```
python -m restaurant_expansion --evaluate
```

- **Cross-validation:** 5-fold CV on out-of-fold predictions, compared with a baseline that always predicts the mean profit.
- **Prediction intervals:** 95% intervals for a single new city (the shaded band in the plot above).
- **Interval coverage:** how often the 95% interval really contains the true value on held-out folds, overall and by city size.
- **Diagnostics:** residual vs. fitted, histogram and Q-Q plot, Shapiro-Wilk normality test, and a check for spread changing with city size.
- **Validation:** tests confirm the from-scratch model matches scikit-learn, and that the intervals cover about 95% of cases in simulation.

On the bundled **synthetic** data, cross-validated RMSE is about 2.9 (in $10,000s) against 4.6 for the baseline. The data is generated with noise of 2.8, so the error matches what the data allows. These numbers show the pipeline works. They are not findings about real restaurants.

![Residual diagnostics](docs/residuals.png)

## Results on real data (U.S. Census)

Data: 2017 Economic Census sales for **NAICS 7225 (Restaurants and Other Eating Places)**, joined with 2017 ACS 5-year population by metropolitan/micropolitan statistical area. 835 of 933 areas had published sales; 549 had 50,000 to 3,000,000 residents; 541 were used for fitting and 8 were held out. The target is **sales (revenue)**, not profit, because the Census does not publish profit by metro area. Prepared with `python -m restaurant_expansion.census`.

Fitted line: sales ≈ −$63M + $20M per 10,000 residents (about **$2,000 of restaurant sales per resident per year**).

| Metric (5-fold cross-validation) | Model | Predict-the-mean baseline |
|---|---|---|
| RMSE | $190.7M | $952.4M |
| MAE | $94.6M | n/a |
| Out-of-fold R² | 0.96 | n/a |

- **Error cut by 80%** versus the baseline.
- **R² flatters the model.** Sales scale almost proportionally with population, so R² is dominated by the gap between big and small metros. On the 8 held-out metros the typical error is **22.6%** (3 of 8 within 15%, 6 of 8 within 30%), and 7 of 8 were under-predicted (median −23.5%). With only 8 metros this is indicative, not conclusive.
- **Population alone is not enough.** In these 8 metros, sales per resident ranges from about $1,300 (Portsmouth, OH) to about $2,900 (Boulder, CO), while the model assumes a single slope.
- **Assumption check fails, and I report it.** Residual spread grows with metro size (Spearman ρ = +0.60, p < 0.001) and residuals are not normal (Shapiro-Wilk p < 0.001). The constant-variance 95% prediction intervals are therefore unreliable here (see the coverage results below).
- **Next step:** I refit on a log-log scale to address this; the comparison follows the plots.

![Sales vs. population on real Census data](docs/real_fit.png)

![Residual diagnostics on real Census data](docs/real_residuals.png)

### Linear vs. log-log

The linear model's residual spread grows with metro size, so I also fitted `log(sales)` on `log(population)` (`--log`), which models proportional error. Same 541 metros, same 8 held-out:

| | Linear | Log-log |
|---|---|---|
| CV RMSE | $190.7M | $192.7M |
| CV out-of-fold R² | 0.960 | 0.959 |
| CV median abs % error | 18.0% | 15.4% |
| CV mean abs % error | 21.8% | 21.9% |
| Held-out mean abs % error (n = 8) | 22.6% | 14.1% |
| Held-out within 15% / within 30% | 3/8 / 6/8 | 4/8 / 7/8 |
| \|residual\| vs. size, Spearman ρ | +0.60 | −0.21 |
| 95% interval coverage, overall (CV) | 95.4% | 93.7% |
| ...small / medium / large metros | 99.4% / 98.9% / 87.8% | 87.2% / 95.0% / 98.9% |
| Median 95% interval width, relative to true value | 328% | 120% |
| Held-out inside 95% interval (n = 8) | 7/8 | 7/8 |

- **Point accuracy:** the log model lowers the *typical* percentage error but not RMSE or mean percentage error, and residuals are still not normal (Shapiro-Wilk p < 0.001).
- **Intervals:** the linear model's overall 95.4% coverage is misleading. It is too wide for small and medium metros (99%) and too narrow for large ones (87.8%), and its intervals are so wide (median 328% of the true value) that they can include negative sales. The log model's intervals are about 2.7 times tighter and always positive, but slightly under-cover small metros (87.2%), consistent with its leftover ρ = −0.21.
- Estimated elasticity is 1.11: 1% more population goes with about 1.11% more restaurant sales.
- Held-out results use only 8 metros, so they are indicative, not conclusive. Boulder, CO is under-predicted by 41% by the log model; population alone does not explain it.

![Log-log residual diagnostics](docs/real_log_residuals.png)

*Source: U.S. Census Bureau, 2017 Economic Census (`ecnbasic` API) and 2017 American Community Survey 5-Year Estimates (Table B01003), accessed 14 October 2026. This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau. The raw data is not stored in this repository; the script above downloads it.*

## Source and attribution

The problem framing (predict restaurant profit from city population) follows a classic regression exercise from introductory machine-learning courses. The implementation, evaluation, tests, CI and dashboard are my own. The bundled sample data in `data/` is synthetic (generated by `scripts/make_sample_data.py`); the real-data results above use U.S. Census Bureau data, downloaded by `python -m restaurant_expansion.census`.

## Caveats and next steps

- Population alone is a weak predictor on the synthetic sample (R² ≈ 0.6). On real Census data it explains most of the variation in total sales (R² 0.96), but the typical error for an individual metro is still about 15-23%, depending on the model (see above). Real decisions should also consider rent, competition, income, foot traffic, and demographics.
- The real-data target is restaurant **sales (revenue)**, not profit, and the data is from the 2017 Economic Census.
- Predictions for populations outside the range seen in existing cities (extrapolation) are unreliable. Check candidate cities whose population is larger than any existing outlet.
- Prediction intervals are approximate: neither model's residuals are normal, and coverage varies with city size (see above).
- Held-out checks use only 8 metros, so they are indicative, not conclusive.

## License

MIT
