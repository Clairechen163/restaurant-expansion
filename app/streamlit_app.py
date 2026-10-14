"""Streamlit dashboard: enter a city's population, get the predicted profit.

Run from the project root:
    pip install -r requirements-app.txt
    pip install -e .
    python -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets Streamlit Cloud import the package without installing it

from restaurant_expansion.analysis import rank_candidates  # noqa: E402
from restaurant_expansion.evaluation import kfold_cv, prediction_interval  # noqa: E402
from restaurant_expansion.model import LinearRegression  # noqa: E402

POP_UNIT = 10_000     # data stores population in 10,000s of people
PROFIT_UNIT = 10_000  # data stores profit in $10,000s


def money(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


st.set_page_config(page_title="Restaurant Expansion", page_icon="🍽️", layout="wide")
st.title("🍽️ Which city next?")
st.caption("Linear regression of restaurant profit on city population.")


@st.cache_data
def load(path):
    return pd.read_csv(path)


# ---- Data -----------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload your own restaurants.csv (city, population, profit)", type="csv")
    st.caption("Without an upload, the bundled **synthetic** sample data is used.")

train = pd.read_csv(upload) if upload else load(ROOT / "data" / "restaurants.csv")
if not {"population", "profit"} <= set(train.columns):
    st.error("CSV needs the columns: city, population, profit (population in 10,000s, profit in $10,000s).")
    st.stop()
if len(train) < 10:
    st.error("Need at least 10 restaurants to fit and evaluate the model.")
    st.stop()
cands = load(ROOT / "data" / "candidate_cities.csv")

model = LinearRegression("normal").fit(train["population"], train["profit"])
r2 = model.score(train["population"], train["profit"])
cv = kfold_cv(train["population"], train["profit"], k=5)

# ---- Predictor ------------------------------------------------------------
left, right = st.columns([1, 2])
with left:
    st.subheader("Predict a city")
    people = st.number_input("City population (people)", min_value=0, value=120_000, step=5_000)
    pop = people / POP_UNIT
    profit = float(model.predict(pop))
    lo, hi = prediction_interval(model, train["population"], train["profit"], pop)
    st.metric("Predicted profit", money(profit * PROFIT_UNIT))
    st.write(f"**95% prediction interval:** {money(float(lo) * PROFIT_UNIT)} to {money(float(hi) * PROFIT_UNIT)}")
    st.caption("Where a single new city's profit is expected to fall 95% of the time, "
               "assuming the linear model holds. Use your own time unit consistently.")
    pmin, pmax = train["population"].min(), train["population"].max()
    if not pmin <= pop <= pmax:
        st.warning(f"Outside the existing outlets' range ({pmin * POP_UNIT:,.0f} to {pmax * POP_UNIT:,.0f} people). "
                   "This is an extrapolation, so treat it with caution.")
    if profit <= 0:
        st.error("Predicted loss: not recommended on population alone.")
    elif float(lo) <= 0:
        st.info("Predicted profit is positive, but the interval includes losses: the outcome is uncertain.")
    else:
        st.success("Predicted profit is positive, and so is the whole interval.")

    with st.expander("Model quality"):
        st.write(f"profit = {model.intercept_:.2f} + {model.coef_:.2f} × population (units: 10,000s)")
        st.write(f"In-sample R² = {r2:.2f}")
        st.write(f"{cv['k']}-fold cross-validated RMSE = {money(cv['rmse'] * PROFIT_UNIT)} "
                 f"vs. {money(cv['baseline_rmse'] * PROFIT_UNIT)} for always predicting the mean.")
        st.caption("RMSE is the typical size of a prediction error on cities the model has not seen.")

with right:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(train["population"] * POP_UNIT, train["profit"] * PROFIT_UNIT,
               marker="x", c="tab:red", label="Existing restaurants")
    xs = np.linspace(0, max(train["population"].max(), pop) * 1.1, 100)
    band_lo, band_hi = prediction_interval(model, train["population"], train["profit"], xs)
    ax.fill_between(xs * POP_UNIT, band_lo * PROFIT_UNIT, band_hi * PROFIT_UNIT,
                    color="tab:blue", alpha=0.12, label="95% prediction interval")
    ax.plot(xs * POP_UNIT, model.predict(xs) * PROFIT_UNIT, label="Fitted line")
    ax.scatter([people], [profit * PROFIT_UNIT], s=120, c="tab:green", zorder=5, label="Your city")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Population")
    ax.set_ylabel("Profit ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---- Candidate ranking ----------------------------------------------------
st.subheader("Candidate cities, ranked")
ranked = rank_candidates(model, cands)
c_lo, c_hi = prediction_interval(model, train["population"], train["profit"], ranked["population"].to_numpy())
table = pd.DataFrame({
    "Rank": ranked["rank"],
    "City": ranked["city"],
    "Population": (ranked["population"] * POP_UNIT).round().astype(int),
    "Predicted profit ($)": (ranked["predicted_profit"] * PROFIT_UNIT).round().astype(int),
    "95% interval low ($)": (c_lo * PROFIT_UNIT).round().astype(int),
    "95% interval high ($)": (c_hi * PROFIT_UNIT).round().astype(int),
    "Recommend": ranked["recommend"].map({True: "Yes", False: "No"}),
})
st.dataframe(table, hide_index=True)
