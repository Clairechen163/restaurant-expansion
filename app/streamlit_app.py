"""Streamlit dashboard: enter a city's population, get the predicted profit.

Run from the project root:
    pip install -r requirements-app.txt
    pip install -e .
    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from restaurant_expansion.analysis import rank_candidates
from restaurant_expansion.model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
POP_UNIT = 10_000     # data stores population in 10,000s of people
PROFIT_UNIT = 10_000  # data stores profit in $10,000s

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
cands = load(ROOT / "data" / "candidate_cities.csv")

model = LinearRegression("normal").fit(train["population"], train["profit"])
r2 = model.score(train["population"], train["profit"])

# ---- Predictor ------------------------------------------------------------
left, right = st.columns([1, 2])
with left:
    st.subheader("Predict a city")
    people = st.number_input("City population (people)", min_value=0, value=120_000, step=5_000)
    pop = people / POP_UNIT
    profit_usd = float(model.predict(pop)) * PROFIT_UNIT
    st.metric("Predicted profit", f"-${-profit_usd:,.0f}" if profit_usd < 0 else f"${profit_usd:,.0f}")
    st.caption("Per year if your profit data is annual; use your own time unit consistently.")
    lo, hi = train["population"].min(), train["population"].max()
    if not lo <= pop <= hi:
        st.warning(f"Outside the existing outlets' range ({lo * POP_UNIT:,.0f} to {hi * POP_UNIT:,.0f} people). "
                   "This is an extrapolation, so treat it with caution.")
    if profit_usd <= 0:
        st.error("Predicted loss: not recommended on population alone.")
    else:
        st.success("Predicted profit is positive.")
    st.write(f"Model: profit = {model.intercept_:.2f} + {model.coef_:.2f} × population   (R² = {r2:.2f})")

with right:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(train["population"] * POP_UNIT, train["profit"] * PROFIT_UNIT,
               marker="x", c="tab:red", label="Existing restaurants")
    xs = np.linspace(0, max(train["population"].max(), pop) * 1.1, 100)
    ax.plot(xs * POP_UNIT, model.predict(xs) * PROFIT_UNIT, label="Fitted line")
    ax.scatter([people], [profit_usd], s=120, c="tab:green", zorder=5, label="Your city")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Population")
    ax.set_ylabel("Profit ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---- Candidate ranking ----------------------------------------------------
st.subheader("Candidate cities, ranked")
ranked = rank_candidates(model, cands)
table = pd.DataFrame({
    "Rank": ranked["rank"],
    "City": ranked["city"],
    "Population": (ranked["population"] * POP_UNIT).round().astype(int),
    "Predicted profit ($)": (ranked["predicted_profit"] * PROFIT_UNIT).round().astype(int),
    "Recommend": ranked["recommend"].map({True: "Yes", False: "No"}),
})
st.dataframe(table, hide_index=True)
