"""Generate SYNTHETIC sample data. Replace the CSVs in data/ with your real data."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 60
pop = np.clip(rng.gamma(shape=3.0, scale=2.2, size=n) + 5, 5, 23)
profit = -3.9 + 1.19 * pop + rng.normal(0, 2.8, n)
pd.DataFrame({"city": [f"Existing-{i+1:02d}" for i in range(n)],
              "population": pop.round(3), "profit": profit.round(3)}).to_csv("data/restaurants.csv", index=False)

cands = {"Riverton": 3.5, "Lakeside": 7.2, "Hillcrest": 11.0, "Port Avery": 14.8,
         "Northgate": 18.9, "Maple Falls": 5.4, "Eastbrook": 9.6, "Metro City": 24.5}
pd.DataFrame({"city": list(cands), "population": list(cands.values())}).to_csv("data/candidate_cities.csv", index=False)
print("wrote data/restaurants.csv and data/candidate_cities.csv")
