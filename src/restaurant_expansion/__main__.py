import argparse

import pandas as pd

from .analysis import run
from .evaluation import holdout_check, run_evaluation, summarize_holdout


def main():
    p = argparse.ArgumentParser(description="Rank candidate cities for a new restaurant.")
    p.add_argument("--restaurants", default="data/restaurants.csv")
    p.add_argument("--candidates", default="data/candidate_cities.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--method", choices=["normal", "gd"], default="normal")
    p.add_argument("--target", default="profit", help="name of the CSV column to predict (default: profit)")
    p.add_argument("--evaluate", action="store_true",
                   help="also run cross-validation and residual diagnostics")
    p.add_argument("--folds", type=int, default=5, help="number of CV folds (with --evaluate)")
    p.add_argument("--log", action="store_true",
                   help="fit log(target) on log(population): proportional errors, constant spread")
    p.add_argument("--actuals", help="CSV with city and actual_<target> for held-out cities: report %% error")
    a = p.parse_args()

    model, ranked, r2 = run(a.restaurants, a.candidates, a.out, a.method, a.target, a.log)
    if a.log:
        print(f"Model: log({a.target}) = {model.intercept_:.3f} + {model.coef_:.3f} * log(population)   "
              f"(R^2 = {r2:.3f}; elasticity {model.coef_:.3f}: +1% population -> +{model.coef_:.2f}% {a.target})")
    else:
        print(f"Model: {a.target} = {model.intercept_:.3f} + {model.coef_:.3f} * population   (R^2 = {r2:.3f})")
    print(f"Units: population in 10,000s, {a.target} in $10,000s\n")
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved results to {a.out}/")

    if a.actuals:
        check = holdout_check(ranked, pd.read_csv(a.actuals), a.target)
        check.to_csv(f"{a.out}/holdout_check.csv", index=False)
        print("\n" + check.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print(summarize_holdout(check))

    if a.evaluate:
        result = run_evaluation(pd.read_csv(a.restaurants), a.out, k=a.folds, target=a.target, log=a.log)
        print("\n" + result["report"])
        print(f"\nSaved evaluation.txt and residuals.png to {a.out}/")


if __name__ == "__main__":
    main()
