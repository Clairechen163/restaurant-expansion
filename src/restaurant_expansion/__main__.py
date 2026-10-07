import argparse

from .analysis import run


def main():
    p = argparse.ArgumentParser(description="Rank candidate cities for a new restaurant.")
    p.add_argument("--restaurants", default="data/restaurants.csv")
    p.add_argument("--candidates", default="data/candidate_cities.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--method", choices=["normal", "gd"], default="normal")
    a = p.parse_args()

    model, ranked, r2 = run(a.restaurants, a.candidates, a.out, a.method)
    print(f"Model: profit = {model.intercept_:.3f} + {model.coef_:.3f} * population   (R^2 = {r2:.3f})")
    print("Units: population in 10,000s, profit in $10,000s\n")
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()
