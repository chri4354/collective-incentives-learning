"""Collective reward with and without the cap at w = 1, for the conditions where w^C is near 1.

Writes sim/no_cap.csv: learned weight (population mean over the last 500 rounds,
mean over runs) and the within-group SD of weights, capped vs uncapped.
"""
import csv
import os

import numpy as np

from simulate import ETA, LAST, RUNS, SIGMA_S, T, W0, w_col

CONDITIONS = [(10.0, 25), (10.0, 100), (20.0, 25), (20.0, 100), (30.0, 25), (30.0, 100)]
SEED = 20261011


def run(sp, n, cap, rng):
    lam = ETA / (SIGMA_S**2 + sp**2)
    W = np.full((RUNS, n), W0)
    w_mean, w_sd = [], []
    for t in range(T):
        eta = rng.normal(0, SIGMA_S, (RUNS, 1))
        eps = rng.normal(0, sp, (RUNS, n))
        a = (1 - W) * eta + W * eps
        W = W - lam * a.mean(axis=1, keepdims=True) * (eps - eta)
        if cap:
            W = np.clip(W, 0.0, 1.0)
        if t >= T - LAST:
            w_mean.append(W.mean(axis=1))
            w_sd.append(W.std(axis=1))
    return np.mean(w_mean), np.mean(w_sd)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for sp, n in CONDITIONS:
        for cap in (True, False):
            w, sd = run(sp, n, cap, rng)
            rows.append({"sigma_p": sp, "n": n, "cap": cap, "w_col_opt": w_col(sp, n),
                         "w_learned": w, "within_group_sd": sd})
            print(rows[-1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_cap.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
