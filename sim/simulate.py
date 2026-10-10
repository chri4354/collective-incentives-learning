"""Simulations for 'Collective incentives that promote collective intelligence'.

Each round, n agents see a shared signal y = theta + eta (sd sigma_s) and a
private signal x_i = theta + eps_i (sd sigma_p), report
a_i = (1 - w_i) y + w_i x_i, and update w_i by the delta rule
    w_i <- clip(w_i - lam * e * (x_i - y), 0, 1)
where e is their own error (individual reward) or the crowd's error
(collective reward). theta is set to 0 without loss of generality.

Runs every condition in the Methods section and writes:
  sim/results.csv      one row per (sigma_p, n, reward): learned weight, errors, convergence
  sim/switch.csv       re-learning after the reward scheme switches halfway
  sim/traces.npz       ensemble-mean weight trajectories, for the figures
"""
import csv
import os

import numpy as np

SIGMA_S = 10.0
SIGMA_PS = [10.0, 20.0, 30.0]
NS = [1, 5, 25, 100]
T = 2000
RUNS = 100
W0 = 0.5
ETA = 0.05            # lam = ETA / (sigma_s^2 + sigma_p^2)
LAST = 500            # rounds averaged for the stationary outcomes
TOL = 0.05            # "within 5% of the target weight"
SWITCH_AT = T // 2
SEED = 20261010

OUT = os.path.dirname(os.path.abspath(__file__))


def w_ind(sp, n):
    return SIGMA_S**2 / (SIGMA_S**2 + sp**2)


def w_col(sp, n):
    return SIGMA_S**2 / (SIGMA_S**2 + sp**2 / n)


def rmse_ind(w, sp):
    return np.sqrt((1 - w) ** 2 * SIGMA_S**2 + w**2 * sp**2)


def rmse_crowd(w, sp, n):
    return np.sqrt((1 - w) ** 2 * SIGMA_S**2 + w**2 * sp**2 / n)


def simulate(sp, n, rewards, rng):
    """rewards: length-T sequence of 'I' / 'C'. Returns per-round arrays over runs."""
    lam = ETA / (SIGMA_S**2 + sp**2)
    W = np.full((RUNS, n), W0)
    mean_w = np.empty((T, RUNS))
    sq_ind = np.empty((T, RUNS))     # squared error of individuals, averaged over agents
    sq_crowd = np.empty((T, RUNS))
    for t in range(T):
        eta = rng.normal(0, SIGMA_S, (RUNS, 1))
        eps = rng.normal(0, sp, (RUNS, n))
        a = (1 - W) * eta + W * eps
        crowd = a.mean(axis=1, keepdims=True)
        e = a if rewards[t] == "I" else crowd
        W = np.clip(W - lam * e * (eps - eta), 0.0, 1.0)
        mean_w[t] = W.mean(axis=1)
        sq_ind[t] = (a**2).mean(axis=1)
        sq_crowd[t] = crowd[:, 0] ** 2
    return mean_w, sq_ind, sq_crowd


def first_within(trace, target, start=0):
    """First round (counted from `start`) at which the ensemble-mean weight is within TOL of target."""
    hit = np.flatnonzero(np.abs(trace[start:] - target) <= TOL * target)
    return int(hit[0]) + 1 if hit.size else None


def main():
    rng = np.random.default_rng(SEED)
    rows, switch_rows, traces = [], [], {}
    for sp in SIGMA_PS:
        for n in NS:
            for reward in ["I", "C"]:
                target = w_ind(sp, n) if reward == "I" else w_col(sp, n)
                mean_w, sq_ind, sq_crowd = simulate(sp, n, [reward] * T, rng)
                ens = mean_w.mean(axis=1)
                traces[f"{reward}_sp{int(sp)}_n{n}"] = ens
                w_run = mean_w[-LAST:].mean(axis=0)          # per-run stationary weight
                rows.append({
                    "sigma_p": sp, "n": n, "reward": reward,
                    "w_ind_opt": w_ind(sp, n), "w_col_opt": w_col(sp, n), "target": target,
                    "w_learned": w_run.mean(), "w_learned_sd": w_run.std(ddof=1),
                    "rmse_ind_sim": np.sqrt(sq_ind[-LAST:].mean()),
                    "rmse_crowd_sim": np.sqrt(sq_crowd[-LAST:].mean()),
                    "rmse_ind_at_target": rmse_ind(target, sp),
                    "rmse_crowd_at_target": rmse_crowd(target, sp, n),
                    "rmse_crowd_best": rmse_crowd(w_col(sp, n), sp, n),
                    "rmse_shared_alone": SIGMA_S,
                    "rounds_to_5pct": first_within(ens, target),
                    "runs_within_5pct": float(np.mean(np.abs(w_run - target) <= TOL * target)),
                })
            for first, second in [("I", "C"), ("C", "I")]:
                schedule = [first] * SWITCH_AT + [second] * (T - SWITCH_AT)
                mean_w, _, _ = simulate(sp, n, schedule, rng)
                ens = mean_w.mean(axis=1)
                traces[f"{first}to{second}_sp{int(sp)}_n{n}"] = ens
                new_target = w_ind(sp, n) if second == "I" else w_col(sp, n)
                switch_rows.append({
                    "sigma_p": sp, "n": n, "switch": f"{first}->{second}",
                    "w_before": mean_w[SWITCH_AT - LAST // 2:SWITCH_AT].mean(),
                    "w_after": mean_w[-LAST:].mean(), "new_target": new_target,
                    "rounds_to_5pct_after_switch": first_within(ens, new_target, SWITCH_AT),
                })
            print(f"done sigma_p={sp:g} n={n}")

    for name, data in [("results.csv", rows), ("switch.csv", switch_rows)]:
        with open(os.path.join(OUT, name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    np.savez_compressed(os.path.join(OUT, "traces.npz"), **traces)


if __name__ == "__main__":
    main()
