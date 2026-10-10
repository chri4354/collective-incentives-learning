"""Figures from the outputs of simulate.py -> ../figures/fig1.pdf, fig2.pdf (+ .png previews)."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate import (LAST, NS, SIGMA_PS, SIGMA_S, ETA, SWITCH_AT, T, TOL, W0,
                      rmse_crowd, rmse_ind, w_col, w_ind)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

IND, COL = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated pair)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
})

res = pd.read_csv(os.path.join(HERE, "results.csv"))
traces = np.load(os.path.join(HERE, "traces.npz"))
nn = np.geomspace(1, 100, 200)


def panel_label(ax, s):
    ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=9, color=INK)


def logx(ax):
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.minorticks_off()


# ---------- Figure 1: what is learned, and what it does to individual and crowd error ----------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=True)
for j, sp in enumerate(SIGMA_PS):
    ax = axes[0, j]
    ax.plot(nn, [w_ind(sp, n) for n in nn], color=IND, ls="--", lw=1)
    ax.plot(nn, [w_col(sp, n) for n in nn], color=COL, ls="--", lw=1)
    for reward, color, label in [("I", IND, "individual reward"), ("C", COL, "collective reward")]:
        d = res[(res.sigma_p == sp) & (res.reward == reward)]
        ax.errorbar(d.n, d.w_learned, yerr=d.w_learned_sd, fmt="o", ms=4.5, color=color,
                    mec="white", mew=0.8, elinewidth=1, capsize=0, label=label, zorder=3)
    ax.set_ylim(0, 1.04)
    ax.set_title(rf"private noise $\sigma_p = {sp:g}$ ($\sigma_s = {SIGMA_S:g}$)", color=INK)
    logx(ax)
    if j == 0:
        ax.set_ylabel("weight on private signal, $w$")
        ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    panel_label(ax, "ABC"[j])

    ax = axes[1, j]
    ax.axhline(SIGMA_S, color=MUTED, ls=":", lw=1)
    ax.text(1.05, SIGMA_S + 0.5, "shared signal alone", color=MUTED, fontsize=6.5, va="bottom")
    for reward, color, wfun in [("I", IND, w_ind), ("C", COL, w_col)]:
        d = res[(res.sigma_p == sp) & (res.reward == reward)]
        ax.plot(nn, [rmse_crowd(wfun(sp, n), sp, n) for n in nn], color=color, lw=1.4)
        ax.plot(nn, [rmse_ind(wfun(sp, n), sp) for n in nn], color=color, lw=1, ls=(0, (1, 1.5)))
        ax.plot(d.n, d.rmse_crowd_sim, "o", ms=4.5, color=color, mec="white", mew=0.8, zorder=3)
        ax.plot(d.n, d.rmse_ind_sim, "o", ms=4.5, mfc="white", mec=color, mew=1, zorder=3)
    ax.set_ylim(0, 29)
    ax.set_xlabel("group size, $n$")
    logx(ax)
    if j == 0:
        ax.set_ylabel("RMS error")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=INK, marker="o", ms=4, mec="white", lw=1.4, label="crowd (mean)"),
                           Line2D([], [], color=INK, marker="o", ms=4, mfc="white", lw=1, ls=(0, (1, 1.5)),
                                  label="individual")],
                  loc="upper left", bbox_to_anchor=(0, 0.93), handletextpad=0.3)
    panel_label(ax, "DEF"[j])
fig.tight_layout(h_pad=1.2, w_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig1.{ext}"), dpi=300)
plt.close(fig)


# ---------- Figure 2: learning dynamics ----------
markers = {10.0: "s", 20.0: "o", 30.0: "^"}


def relaxation_time(trace, target):
    """Rounds for the ensemble-mean weight to close 1 - 1/e of its initial gap to target."""
    gap = abs(W0 - target)
    if gap < 0.02:                      # starts (almost) at the target: nothing to learn
        return None
    hit = np.flatnonzero(np.abs(trace - target) <= gap / np.e)
    return int(hit[0]) + 1 if hit.size else None

SP = 20.0
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
t = np.arange(1, T + 1)

ax = axes[0]                                    # A: trajectories by group size
for n, alpha in zip([5, 25, 100], [0.45, 0.7, 1.0]):
    ax.plot(t, traces[f"C_sp{int(SP)}_n{n}"], color=COL, alpha=alpha, lw=1.1)
    ax.axhline(w_col(SP, n), color=COL, alpha=alpha, ls="--", lw=0.8)
    ax.text(T * 1.01, w_col(SP, n), f"n={n}", color=INK, fontsize=6.5, va="center")
for n in [5, 25, 100]:
    ax.plot(t, traces[f"I_sp{int(SP)}_n{n}"], color=IND, lw=1.1, alpha=0.8)
ax.axhline(w_ind(SP, 1), color=IND, ls="--", lw=0.8)
ax.text(T * 1.01, w_ind(SP, 1), "all n", color=INK, fontsize=6.5, va="center")
ax.set_xlim(0, T)
ax.set_ylim(0, 1.02)
ax.set_xlabel("round")
ax.set_ylabel("mean weight on private signal")
ax.set_title(rf"learning, $\sigma_p={SP:g}$", color=INK)
panel_label(ax, "A")

ax = axes[1]                                    # B: switching the reward scheme halfway
n = 25
for key in ["ItoC", "CtoI"]:
    tr = traces[f"{key}_sp{int(SP)}_n{n}"]
    c1, c2 = (IND, COL) if key == "ItoC" else (COL, IND)
    ax.plot(t[:SWITCH_AT], tr[:SWITCH_AT], color=c1, lw=1.1)
    ax.plot(t[SWITCH_AT - 1:], tr[SWITCH_AT - 1:], color=c2, lw=1.1)
ax.axvline(SWITCH_AT, color=MUTED, lw=0.8)
ax.axhline(w_col(SP, n), color=COL, ls="--", lw=0.8)
ax.axhline(w_ind(SP, n), color=IND, ls="--", lw=0.8)
ax.set_xlim(0, T)
ax.set_ylim(0, 1.02)
ax.set_xlabel("round")
ax.set_title(rf"reward switched, $n={n}$", color=INK)
panel_label(ax, "B")

ax = axes[2]                                    # C: relaxation time vs group size
rows = []
for sp in SIGMA_PS:
    lam = ETA / (SIGMA_S**2 + sp**2)
    for reward, color in [("I", IND), ("C", COL)]:
        k = nn if reward == "C" else np.ones_like(nn)
        ax.plot(nn, 1 / (lam * (SIGMA_S**2 + sp**2 / k)), color=color, lw=1)
        for n_ in NS:
            target = w_col(sp, n_) if reward == "C" else w_ind(sp, n_)
            tau = relaxation_time(traces[f"{reward}_sp{int(sp)}_n{n_}"], target)
            pred = 1 / (lam * (SIGMA_S**2 + sp**2 / (n_ if reward == "C" else 1)))
            rows.append({"sigma_p": sp, "n": n_, "reward": reward, "tau_sim": tau, "tau_pred": pred})
            if tau is not None:
                ax.plot(n_, tau, markers[sp], ms=4.5, color=color, mec="white", mew=0.8, zorder=3)
    ax.text(105, 1 / (lam * (SIGMA_S**2 + sp**2 / 100)), rf"$\sigma_p={sp:g}$", color=INK, fontsize=6.5, va="center")
ax.text(105, 8, "individual\n(all $\\sigma_p$)", color=INK, fontsize=6.5, va="center")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "timescales.csv"), index=False)
logx(ax)
ax.set_xlim(0.85, 100)
ax.set_ylim(0, 200)
ax.set_xlabel("group size, $n$")
ax.set_ylabel("relaxation time (rounds)")
ax.set_title("time to learn", color=INK)
panel_label(ax, "C")

fig.tight_layout(w_pad=2.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig2.{ext}"), dpi=300)
plt.close(fig)
print("figures written to", os.path.normpath(FIG))
