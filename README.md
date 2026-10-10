# Collective incentives that promote collective intelligence

Simulation code for the paper *Collective incentives that promote collective intelligence* (Pescetelli, in preparation).

In each round, a group of *n* agents estimates a quantity. Everyone sees the same shared signal; each agent also sees a private one. Agent *i* reports `(1 − w_i)·shared + w_i·private` and learns its weight `w_i` with the delta rule, from either its own error (individual reward) or the error of the crowd's mean (collective reward). Under individual reward agents learn the individually optimal weight; under collective reward they learn the weight that is optimal for the group, which grows with *n*.

## Run

```bash
pip install -r requirements.txt
cd sim
python simulate.py   # all conditions + halfway reward switch -> results.csv, switch.csv, traces.npz
python no_cap.py     # collective reward with and without the cap at w = 1 -> no_cap.csv
python figures.py    # figures/fig1, figures/fig2 and timescales.csv
```

Everything runs in about 15 seconds. Seeds are fixed, so the outputs in this repository are reproduced exactly.

## Files

| File | What it is |
|---|---|
| `sim/simulate.py` | Model and main simulations: n ∈ {1, 5, 25, 100}, σ_s = 10, σ_p ∈ {10, 20, 30}, individual vs collective reward, 2,000 rounds × 100 runs; plus the reward switched at round 1,000 |
| `sim/no_cap.py` | Check that the small shortfall of learned weights near w = 1 comes from capping weights at 1 (SI Appendix, S4) |
| `sim/figures.py` | Figures 1–2 and relaxation times |
| `sim/results.csv` | Per condition: benchmark weights, learned weight (mean, SD over runs), individual and crowd RMS error, convergence |
| `sim/switch.csv` | Re-learning after the reward switch |
| `sim/timescales.csv` | Simulated vs predicted relaxation times |
| `sim/no_cap.csv` | Learned weights with and without the cap |
| `sim/traces.npz` | Run-averaged weight trajectories used in the figures |
| `figures/` | Figures 1–2 (PDF and PNG) |

Tested with Python 3.10, numpy 1.26, pandas 2.2, matplotlib 3.10.
