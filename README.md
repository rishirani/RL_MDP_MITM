# Learning-Time Scaling Laws for Bellman Deviation Detection of Man-in-the-Middle Attacks in Model-Free Reinforcement Learning

Simulation code and supplementary document for the paper by R. Rani and M. Franceschetti
(UC San Diego), prepared for IEEE Transactions on Control of Networked Systems.

`supplementary.pdf` is the supplementary document. It extends the analysis to delayed and
intermittent attacks, compares Bellman deviation detection (BDD) with model-based detection,
and reports further simulations. The paper is self-contained and does not rely on it.

## Setup

```bash
pip install -r requirements.txt   # numpy, matplotlib, numba, scipy
```

The scripts run in stages. Each simulation stage saves its results (`.npz` / `.npy`, with a
`.log` of the printed numbers), and the `plot` stages read only saved files, so figures can be
regenerated in seconds without rerunning the simulations. The saved results used in the paper
and the supplement are committed. Run times are for a 10-core laptop.

## Paper, Section VII: `theory_validation.py`

Tests each link of the learning-time analysis (Assumption 2 + Lemma 1 => Theorem 8 =>
Theorem 9) on a two-state, two-action Gilbert-Elliott packet-loss channel (gamma = 0.3,
reward noise sigma_w = 0.3), small enough that the certified regime is reachable.

```bash
python3 theory_validation.py calib       # ~30 s: certificate constants and certified ratios
python3 theory_validation.py exp12       # ~2 h: learning-rate and information-advantage runs
python3 theory_validation.py phiV        # seconds: maximum signal over the visitation set (same adversary seeds)
python3 theory_validation.py analyze12   # slopes, level-set slopes, Pr[E] checks
python3 theory_validation.py exp3        # ~45 min: end-to-end detection along the certified line
python3 theory_validation.py single      # ~2 min: single-instance detection runs
python3 theory_validation.py plot        # theory_ia_map.png (Fig. 3), theory_single_instance.png (Fig. 2)
```

Set `TV_QUICK=1` for a small smoke test. In the paper repository the two figures are named
`ia_learning_time_map.png` and `bdd_single_instance.png`.

## Supplement

| Script | Supplement section | Stages | Figures |
|---|---|---|---|
| `dynamic_attacks.py` | IX, intermittent attacks (Theorem 11) | `run` (~1 min), `analyze`, `plot` | `dynamic_attacks.png` (Fig. 4) |
| `model_based_comparison.py` | X, comparison with model-based detection | `grid`, `detect`, `neutral`, `sweep`, `plot` (~5 min in total) | `mb_level_sets.png` (Fig. 5), `mb_value_neutral.png` (Fig. 6), Table I |
| `networked_example.py` | XI, larger networked control example | `damage`, `calib` (~7 min), `visitV` (~1 min), `detect` (~2 min), `plot` | `net_detection.png` (Fig. 7) |

`dynamic_attacks.py` and `model_based_comparison.py` import `theory_validation.py` and use its
saved results (`theory_validation_exp12*.npz`, `theory_validation_iagrid.npy`,
`theory_validation_single.npz`); `dynamic_attacks.py` checks that it reproduces the Fig. 2
instance exactly.
