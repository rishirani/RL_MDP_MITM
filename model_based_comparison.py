"""Model-free (BDD, Algorithm 1) versus model-based detection (supplement, Section X).

Model-based (MB) detector: the agent keeps the transition counts of its t_agt training samples
(estimate p_bar of P, N^2 M numbers instead of the N M entries of Q_hat). During detection it
forms the empirical next-state frequencies q_hat_{i,j} of each retained pair (pruning as in
Algorithm 1) and declares a breach if  max_{retained} ||q_hat_{i,j} - p_bar_{i,j}||_1 > xi_MB.
With xi_MB = eps_bar_P > max ||p_bar - p||_1, the argument of Theorem 5 gives detection w.h.p.
as t_det -> inf whenever  psi_V := max_{V} ||p_hat_{i,j} - p_{i,j}||_1 > eps_P + eps_bar_P (V = all pairs here).

Stages (run in this order; results are saved, `plot` only reads saved files):
  python3 model_based_comparison.py grid     # MB information-advantage event on the Section VII grid
  python3 model_based_comparison.py detect   # FA / miss at certified operating points
  python3 model_based_comparison.py neutral  # 3-state channel, value-neutral attack (Phi = 0, P_hat != P)
  python3 model_based_comparison.py sweep    # miss rates at a common false-alarm level (Table I)
  python3 model_based_comparison.py plot
"""
import sys
import time

import numpy as np
from numba import njit, prange

import theory_validation as tv

N, M, GAMMA = tv.N, tv.M, tv.GAMMA


# ---------------------------------------------------------------- model-based agent
def mb_agent_errors(n_runs, checkpoints, seed):
    """max_{i,j} ||p_bar_{i,j} - p_{i,j}||_1 for the synchronous generative sampling of Section VII:
    every pair is sampled once per iteration, so a pair has floor(t / NM) samples at sample count t."""
    rng = np.random.default_rng(seed)
    n_pair = np.maximum(checkpoints // (N * M), 1)
    inc = np.diff(np.concatenate([[0], n_pair]))
    good = np.zeros((n_runs, N, M))
    out = np.zeros((n_runs, len(checkpoints)))
    for k, dn in enumerate(inc):
        good += rng.binomial(dn, tv.P_GOOD, size=(n_runs, N, M))
        pbar = good / n_pair[k]
        out[:, k] = (2 * np.abs(pbar - tv.P_GOOD)).reshape(n_runs, -1).max(1)
    return out


def psi_from_counts(cnt):
    """||p_hat_{i,j} - p_{i,j}||_1, shape (..., N, M)."""
    return np.abs(tv.P - tv.p_hat_from_counts(cnt)).sum(-1)


def certificates_mb(cal_err, cal_psi, ta, tvv, delta):
    qa = np.sqrt(ta) * np.quantile(cal_err, 1 - delta / 2, axis=0)
    qv = np.sqrt(tvv) * np.quantile(cal_psi, delta / 2, axis=0)
    B = qa[ta >= 1e3].max()
    A = qv[tvv >= tv.T_ADV_MIN].min()
    return B, A, (2 * B / A) ** 2


def grid_prob(sig, err, k):
    G = np.zeros((sig.shape[1], err.shape[1]))
    for i in range(sig.shape[1]):
        for j in range(err.shape[1]):
            G[i, j] = tv.prob_ia_real(sig[:, i], err[:, j], k)
    return G


# ---------------------------------------------------------------- stage: grid
def grid():
    Qs, _, C = tv.bellman_targets(GAMMA)
    d = dict(np.load("theory_validation_exp12.npz"))
    ta, tvv, ta_cal = d["t_agt"], d["t_adv"], d["cal_t_agt"]
    n_cal_adv, n_eva_adv = d["cal_phimin"].shape[0], d["phimin"].shape[0]
    # same seeds as theory_validation.exp12 -> identical adversary runs (paired comparison)
    cnt_cal = tv.run_adversary(n_cal_adv, tvv, 2 * 10**7)
    cnt_eva = tv.run_adversary(n_eva_adv, tvv, 3 * 10**7)
    phimin_eva = np.abs(tv.phi_from_counts(cnt_eva, C)).reshape(n_eva_adv, len(tvv), -1).min(-1)
    print("paired with Section VII adversary runs:", np.allclose(phimin_eva, d["phimin"]))
    res = dict(t_agt=ta, t_adv=tvv, cal_t_agt=ta_cal)
    res["cal_psi"] = psi_from_counts(cnt_cal).reshape(n_cal_adv, len(tvv), -1).max(-1)   # psi_V, V = all pairs
    res["psi"] = psi_from_counts(cnt_eva).reshape(n_eva_adv, len(tvv), -1).max(-1)
    res["cal_err"] = mb_agent_errors(d["cal_err"].shape[0], ta_cal, 11)
    res["err"] = mb_agent_errors(d["err"].shape[0], ta, 12)
    res["G_mb"] = grid_prob(res["psi"], res["err"], 2.0)
    res["G_bdd"] = np.load("theory_validation_iagrid.npy")
    np.savez("mb_grid.npz", **res)

    print("=== MB rates (slope of log quantile vs log t)")
    for lev, name, X, t in [(0.75, "agent q_0.75(max||pbar-p||_1)", res["err"], ta),
                            (0.25, "adv   q_0.25(psi_V)", res["psi"], tvv)]:
        s, ci = tv.boot_slope(t, X, lev, 1e3 if X is res["err"] else tv.T_ADV_MIN)
        print(f"  {name}: slope {s:.4f} CI [{ci[0]:.4f},{ci[1]:.4f}]")
    for lev in (0.5, 0.8, 0.9):
        for name, G in (("BDD", res["G_bdd"]), ("MB ", res["G_mb"])):
            x, y = tv.contour_points(G, ta, tvv, lev)
            keep = (x >= tv.T_ADV_MIN) & (x <= 1e5); x, y = x[keep], y[keep]
            if len(x) > 2:
                sl = np.polyfit(np.log(x), np.log(y), 1)[0]
                print(f"  {name} Pr = {lev}: contour slope {sl:.3f}; t_agt/t_adv in [{(y/x).min():.3g},{(y/x).max():.3g}]")
    cal_bdd = dict(np.load("theory_validation_exp12_partial.npz"))
    for delta in tv.DELTAS:
        Bp, Ap, Cp = certificates_mb(res["cal_err"], res["cal_psi"], ta_cal, tvv, delta)
        _, _, Cb = tv.certificates(cal_bdd, delta)
        print(f"  delta={delta}: MB B_hat={Bp:.3f} A_hat={Ap:.4f} C_hat_MB={Cp:.4g} | BDD C_hat={Cb:.4g} | ratio {Cb/Cp:.3g}")


# ---------------------------------------------------------------- stage: detect
@njit(parallel=True, cache=True)
def mb_detect_runs(pbar, Qgood, t_det, m, seed0):
    """MB detector, uniform actions. pbar, Qgood: (R, N, M) Pr(next = Good). Returns
    max retained ||q_hat - p_bar||_1 per run."""
    n_runs = pbar.shape[0]
    out = np.zeros(n_runs)
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        G = np.zeros((N, M))
        T = np.zeros((N, M))
        s = np.random.randint(N)
        for _ in range(t_det):
            a = np.random.randint(M)
            xn = 0 if np.random.random() < Qgood[r, s, a] else 1
            if xn == 0:
                G[s, a] += 1.0
            T[s, a] += 1.0
            s = xn
        best = 0.0
        for i in range(N):
            for j in range(M):
                if T[i, j] > m * t_det:
                    b = 2.0 * abs(G[i, j] / T[i, j] - pbar[r, i, j])
                    if b > best:
                        best = b
        out[r] = best
    return out


D_DELTA, D_RHO, D_TDET_FACTOR, D_RUNS = 0.5, 0.5, 10, 400
D_T_ADV = np.array([316, 1000, 3162, 10000], dtype=np.int64)


def detect():
    Qs, _, C = tv.bellman_targets(GAMMA)
    g = dict(np.load("mb_grid.npz"))
    Bp, Ap, Cp = certificates_mb(g["cal_err"], g["cal_psi"], g["cal_t_agt"], g["t_adv"], D_DELTA)
    B, A, Cb = tv.certificates(dict(np.load("theory_validation_exp12_partial.npz")), D_DELTA)
    t_agt = np.ceil((1 + D_RHO) ** 2 * Cp * D_T_ADV).astype(np.int64)
    print(f"delta={D_DELTA} rho={D_RHO}: C_hat_MB={Cp:.4g} -> t_agt={t_agt.tolist()} (BDD C_hat={Cb:.4g})")
    m = tv.M_FRAC / (N * M)
    cnt = tv.run_adversary(D_RUNS, D_T_ADV, 6 * 10**7)          # same adversary runs as theory_validation.exp3
    Qh = tv.run_agent(D_RUNS, t_agt, GAMMA, 9 * 10**7)          # BDD agent at the MB certified time
    rng = np.random.default_rng(13)
    rows = []
    for k, (tva, ta) in enumerate(zip(D_T_ADV, t_agt)):
        t_det = int(D_TDET_FACTOR * ta)
        n_pair = ta // (N * M)
        pbar = rng.binomial(n_pair, tv.P_GOOD, size=(D_RUNS, N, M)) / n_pair
        ph = np.ascontiguousarray(tv.p_hat_from_counts(cnt[:, k])[..., 0])
        sec = np.broadcast_to(tv.P_GOOD, (D_RUNS, N, M)).copy()
        thr_mb = (1 + D_RHO) * Bp / np.sqrt(ta)
        thr_bdd = (1 + GAMMA) * (1 + D_RHO) * B / np.sqrt(ta)
        t0 = time.time()
        mb_sec = mb_detect_runs(pbar, sec, t_det, m, 7 * 10**7 + 1000 * k)
        mb_att = mb_detect_runs(pbar, ph, t_det, m, 8 * 10**7 + 1000 * k)
        Qk = np.ascontiguousarray(Qh[:, k])
        bd_sec = tv.detect_runs(Qk, sec, t_det, GAMMA, tv.R, tv.SIGMA_W, m, 7 * 10**7 + 1000 * k)
        bd_att = tv.detect_runs(Qk, ph, t_det, GAMMA, tv.R, tv.SIGMA_W, m, 8 * 10**7 + 1000 * k)
        r = (tva, ta, t_det, (mb_sec > thr_mb).mean(), (mb_att <= thr_mb).mean(),
             (bd_sec > thr_bdd).mean(), (bd_att <= thr_bdd).mean())
        rows.append(r)
        print(f"  t_adv={tva} t_agt={ta} t_det={t_det}: MB FA={r[3]:.4f} miss={r[4]:.4f} | "
              f"BDD (same t_agt) FA={r[5]:.4f} miss={r[6]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    np.save("mb_detect.npy", np.array(rows))


# ---------------------------------------------------------------- stage: value-neutral attack
# 3-state channel: 0 = Good, 1 = Medium, 2 = Bad; actions 0 = low, 1 = high power.
P3 = np.array([[[0.70, 0.20, 0.10], [0.60, 0.25, 0.15]],
               [[0.30, 0.45, 0.25], [0.35, 0.40, 0.25]],
               [[0.10, 0.30, 0.60], [0.20, 0.30, 0.50]]])
R3 = np.zeros((3, 2, 3))
R3[:, 0, :] = [1.0, 0.5, 0.0]               # delivered fraction at low power
R3[:, 1, :] = [1.2, 0.8, 0.1]               # high power: more delivered ...
R3[:, 1, :] -= 0.3                          # ... minus power cost
NEUTRAL_L1 = 0.2                            # ||p_tilde_{i,j}||_1 of the value-neutral attack
NT_RUNS = 200
NT_CPS = np.unique(np.round(np.logspace(3, 6, 13)).astype(np.int64))


def neutral_attack(C3):
    """p_hat = p - p_tilde with p_tilde orthogonal to 1 and to c_{i,j}: Phi_{i,j} = 0 for every pair."""
    Ph = np.zeros_like(P3)
    for i in range(3):
        for j in range(2):
            v = np.cross(np.ones(3), C3[i, j])          # orthogonal to 1 and c_{i,j}
            v *= NEUTRAL_L1 / np.abs(v).sum()
            s = 1.0
            while np.any(P3[i, j] - s * v < 0) or np.any(P3[i, j] - s * v > 1):
                s = -s if s > 0 else -s / 2              # flip sign, then shrink, until feasible
            Ph[i, j] = P3[i, j] - s * v
    return Ph


@njit(parallel=True, cache=True)
def neutral_traces(Q, Pbar, Prep, Rw, sigma_w, gamma, cps, m, n_runs, seed0):
    """Both detectors on one detection trajectory with reported kernel Prep (3, 2, 3), oracle
    Q = Q* and p_bar = P. Returns max retained |BDPM| and max retained ||q_hat - p||_1 at each checkpoint."""
    K = cps.shape[0]
    bd = np.zeros((n_runs, K))
    mb = np.zeros((n_runs, K))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        V = np.zeros(3)
        for x in range(3):
            V[x] = max(Q[x, 0], Q[x, 1])
        D = np.zeros((3, 2))
        T = np.zeros((3, 2))
        Cn = np.zeros((3, 2, 3))
        s = np.random.randint(3)
        c = 0
        t = 0
        while c < K:
            a = np.random.randint(2)
            u = np.random.random()
            xn = 0 if u < Prep[s, a, 0] else (1 if u < Prep[s, a, 0] + Prep[s, a, 1] else 2)
            D[s, a] += Q[s, a] - (Rw[s, a, xn] + sigma_w * np.random.standard_normal()) - gamma * V[xn]
            T[s, a] += 1.0
            Cn[s, a, xn] += 1.0
            s = xn
            t += 1
            if t == cps[c]:
                b1 = 0.0
                b2 = 0.0
                for i in range(3):
                    for j in range(2):
                        if T[i, j] > m * t:
                            v1 = abs(D[i, j]) / T[i, j]
                            v2 = 0.0
                            for x in range(3):
                                v2 += abs(Cn[i, j, x] / T[i, j] - Pbar[i, j, x])
                            b1 = max(b1, v1)
                            b2 = max(b2, v2)
                bd[r, c] = b1
                mb[r, c] = b2
                c += 1
    return bd, mb


def value_iteration3(gamma, tol=1e-14):
    V = np.zeros(3)
    while True:
        Q = (P3 * (R3 + gamma * V[None, None, :])).sum(axis=2)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < tol:
            return (P3 * (R3 + gamma * Vn[None, None, :])).sum(axis=2), Vn
        V = Vn


def neutral():
    Q3, V3 = value_iteration3(GAMMA)
    C3 = R3 + GAMMA * V3[None, None, :]
    Ph = neutral_attack(C3)
    phi = ((P3 - Ph) * C3).sum(-1)
    psi = np.abs(P3 - Ph).sum(-1)
    print("Q* =", Q3.round(4).tolist())
    print("max |Phi_{i,j}| =", np.abs(phi).max(), " min ||p_tilde||_1 =", psi.min().round(4))
    m = tv.M_FRAC / (3 * 2)
    t0 = time.time()
    bd_s, mb_s = neutral_traces(Q3, P3, P3, R3, tv.SIGMA_W, GAMMA, NT_CPS, m, NT_RUNS, 101)
    bd_a, mb_a = neutral_traces(Q3, P3, Ph, R3, tv.SIGMA_W, GAMMA, NT_CPS, m, NT_RUNS, 202)
    print(f"  {NT_RUNS} runs per case to t_det={NT_CPS[-1]:.0e} ({time.time()-t0:.0f}s)")
    for name, s, a in (("BDD", bd_s, bd_a), ("MB ", mb_s, mb_a)):
        print(f"  {name} t_det=1e6: secure median {np.median(s[:, -1]):.2e}, attacked median {np.median(a[:, -1]):.2e}; "
              f"attacked > secure 99th pct: {(a[:, -1] > np.quantile(s[:, -1], 0.99)).mean():.3f}")
    np.savez("mb_neutral.npz", t=NT_CPS, bd_s=bd_s, bd_a=bd_a, mb_s=mb_s, mb_a=mb_a, Ph=Ph, phi=phi, psi=psi)


# ---------------------------------------------------------------- stage: sweep (equal false-alarm rate)
SW_T_ADV, SW_RUNS, SW_FA, SW_TDET_FACTOR = 1000, 400, 0.01, 4
SW_RATIOS = np.array([10, 30, 100, 300, 1000, 3000], dtype=np.int64)


def sweep():
    """Miss rate of each detector at a common false-alarm rate SW_FA, as t_agt / t_adv varies.
    Thresholds are oracle: the (1 - SW_FA)-quantile of each statistic over independent secure runs."""
    t_agt = SW_RATIOS * SW_T_ADV
    m = tv.M_FRAC / (N * M)
    cnt = tv.run_adversary(SW_RUNS, np.array([SW_T_ADV], dtype=np.int64), 6 * 10**7)
    ph = np.ascontiguousarray(tv.p_hat_from_counts(cnt[:, 0])[..., 0])
    sec = np.broadcast_to(tv.P_GOOD, (SW_RUNS, N, M)).copy()
    Qh_cal = tv.run_agent(SW_RUNS, t_agt, GAMMA, 21 * 10**7)
    Qh = tv.run_agent(SW_RUNS, t_agt, GAMMA, 22 * 10**7)
    rng = np.random.default_rng(23)
    rows = []
    for k, ta in enumerate(t_agt):
        t_det = int(SW_TDET_FACTOR * ta)
        n_pair = ta // (N * M)
        pb_cal = rng.binomial(n_pair, tv.P_GOOD, size=(SW_RUNS, N, M)) / n_pair
        pb = rng.binomial(n_pair, tv.P_GOOD, size=(SW_RUNS, N, M)) / n_pair
        sd = 3 * 10**8 + 1000 * k
        thr_mb = np.quantile(mb_detect_runs(pb_cal, sec, t_det, m, sd), 1 - SW_FA)
        thr_bd = np.quantile(tv.detect_runs(np.ascontiguousarray(Qh_cal[:, k]), sec, t_det, GAMMA, tv.R, tv.SIGMA_W, m, sd), 1 - SW_FA)
        Qk = np.ascontiguousarray(Qh[:, k])
        mb_fa = (mb_detect_runs(pb, sec, t_det, m, sd + 1) > thr_mb).mean()
        mb_miss = (mb_detect_runs(pb, ph, t_det, m, sd + 2) <= thr_mb).mean()
        bd_fa = (tv.detect_runs(Qk, sec, t_det, GAMMA, tv.R, tv.SIGMA_W, m, sd + 1) > thr_bd).mean()
        bd_miss = (tv.detect_runs(Qk, ph, t_det, GAMMA, tv.R, tv.SIGMA_W, m, sd + 2) <= thr_bd).mean()
        rows.append((ta / SW_T_ADV, mb_fa, mb_miss, bd_fa, bd_miss))
        print(f"  t_agt/t_adv={ta // SW_T_ADV}: MB FA={mb_fa:.4f} miss={mb_miss:.4f} | BDD FA={bd_fa:.4f} miss={bd_miss:.4f}", flush=True)
    np.save("mb_sweep.npy", np.array(rows))


# ---------------------------------------------------------------- stage: plot
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "mathtext.fontset": "cm"})

    # --- (a) level sets of Pr[sufficient condition] for both detectors
    g = dict(np.load("mb_grid.npz"))
    ta, tvv = g["t_agt"], g["t_adv"]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for G, col, name in ((g["G_bdd"], BLUE, "BDD"), (g["G_mb"], INK, "model-based")):
        for lev, ls in ((0.5, "-"), (0.8, "--")):
            x, y = tv.contour_points(G, ta, tvv, lev)
            ax.plot(x, y, color=col, ls=ls, lw=1.4, label=f"{name}, Pr = {lev}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(tv.T_ADV_MIN, 1e5); ax.set_ylim(1e3, 1e8)
    ax.set_xlabel(r"adversary learning time $t_{\rm adv}$")
    ax.set_ylabel(r"agent learning time $t_{\rm agt}$")
    ax.legend(fontsize=6.3, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(); fig.savefig("mb_level_sets.png", dpi=300); plt.close(fig)

    # --- (b) value-neutral attack: both statistics vs detection time
    d = dict(np.load("mb_neutral.npz"))
    t = d["t"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7), sharex=True)
    for ax, (s, a, title) in zip(axs, ((d["bd_s"], d["bd_a"], "BDD: max. $|$BDPM$|$"),
                                       (d["mb_s"], d["mb_a"], r"model-based: max. $\|\hat{q}-\bar{p}\|_1$"))):
        for X, col, lab in ((s, BLUE, "no attack"), (a, ORANGE, "attack")):
            lo, med, hi = np.quantile(X, [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(t, med, color=col, lw=1.3, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xlabel(r"detection time $t$")
    axs[0].legend(fontsize=6.3, loc="lower left", handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(); fig.savefig("mb_value_neutral.png", dpi=300); plt.close(fig)
    print("saved mb_level_sets.png, mb_value_neutral.png")


if __name__ == "__main__":
    {"grid": grid, "detect": detect, "neutral": neutral, "sweep": sweep, "plot": plot}[sys.argv[1]]()
