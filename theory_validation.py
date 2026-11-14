"""Direct numerical validation of the learning-time theory (paper, Section VII).

Tests each link of the chain  Assumption 2 + Lemma 1  =>  Theorem 8 (linear law)
=>  Theorem 9 (end-to-end detection)  on a 2-state, 2-action networked example
(a Gilbert-Elliott packet-loss channel observed through the intercepted link),
small enough that the certified regime t_agt >= T*_agt(t_adv, delta) is
computationally reachable.

  Exp. 1  certificate rates: (1 - delta/2)-quantile of ||Q_hat_t - Q*||_inf vs t,
          delta/2-quantile of Phi_min vs t_adv (log-log; slope -1/2 expected);
          negative control gamma = 0.8, where 1/n Q-learning is slower than t^{-1/2}.
  Exp. 2  probability of the information-advantage event on (t_adv, t_agt):
          contours should be lines of slope 1; certified line T* = C t_adv;
          Pr[E] >= 1 - delta above it (certificates calibrated on independent runs).
  Exp. 3  Algorithm 1 along t_agt = (1+rho)^2 C t_adv, t_det proportional to t_adv,
          with the deterministic certificate threshold (1+rho) eps_hat_{delta/2}(t_agt).

Agent:     Polyak-Ruppert averaged synchronous Q-learning with a generative model
           (step size k^{-0.75}), Gaussian reward noise sigma_w; t_agt counts samples.
Adversary: Algorithm 2 (empirical frequencies) on one trajectory, uniform actions.
Detector:  Algorithm 1 with uniformly random actions, m = 1/(2NM).

Usage (in order; set TV_QUICK=1 for a small smoke test):
  python3 theory_validation.py calib       # ~30 s: certificate constants and certified ratios
  python3 theory_validation.py exp12       # ~2 h on 10 cores: Exp. 1-2 runs -> theory_validation_exp12*.npz
  python3 theory_validation.py analyze12   # slopes, contour slopes, Pr[E] checks -> theory_validation_iagrid.npy
  python3 theory_validation.py exp3        # ~45 min: Exp. 3 (needs theory_validation_exp12_partial.npz)
  python3 theory_validation.py single      # ~2 min: single-instance detection runs
  python3 theory_validation.py plot        # theory_ia_map.png, theory_single_instance.png
"""
import sys
import time

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- MDP
N, M = 2, 2                      # states: 0 = Good, 1 = Bad; actions: 0 = low, 1 = high power
P_GOOD = np.array([[0.8137, 0.6561],   # Pr(next = Good | state, action)
                   [0.2718, 0.3183]])  # off the k/n lattice: Phi_{i,j} != 0 a.s.
P = np.zeros((N, M, N))
P[:, :, 0] = P_GOOD
P[:, :, 1] = 1.0 - P_GOOD
# r(x, u, x'): packet delivered (x' = Good) or partially delivered at high power, minus power cost
R = np.zeros((N, M, N))
R[:, 0, 0], R[:, 0, 1] = 1.0, 0.0
R[:, 1, 0], R[:, 1, 1] = 1.5, 0.3
R[:, 1, :] -= 0.3
SIGMA_W = 0.3
M_FRAC = 0.5                     # m = M_FRAC / (N M) < 1/(N M)


def value_iteration(gamma, tol=1e-14):
    V = np.zeros(N)
    while True:
        Q = (P * (R + gamma * V[None, None, :])).sum(axis=2)
        Vn = Q.max(axis=1)
        if np.abs(Vn - V).max() < tol:
            return (P * (R + gamma * Vn[None, None, :])).sum(axis=2), Vn
        V = Vn


def bellman_targets(gamma):
    Qs, Vs = value_iteration(gamma)
    C = R + gamma * Vs[None, None, :]          # c_{i,j}(x') = r(i,j,x') + gamma V*(x')
    return Qs, Vs, C


# ---------------------------------------------------------------- agent
PR_ALPHA = 0.75   # step size eta_k = k^{-alpha}, alpha in (1/2, 1)  [Li, Yang, Liang, Zhang, Jordan 2023]


@njit(parallel=True, cache=True)
def agent_runs(n_runs, checkpoints, gamma, Qstar, Pg, Rw, sigma_w, seed0):
    """Polyak-Ruppert averaged synchronous Q-learning with a generative model.

    Each iteration k draws one transition and noisy reward for every pair (x, u) and sets
    Q_k = (1 - eta_k) Q_{k-1} + eta_k (r + w + gamma max_b Q_{k-1}(x', b)), eta_k = k^{-PR_ALPHA};
    Q_hat is the running average of Q_1, ..., Q_k. t_agt counts samples (N M per iteration).
    Returns Q_hat at each checkpoint (first iteration whose sample count reaches it)."""
    K = checkpoints.shape[0]
    out = np.zeros((n_runs, K, N, M))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        Q = np.zeros((N, M))
        Qn = np.zeros((N, M))
        Qbar = np.zeros((N, M))
        it = 0
        c = 0
        while c < K:
            it += 1
            eta = it ** (-PR_ALPHA)
            for s in range(N):
                for a in range(M):
                    xn = 0 if np.random.random() < Pg[s, a] else 1
                    vn = Q[xn, 0] if Q[xn, 0] > Q[xn, 1] else Q[xn, 1]
                    tgt = Rw[s, a, xn] + sigma_w * np.random.standard_normal() + gamma * vn
                    Qn[s, a] = (1.0 - eta) * Q[s, a] + eta * tgt
            for s in range(N):
                for a in range(M):
                    Q[s, a] = Qn[s, a]
                    Qbar[s, a] += (Q[s, a] - Qbar[s, a]) / it
            while c < K and it * N * M >= checkpoints[c]:
                out[r, c] = Qbar
                c += 1
    return out


# ---------------------------------------------------------------- adversary
@njit(parallel=True, cache=True)
def adversary_runs(n_runs, checkpoints, Pg, seed0):
    """Transition counts n(i,j,x') of one uniform-action trajectory at each checkpoint."""
    K = checkpoints.shape[0]
    out = np.zeros((n_runs, K, N, M, N))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        cnt = np.zeros((N, M, N))
        s = np.random.randint(N)
        k = 0
        t = 0
        while k < K:
            a = np.random.randint(M)
            xn = 0 if np.random.random() < Pg[s, a] else 1
            cnt[s, a, xn] += 1.0
            s = xn
            t += 1
            if t == checkpoints[k]:
                out[r, k] = cnt
                k += 1
    return out


def p_hat_from_counts(cnt):
    tot = cnt.sum(axis=-1, keepdims=True)
    return np.where(tot > 0, cnt / np.maximum(tot, 1), 1.0 / N)


def phi_from_counts(cnt, C):
    """Phi_{i,j} = (p_{i,j} - p_hat_{i,j}) c_{i,j}^T, shape (..., N, M)."""
    return ((P - p_hat_from_counts(cnt)) * C).sum(axis=-1)


def phi_V(phi):
    """Maximum Bellman deviation signal over the visitation set V. Detection uses uniformly random actions with
    m = 1/(2NM), which retain every pair (see `phiV`), so V = all pairs. phi: (..., N, M) -> (...)."""
    return np.abs(phi).reshape(*phi.shape[:-2], -1).max(-1)


def visit_freq_min(ph):
    """Smallest asymptotic visit frequency of a pair under uniform actions and reported kernel ph (..., N, M, N)."""
    K = ph.mean(-2)                                   # (..., N, N) state chain under uniform actions
    pg = K[..., 1, 0] / (K[..., 1, 0] + K[..., 0, 1])  # stationary Pr(Good) for a 2-state chain
    return np.minimum(pg, 1 - pg) / M


# ---------------------------------------------------------------- detector
@njit(parallel=True, cache=True)
def detect_runs(Qh, Qgood, t_det, gamma, Rw, sigma_w, m, seed0):
    """Algorithm 1, uniform actions. Qh: (R, N, M); Qgood: (R, N, M) Pr(next = Good) of the
    reported kernel. Returns max retained |BDPM| per run."""
    n_runs = Qh.shape[0]
    out = np.zeros(n_runs)
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        V0 = max(Qh[r, 0, 0], Qh[r, 0, 1])
        V1 = max(Qh[r, 1, 0], Qh[r, 1, 1])
        D = np.zeros((N, M))
        T = np.zeros((N, M))
        s = np.random.randint(N)
        for _ in range(t_det):
            a = np.random.randint(M)
            xn = 0 if np.random.random() < Qgood[r, s, a] else 1
            v = V0 if xn == 0 else V1
            D[s, a] += Qh[r, s, a] - (Rw[s, a, xn] + sigma_w * np.random.standard_normal()) - gamma * v
            T[s, a] += 1.0
            s = xn
        best = 0.0
        for i in range(N):
            for j in range(M):
                if T[i, j] > m * t_det:
                    b = abs(D[i, j]) / T[i, j]
                    if b > best:
                        best = b
        out[r] = best
    return out


# ---------------------------------------------------------------- helpers
def logspace_int(lo, hi, per_decade):
    n = int(round((np.log10(hi) - np.log10(lo)) * per_decade)) + 1
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n)).astype(np.int64))


def agent_errors(Qhat, Qstar):
    return np.abs(Qhat - Qstar).reshape(Qhat.shape[0], Qhat.shape[1], -1).max(axis=2)


def run_agent(n_runs, cps, gamma, seed0):
    Qs, _, _ = bellman_targets(gamma)
    t0 = time.time()
    Qh = agent_runs(n_runs, cps, gamma, Qs, P_GOOD, R, SIGMA_W, seed0)
    print(f"  agent: {n_runs} runs to t={cps[-1]:.2e}, gamma={gamma}: {time.time()-t0:.0f}s", flush=True)
    return Qh


def run_adversary(n_runs, cps, seed0):
    t0 = time.time()
    cnt = adversary_runs(n_runs, cps, P_GOOD, seed0)
    print(f"  adversary: {n_runs} runs to t={cps[-1]:.2e}: {time.time()-t0:.0f}s", flush=True)
    return cnt


# ---------------------------------------------------------------- stages
GAMMA = 0.3
DELTAS = (0.5, 0.2)


def calib():
    """Quick look: certificate constants and certified ratio for each delta."""
    Qs, Vs, C = bellman_targets(GAMMA)
    print("Q* =", Qs.round(4).tolist(), " V* =", Vs.round(4).tolist())
    print("Var c_ij:", (((C - (P * C).sum(-1, keepdims=True)) ** 2 * P).sum(-1)).round(4).tolist())
    cpa = logspace_int(1e2, 1e6, 4)
    cpv = logspace_int(1e2, 1e5, 4)
    e = agent_errors(run_agent(1000, cpa, GAMMA, 1), Qs)
    pm = np.abs(phi_from_counts(run_adversary(2000, cpv, 10**6), C)).reshape(2000, len(cpv), -1).min(-1)
    for d in DELTAS:
        qa = np.sqrt(cpa) * np.quantile(e, 1 - d / 2, axis=0)
        qv = np.sqrt(cpv) * np.quantile(pm, d / 2, axis=0)
        print(f"delta={d}: sqrt(t) q_agent = {qa.round(2).tolist()}")
        print(f"          sqrt(t) q_adv   = {qv.round(4).tolist()}")
        B, A = qa[cpa >= 1e4].max(), qv[cpv >= 1e3].min()
        print(f"          B={B:.3f} A={A:.4f} certified ratio (2(1+g)B/A)^2 = {(2*(1+GAMMA)*B/A)**2:.3g}")


# ---------------------------------------------------------------- Exp. 1-2 simulation
CAL_AGT, CAL_ADV = 1000, 4000          # independent calibration runs (certificates)
EVA_AGT, EVA_ADV = 1000, 4000          # evaluation runs
T_AGT_MAX, T_ADV_MAX = 10**9, 10**6
NEG_GAMMA, NEG_T_MAX = 0.8, 10**8
if __import__("os").environ.get("TV_QUICK"):   # smoke test sizes
    CAL_AGT, CAL_ADV, EVA_AGT, EVA_ADV = 200, 400, 200, 400
    T_AGT_MAX, T_ADV_MAX, NEG_T_MAX = 10**7, 10**5, 10**6


def exp12():
    Qs, _, C = bellman_targets(GAMMA)
    cpa = logspace_int(1e2, T_AGT_MAX, 8)
    cpv = logspace_int(1e2, T_ADV_MAX, 8)
    cpa_cal = logspace_int(1e2, 1e7, 8)
    res = {}
    res["cal_t_agt"], res["cal_t_adv"] = cpa_cal, cpv
    res["cal_err"] = agent_errors(run_agent(CAL_AGT, cpa_cal, GAMMA, 1), Qs)
    res["cal_phimin"] = np.abs(phi_from_counts(run_adversary(CAL_ADV, cpv, 2 * 10**7), C)).reshape(CAL_ADV, len(cpv), -1).min(-1)
    res["t_adv"] = cpv
    res["phimin"] = np.abs(phi_from_counts(run_adversary(EVA_ADV, cpv, 3 * 10**7), C)).reshape(EVA_ADV, len(cpv), -1).min(-1)
    np.savez("theory_validation_exp12_partial.npz", **res)
    res["t_agt"] = cpa
    res["err"] = agent_errors(run_agent(EVA_AGT, cpa, GAMMA, 10**7), Qs)
    np.savez("theory_validation_exp12.npz", **res)
    print("saved theory_validation_exp12.npz")


def phiV():
    """Recompute the adversary's signals of Exp. 1-2 (same seeds) as Phi_V = max over V = all pairs; add them to
    the saved results without rerunning the agent. Also checks that uniform detection actions retain every pair."""
    Qs, _, C = bellman_targets(GAMMA)
    for f in ("theory_validation_exp12_partial.npz", "theory_validation_exp12.npz"):
        d = dict(np.load(f))
        cpv = d["cal_t_adv"]
        cal = phi_from_counts(run_adversary(len(d["cal_phimin"]), cpv, 2 * 10**7), C)
        eva = phi_from_counts(run_adversary(len(d["phimin"]), d["t_adv"], 3 * 10**7), C)
        assert np.allclose(np.abs(cal).reshape(*cal.shape[:2], -1).min(-1), d["cal_phimin"]), "seed mismatch"
        assert np.allclose(np.abs(eva).reshape(*eva.shape[:2], -1).min(-1), d["phimin"]), "seed mismatch"
        d["cal_phiV"], d["phiV"] = phi_V(cal), phi_V(eva)
        np.savez(f, **d)
        print(f"{f}: added cal_phiV, phiV")
    print(f"min asymptotic pair visit frequency, uniform actions, true kernel: {visit_freq_min(P).min():.3f} "
          f"(m = {M_FRAC / (N * M):.3f})")


T_ADV_MIN = 316   # below this, some adversary models give a pair visit frequency < m, so V = all pairs can fail


def certificates(d, delta):
    """Empirical certificates from the calibration runs: eps_hat = B/sqrt(t), phi_hat = A/sqrt(t)."""
    ta, tv = d["cal_t_agt"], d["cal_t_adv"]
    qa = np.sqrt(ta) * np.quantile(d["cal_err"], 1 - delta / 2, axis=0)
    qv = np.sqrt(tv) * np.quantile(d["cal_phiV"], delta / 2, axis=0)
    B = qa[ta >= 1e3].max()
    A = qv[tv >= T_ADV_MIN].min()
    return B, A, (2 * (1 + GAMMA) * B / A) ** 2


def loglog_slope(t, q, tmin):
    sel = t >= tmin
    return np.polyfit(np.log(t[sel]), np.log(q[sel]), 1)[0]


def boot_slope(t, X, level, tmin, nb=300, seed=0):
    """Slope of log quantile vs log t with a run-bootstrap 95% CI. X: (runs, K)."""
    rng = np.random.default_rng(seed)
    s0 = loglog_slope(t, np.quantile(X, level, axis=0), tmin)
    bs = [loglog_slope(t, np.quantile(X[rng.integers(len(X), size=len(X))], level, axis=0), tmin)
          for _ in range(nb)]
    return s0, np.percentile(bs, [2.5, 97.5])


def prob_ia_real(phimin_col, err_col, k):
    """Pr[Phi_min > k * err] with independent agent/adversary runs, all pairings."""
    sp = np.sort(phimin_col)
    return 1.0 - np.searchsorted(sp, k * err_col, side="right").mean() / len(sp)


def ia_grid(d):
    k = 2 * (1 + GAMMA)
    ta, tv = d["t_agt"], d["t_adv"]
    G = np.zeros((len(tv), len(ta)))
    for i in range(len(tv)):
        for j in range(len(ta)):
            G[i, j] = prob_ia_real(d["phiV"][:, i], d["err"][:, j], k)
    return G


def contour_points(G, ta, tv, level):
    """For each t_adv, the t_agt (log-interpolated) where Pr[IA] first reaches `level`."""
    xs, ys = [], []
    for i in range(len(tv)):
        row = G[i]
        idx = np.where(row >= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            continue
        j = idx[0]
        x0, x1 = np.log(ta[j - 1]), np.log(ta[j])
        y0, y1 = row[j - 1], row[j]
        xs.append(tv[i])
        ys.append(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))
    return np.array(xs), np.array(ys)


def analyze12():
    d = dict(np.load("theory_validation_exp12.npz"))
    ta, tv = d["t_agt"], d["t_adv"]
    print("=== Exp. 1: certificate rates (eval runs; slope of log-quantile vs log t, 95% bootstrap CI)")
    for delta in DELTAS:
        s, ci = boot_slope(ta, d["err"], 1 - delta / 2, 1e3)
        print(f"  agent  q_{1-delta/2:.2f}(||Q-Q*||), t in [1e3,1e9]: slope {s:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]")
        s, ci = boot_slope(tv, d["phiV"], delta / 2, T_ADV_MIN)
        print(f"  adv    q_{delta/2:.2f}(Phi_V),     t in [316,1e6]: slope {s:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]")
    print("=== Exp. 2: information advantage event")
    G = ia_grid(d)
    for lev in (0.5, 0.8, 0.9):
        x, y = contour_points(G, ta, tv, lev)
        sel = (x <= 1e5) & (x >= T_ADV_MIN)
        if sel.sum() >= 3:
            sl = np.polyfit(np.log(x[sel]), np.log(y[sel]), 1)
            print(f"  Pr[IA_real] = {lev}: contour slope {sl[0]:.3f} over t_adv in [{x[sel].min():.0f},{x[sel].max():.0f}]"
                  f" ({sel.sum()} pts); t_agt/t_adv in [{(y/x)[sel].min():.3g},{(y/x)[sel].max():.3g}]")
    for delta in DELTAS:
        B, A, C = certificates(d, delta)
        print(f"  delta={delta}: B_hat={B:.3f}, A_hat={A:.4f}, certified slope C_hat={C:.4g}")
        # Pr[E] = Pr[Phi_min >= A/sqrt(t_adv)] * Pr[err <= B/sqrt(t_agt)] (independent runs)
        pa = (d["err"] <= B / np.sqrt(ta)).mean(axis=0)
        pv = (d["phiV"] >= A / np.sqrt(tv)).mean(axis=0)
        worst, nc = 1.0, 0
        for i in range(len(tv)):
            for j in range(len(ta)):
                if ta[j] >= C * tv[i] and ta[j] >= 1e3 and tv[i] >= T_ADV_MIN:
                    worst = min(worst, pa[j] * pv[i]); nc += 1
        print(f"     min Pr[E] over {nc} cells with t_agt >= C_hat t_adv: {worst:.4f}  (required >= {1-delta})")
        # Pr[IA with certified threshold] at the certified line
        pc = []
        for i in range(len(tv)):
            tstar = C * tv[i]
            if tstar <= ta[-1] and tv[i] >= T_ADV_MIN:
                pc.append((d["phiV"][:, i] > 2 * (1 + GAMMA) * B / np.sqrt(tstar)).mean())
        if pc:
            print(f"     Pr[Phi_V > 2(1+g) eps_hat(T*)] along the certified line: min {min(pc):.4f} max {max(pc):.4f}"
                  f" ({len(pc)} t_adv values; required >= {1-delta/2})")
    np.save("theory_validation_iagrid.npy", G)


# ---------------------------------------------------------------- Exp. 3 end-to-end detection
E3_DELTA, E3_RHO, E3_TDET_FACTOR, E3_RUNS = 0.5, 0.5, 10, 400
E3_T_ADV = np.array([316, 1000, 3162, 10000], dtype=np.int64)
if __import__("os").environ.get("TV_QUICK"):
    E3_RUNS, E3_TDET_FACTOR, E3_T_ADV = 100, 1, np.array([100, 200], dtype=np.int64)


def exp3():
    Qs, Vs, C = bellman_targets(GAMMA)
    cal = dict(np.load("theory_validation_exp12_partial.npz"))
    B, A, Cd = certificates(cal, E3_DELTA)
    t_agt = np.ceil((1 + E3_RHO) ** 2 * Cd * E3_T_ADV).astype(np.int64)
    print(f"delta={E3_DELTA} rho={E3_RHO}: B_hat={B:.3f} A_hat={A:.4f} C_hat={Cd:.4g}; t_agt={t_agt.tolist()}")
    Qh = run_agent(E3_RUNS, t_agt, GAMMA, 5 * 10**7)
    cnt = run_adversary(E3_RUNS, E3_T_ADV, 6 * 10**7)
    m = M_FRAC / (N * M)
    Delta = (C.max(-1) - C.min(-1)).max()
    s2 = SIGMA_W ** 2 + Delta ** 2 / 4
    rows = []
    for k, (tv, ta) in enumerate(zip(E3_T_ADV, t_agt)):
        t_det = int(E3_TDET_FACTOR * ta)
        eps_c = B / np.sqrt(ta)
        phi = A / np.sqrt(tv)
        thr = (1 + GAMMA) * (1 + E3_RHO) * eps_c
        Qk = np.ascontiguousarray(Qh[:, k])
        ph = p_hat_from_counts(cnt[:, k])
        sec_rows = np.broadcast_to(P_GOOD, (E3_RUNS, N, M)).copy()
        t0 = time.time()
        st_sec = detect_runs(Qk, sec_rows, t_det, GAMMA, R, SIGMA_W, m, 7 * 10**7 + 1000 * k)
        st_att = detect_runs(Qk, np.ascontiguousarray(ph[..., 0]), t_det, GAMMA, R, SIGMA_W, m, 8 * 10**7 + 1000 * k)
        err = np.abs(Qk - Qs).reshape(E3_RUNS, -1).max(1)
        phiv = phi_V(((P - ph) * C).sum(-1))
        ev = (err <= eps_c) & (phiv >= phi)
        vmin = visit_freq_min(ph).min()
        fa, miss = st_sec > thr, st_att <= thr
        # Theorem 9, sub-Gaussian reward noise (Gaussian here)
        fa_bd = 2 * N * M * np.exp(-m * t_det * (1 + GAMMA) ** 2 * E3_RHO ** 2 * eps_c ** 2 / (2 * s2))
        miss_bd = 2 * np.exp(-m * t_det * E3_RHO ** 2 * phi ** 2 / (8 * (1 + E3_RHO) ** 2 * s2))   # + r_V(t_det)
        rows.append((tv, ta, t_det, fa.mean(), miss.mean(), ev.mean(), fa[ev].mean(), miss[ev].mean(), fa_bd, miss_bd))
        print(f"  t_adv={tv} t_agt={ta} t_det={t_det}: P(FA)={fa.mean():.4f} P(miss)={miss.mean():.4f} "
              f"P(E)={ev.mean():.3f} | given E: FA={fa[ev].mean():.4f} miss={miss[ev].mean():.4f} "
              f"| Thm 9 bounds FA<={fa_bd:.3g} miss<={miss_bd:.3g}+r_V | min visit freq under attack {vmin:.3f} (m={m:.3f}) "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save("theory_validation_exp3.npy", np.array(rows))


# ---------------------------------------------------------------- single instance
SI_T_ADV, SI_RUNS, SI_POOL = 1000, 50, 200     # operating point, detection runs per case, training pool


@njit(parallel=True, cache=True)
def detect_traces(Qh, Qgood, t_det, checkpoints, gamma, Rw, sigma_w, m, n_runs, seed0):
    """Algorithm 1 with fixed Q_hat (N, M) and reported kernel Qgood (N, M), uniform actions.
    Returns max retained |BDPM| at each checkpoint t (pairs with T_ij > m t), shape (n_runs, K)."""
    K = checkpoints.shape[0]
    out = np.zeros((n_runs, K))
    V0 = max(Qh[0, 0], Qh[0, 1])
    V1 = max(Qh[1, 0], Qh[1, 1])
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        D = np.zeros((N, M))
        T = np.zeros((N, M))
        s = np.random.randint(N)
        k = 0
        for t in range(1, t_det + 1):
            a = np.random.randint(M)
            xn = 0 if np.random.random() < Qgood[s, a] else 1
            v = V0 if xn == 0 else V1
            D[s, a] += Qh[s, a] - (Rw[s, a, xn] + sigma_w * np.random.standard_normal()) - gamma * v
            T[s, a] += 1.0
            s = xn
            if k < K and t == checkpoints[k]:
                best = 0.0
                for i in range(N):
                    for j in range(M):
                        if T[i, j] > m * t:
                            b = abs(D[i, j]) / T[i, j]
                            if b > best:
                                best = b
                out[r, k] = best
                k += 1
    return out


def single():
    """One training instance at t_adv = SI_T_ADV, t_agt = (1+rho)^2 C_hat t_adv (delta = E3_DELTA).
    Q_hat: first agent run (seed order) with ||Q_hat - Q*|| <= eps_c; P_hat: first adversary run with
    Phi_V > 2(1+gamma) eps_bar, eps_bar = (1+rho) eps_c, so the information advantage condition holds."""
    Qs, Vs, C = bellman_targets(GAMMA)
    cal = dict(np.load("theory_validation_exp12_partial.npz"))
    B, A, Cd = certificates(cal, E3_DELTA)
    t_agt = int(np.ceil((1 + E3_RHO) ** 2 * Cd * SI_T_ADV))
    eps_c = B / np.sqrt(t_agt)
    eps_bar = (1 + E3_RHO) * eps_c
    Qh = run_agent(SI_POOL, np.array([t_agt], dtype=np.int64), GAMMA, 9 * 10**7)[:, 0]
    err = np.abs(Qh - Qs).reshape(SI_POOL, -1).max(1)
    ia = np.where(err <= eps_c)[0][0]
    cnt = run_adversary(SI_POOL, np.array([SI_T_ADV], dtype=np.int64), 9 * 10**7 + 5000)[:, 0]
    phiv = phi_V(phi_from_counts(cnt, C))
    iv = np.where(phiv > 2 * (1 + GAMMA) * eps_bar)[0][0]
    Q1, ph = np.ascontiguousarray(Qh[ia]), p_hat_from_counts(cnt[iv])
    t_det = int(E3_TDET_FACTOR * t_agt)
    cps = logspace_int(1e2, t_det, 10)
    m = M_FRAC / (N * M)
    t0 = time.time()
    sec = detect_traces(Q1, P_GOOD.copy(), t_det, cps, GAMMA, R, SIGMA_W, m, SI_RUNS, 11 * 10**7)
    att = detect_traces(Q1, np.ascontiguousarray(ph[..., 0]), t_det, cps, GAMMA, R, SIGMA_W, m, SI_RUNS, 12 * 10**7)
    thr = (1 + GAMMA) * eps_bar
    print(f"single: t_adv={SI_T_ADV} t_agt={t_agt} t_det={t_det}; agent run {ia} (err={err[ia]:.2e} <= eps_c={eps_c:.2e}), "
          f"adversary run {iv} (Phi_V={phiv[iv]:.2e} > 2(1+g)eps_bar={2*(1+GAMMA)*eps_bar:.2e}); "
          f"threshold={thr:.2e}; final: secure max {sec[:, -1].max():.2e}, attacked min {att[:, -1].min():.2e} "
          f"({time.time()-t0:.0f}s)")
    np.savez("theory_validation_single.npz", t=cps, sec=sec, att=att, thr=thr, eps_bar=eps_bar,
             phiV=phiv[iv], err=err[ia], t_agt=t_agt, t_adv=SI_T_ADV, t_det=t_det)


# ---------------------------------------------------------------- figures
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e6e5e1"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "mathtext.fontset": "cm"})

    # --- (a) heat map of Pr[information advantage]
    d = dict(np.load("theory_validation_exp12.npz"))
    ta, tv = d["t_agt"], d["t_adv"]
    G = np.load("theory_validation_iagrid.npy")
    si = dict(np.load("theory_validation_single.npz"))
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    X, Y = np.meshgrid(tv, ta, indexing="ij")
    pc = ax.pcolormesh(X, Y, G, cmap=cmap, vmin=0, vmax=1, shading="nearest", rasterized=True)
    for delta, ls in zip(DELTAS, ("--", ":")):
        ax.contour(X, Y, G, levels=[1 - delta], colors="white", linewidths=1.1)   # level sets 0.5 (lower), 0.8 (upper)
        _, _, Cd = certificates(d, delta)
        ax.plot(tv, Cd * tv, color=ORANGE, ls=ls, lw=1.6,
                label=rf"$T^\star_{{\rm agt}}(t_{{\rm adv}},\delta)$, $\delta={delta}$")
    ax.plot(si["t_adv"], si["t_agt"], marker="*", ms=9, color="white", mec=INK, mew=0.8, ls="none",
            label="single instance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(T_ADV_MIN, 1e5); ax.set_ylim(1e3, 1e8)
    ax.set_xlabel(r"adversary learning time $t_{\rm adv}$")
    ax.set_ylabel(r"agent learning time $t_{\rm agt}$")
    leg = ax.legend(fontsize=6.3, loc="upper left", frameon=True, borderpad=0.4, handlelength=2.2)
    leg.get_frame().set(facecolor="white", alpha=0.9, edgecolor="none")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label("Pr[information advantage]")
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout(); fig.savefig("theory_ia_map.png", dpi=300); plt.close(fig)

    # --- (b) single instance: max |BDPM| vs detection time, many detection runs
    t, sec, att, thr = si["t"], si["sec"], si["att"], float(si["thr"])
    lb = float(si["phiV"]) - thr                        # attacked-limit lower bound Phi_V - (1+g) eps_bar
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    sel = sec[0] > 0
    t, sec, att = t[sel], sec[:, sel], att[:, sel]
    ymin, ymax = 0.008 * thr, 1.5 * max(sec.max(), att.max())
    ax.axhspan(ymin, thr, color=BLUE, alpha=0.07, lw=0)
    ax.axhspan(thr, ymax, color=ORANGE, alpha=0.07, lw=0)
    for k in range(sec.shape[0]):
        ax.plot(t, sec[k], color=BLUE, lw=0.5, alpha=0.45)
        ax.plot(t, att[k], color=ORANGE, lw=0.5, alpha=0.45)
    ax.plot([], [], color=BLUE, lw=1.5, label=f"no attack ({sec.shape[0]} runs)")
    ax.plot([], [], color=ORANGE, lw=1.5, label=f"MITM attack ({att.shape[0]} runs)")
    ax.axhline(thr, color=INK, ls="--", lw=1.0, label=r"threshold $(1+\gamma)\bar\epsilon$")
    ax.axhline(lb, color=INK, ls=":", lw=1.0, label=r"$\Phi_{\mathcal{V}}-(1+\gamma)\bar\epsilon$")
    ax.axvline(t[-1], color=MUTED, lw=0.8)
    ax.text(t[-1] / 1.15, ymax / 1.4, r"decision at $t_{\rm det}$", color=INK, fontsize=6.5, ha="right", va="top")
    ax.text(t[-1] / 1.15, ymax / 2.4, "breach declared", color=ORANGE, fontsize=7, ha="right", va="top")
    ax.text(t[-1] / 1.15, thr / 4, "no breach", color=BLUE, fontsize=7, ha="right", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"detection time $t$")
    ax.set_ylabel(r"max. retained $|$BDPM$|$")
    ax.legend(fontsize=6.3, loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(); fig.savefig("theory_single_instance.png", dpi=300); plt.close(fig)
    print("saved theory_ia_map.png, theory_single_instance.png")

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "calib"
    {"calib": calib, "exp12": exp12, "phiV": phiV, "analyze12": analyze12, "exp3": exp3, "single": single, "plot": plot}[stage]()
