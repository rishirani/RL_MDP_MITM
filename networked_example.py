"""Networked-control example (supplement, Section XI).

Plant: a sensor's transmit buffer b in {0,...,B} sending over a Gilbert-Elliott fading channel
g in {Good, Bad}; state x = (b, g), N = 2(B+1). The controller (agent) chooses the transmit power
u in {low, high} each slot. The state feedback (buffer occupancy and channel state, from ACKs)
travels over the network link that the MITM adversary intercepts.

Slot dynamics: the head-of-line packet is delivered w.p. S[g, u] if b > 0; a new packet arrives
w.p. LAM; b' = min(b - delivered + arrival, B) (arrivals at a full buffer are lost); the channel
then moves by the Gilbert-Elliott chain. Reward, evaluated by the downstream rewarder on the
reported next state:  r(x, u, x') = 1 - b'/B - KAPPA * 1{u = high}  (delay vs. energy), plus noise.

Attack objective: while spoofing the feedback, the adversary drives the plant with low power,
so the true buffer overflows (packet loss).

Stages (results saved; `plot` reads only saved files):
  python3 networked_example.py calib    # certificate rates, Pr[information advantage] vs t_agt / t_adv
  python3 networked_example.py detect   # FA / miss vs t_det at three operating points, eps-greedy policy
  python3 networked_example.py damage   # true overflow rate: nominal (agent's policy) vs. attack
  python3 networked_example.py plot
"""
import sys
import time

import numpy as np
from numba import njit, prange

B = 7
NB = B + 1
N, M = 2 * NB, 2
LAM = 0.55
S = np.array([[0.60, 0.90],      # delivery prob., Good channel: low, high power
              [0.20, 0.30]])     # Bad channel
GE = np.array([[0.90, 0.10],     # channel transitions G->G, G->B
               [0.30, 0.70]])    # B->G, B->B
KAPPA = 0.05
GAMMA = 0.5
SIGMA_W = 0.1
M_FRAC = 0.5
EPS_GREEDY = 0.1


def idx(b, g):
    return b * 2 + g


def build_mdp():
    P = np.zeros((N, M, N))
    R = np.zeros((N, M, N))
    for b in range(NB):
        for g in range(2):
            for u in range(M):
                s = idx(b, g)
                pd = S[g, u] if b > 0 else 0.0
                for d, pdv in ((1, pd), (0, 1 - pd)):
                    for a, pav in ((1, LAM), (0, 1 - LAM)):
                        bn = min(b - d + a, B)
                        for gn in range(2):
                            P[s, u, idx(bn, gn)] += pdv * pav * GE[g, gn]
                for sn in range(N):
                    R[s, u, sn] = 1.0 - (sn // 2) / B - KAPPA * u
    return P, R


P, R = build_mdp()
PC = np.cumsum(P, axis=-1)


def value_iteration(tol=1e-13):
    V = np.zeros(N)
    while True:
        Q = (P * (R + GAMMA * V[None, None, :])).sum(-1)
        Vn = Q.max(1)
        if np.abs(Vn - V).max() < tol:
            return (P * (R + GAMMA * Vn[None, None, :])).sum(-1), Vn
        V = Vn


QS, VS = value_iteration()
CT = R + GAMMA * VS[None, None, :]


@njit(cache=True)
def draw(pc_row):
    u = np.random.random()
    for k in range(pc_row.shape[0]):
        if u < pc_row[k]:
            return k
    return pc_row.shape[0] - 1


@njit(parallel=True, cache=True)
def agent_runs(n_runs, cps, gamma, PC, R, sigma_w, seed0):
    """Synchronous Polyak-Ruppert averaged Q-learning (step k^-0.75), generative model."""
    K = cps.shape[0]
    n, m = PC.shape[0], PC.shape[1]
    out = np.zeros((n_runs, K, n, m))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        Q = np.zeros((n, m)); Qn = np.zeros((n, m)); Qb = np.zeros((n, m))
        it = 0; c = 0
        while c < K:
            it += 1
            eta = it ** -0.75
            for s in range(n):
                for a in range(m):
                    xn = draw(PC[s, a])
                    vn = Q[xn, 0]
                    for b in range(1, m):
                        vn = max(vn, Q[xn, b])
                    Qn[s, a] = (1 - eta) * Q[s, a] + eta * (R[s, a, xn] + sigma_w * np.random.standard_normal() + gamma * vn)
            for s in range(n):
                for a in range(m):
                    Q[s, a] = Qn[s, a]
                    Qb[s, a] += (Q[s, a] - Qb[s, a]) / it
            while c < K and it * n * m >= cps[c]:
                out[r, c] = Qb
                c += 1
    return out


@njit(parallel=True, cache=True)
def adversary_runs(n_runs, cps, PC, seed0):
    """Transition counts of one uniform-action trajectory (Algorithm 2 input)."""
    K = cps.shape[0]
    n, m = PC.shape[0], PC.shape[1]
    out = np.zeros((n_runs, K, n, m, n))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        cnt = np.zeros((n, m, n))
        s = 0; t = 0; c = 0
        while c < K:
            a = np.random.randint(m)
            xn = draw(PC[s, a])
            cnt[s, a, xn] += 1.0
            s = xn; t += 1
            if t == cps[c]:
                out[r, c] = cnt
                c += 1
    return out


def p_hat(cnt):
    tot = cnt.sum(-1, keepdims=True)
    return np.where(tot > 0, cnt / np.maximum(tot, 1), 1.0 / N)


def phi_min(cnt):
    return np.abs(((P - p_hat(cnt)) * CT).sum(-1)).reshape(cnt.shape[0], cnt.shape[1], -1).min(-1)


@njit(parallel=True, cache=True)
def detect_traces(Qh, PCrep, R, gamma, sigma_w, eps, m_frac, cps, seed0):
    """Algorithm 1 under the agent's eps-greedy policy from Q_hat (per run). PCrep: (runs, N, M, N)
    cumulative reported kernel. Returns max retained |BDPM| at each checkpoint, pruning at m t."""
    n_runs = Qh.shape[0]
    n, m = Qh.shape[1], Qh.shape[2]
    K = cps.shape[0]
    out = np.zeros((n_runs, K))
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        V = np.zeros(n)
        G = np.zeros(n, dtype=np.int64)
        for x in range(n):
            V[x] = Qh[r, x, 0]
            for b in range(1, m):
                if Qh[r, x, b] > V[x]:
                    V[x] = Qh[r, x, b]; G[x] = b
        D = np.zeros((n, m)); T = np.zeros((n, m))
        s = 0; t = 0; c = 0
        mm = m_frac / (n * m)
        while c < K:
            a = np.random.randint(m) if np.random.random() < eps else G[s]
            xn = draw(PCrep[r, s, a])
            D[s, a] += Qh[r, s, a] - (R[s, a, xn] + sigma_w * np.random.standard_normal()) - gamma * V[xn]
            T[s, a] += 1.0
            s = xn; t += 1
            if t == cps[c]:
                best = 0.0
                for i in range(n):
                    for j in range(m):
                        if T[i, j] > mm * t:
                            best = max(best, abs(D[i, j]) / T[i, j])
                out[r, c] = best
                c += 1
    return out


def logspace_int(lo, hi, per_decade):
    k = int(round((np.log10(hi) - np.log10(lo)) * per_decade)) + 1
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), k)).astype(np.int64))


DELTA, RHO = 0.2, 0.5
CAL_RUNS = 400
T_ADV = 10**4
RATIOS = np.array([10, 100, 1000], dtype=np.int64)
D_RUNS = 200
D_CPS = logspace_int(1e3, 1e7, 4)


def calib():
    print("Q* =\n", QS.round(3))
    print("greedy policy (rows b=0..7, cols G,B; 1 = high):\n", QS.argmax(1).reshape(NB, 2))
    print("action gaps min:", np.abs(QS[:, 1] - QS[:, 0]).min().round(4))
    cpa = logspace_int(1e4, 1e8, 4)
    cpv = logspace_int(1e3, 1e6, 4)
    t0 = time.time()
    err = np.abs(agent_runs(CAL_RUNS, cpa, GAMMA, PC, R, SIGMA_W, 1) - QS).reshape(CAL_RUNS, len(cpa), -1).max(-1)
    print(f"agent {time.time()-t0:.0f}s")
    cnt = adversary_runs(4 * CAL_RUNS, cpv, PC, 10**6)
    pm = phi_min(cnt)
    unvisited = (cnt.sum(-1) == 0).reshape(cnt.shape[0], cnt.shape[1], -1).mean(-1).mean(0)
    print("frac. unvisited pairs vs t_adv:", dict(zip(cpv.tolist(), unvisited.round(3).tolist())))
    sa = np.polyfit(np.log(cpa), np.log(np.quantile(err, 1 - DELTA / 2, 0)), 1)[0]
    sv = np.polyfit(np.log(cpv[cpv >= 1e4]), np.log(np.quantile(pm[:, cpv >= 1e4], DELTA / 2, 0)), 1)[0]
    print(f"slopes: agent {sa:.3f}, adversary (t_adv >= 1e4) {sv:.3f}")
    Bh = (np.sqrt(cpa) * np.quantile(err, 1 - DELTA / 2, 0)).max()
    Ah = (np.sqrt(cpv[cpv >= 1e4]) * np.quantile(pm[:, cpv >= 1e4], DELTA / 2, 0)).min()
    print(f"B_hat={Bh:.3f} A_hat={Ah:.4f} C_hat={(2*(1+GAMMA)*Bh/Ah)**2:.3g}")
    k = int(np.where(cpv == T_ADV)[0][0])
    for ra in RATIOS:
        j = int(np.argmin(np.abs(cpa - ra * T_ADV)))
        pr = (pm[:, k][:, None] > 2 * (1 + GAMMA) * err[:, j][None, :]).mean()
        print(f"  t_adv={T_ADV} t_agt={cpa[j]:.2e}: Pr[IA] = {pr:.3f}")
    np.savez("net_calib.npz", t_agt=cpa, t_adv=cpv, err=err, phimin=pm, B=Bh, A=Ah)


def eps_greedy_freq(Q, P2, eps=EPS_GREEDY):
    """Asymptotic pair visit frequencies of the eps-greedy policy from Q under kernel P2 (N, M, N)."""
    g = Q.argmax(1)
    pol = np.full((N, M), eps / M); pol[np.arange(N), g] += 1 - eps
    K = (pol[:, :, None] * P2).sum(1)
    w, v = np.linalg.eig(K.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = pi / pi.sum()
    return pi[:, None] * pol


def visitV():
    """Visitation set V of the eps-greedy deployment policy and the certified slope for Phi_V (main paper,
    Definition 4 and Section VI). V: pairs with asymptotic frequency > m under the eps-greedy policy from Q*."""
    m = M_FRAC / (N * M)
    f = eps_greedy_freq(QS, P)
    V = f > m
    print(f"m = {m:.4f}; |V| = {V.sum()} of {N * M}; V is the greedy pairs of {V.sum()} states: "
          f"{bool((V.sum(1) <= 1).all() and (V.argmax(1)[V.any(1)] == QS.argmax(1)[V.any(1)]).all())}; "
          f"smallest frequency in V {f[V].min():.4f}, largest outside V {f[~V].max():.4f}")
    # does V change under the adversary's models (t_adv = T_ADV) and the agent's own greedy policies?
    cnt = adversary_runs(D_RUNS, np.array([T_ADV], dtype=np.int64), PC, 6 * 10**7)[:, 0]
    same_adv = np.mean([np.array_equal(eps_greedy_freq(QS, p_hat(cnt[r])) > m, V) for r in range(D_RUNS)])
    Qh = agent_runs(D_RUNS, RATIOS * T_ADV, GAMMA, PC, R, SIGMA_W, 5 * 10**7)
    for k, ra in enumerate(RATIOS):
        same = np.mean([np.array_equal(eps_greedy_freq(Qh[r, k], p_hat(cnt[r])) > m, V) for r in range(D_RUNS)])
        print(f"  t_agt/t_adv={ra}: V unchanged under the agent's eps-greedy policy and the adversary's model in {same:.3f} of runs")
    print(f"  V unchanged under the adversary's model (policy from Q*) in {same_adv:.3f} of runs")
    # robust visitation set: pairs retained under every adversary model and agent policy of the detection runs.
    # Phi_V is nondecreasing in V, so a subset of the visitation set gives a valid (conservative) certificate.
    Vr = V.copy()
    for r in range(D_RUNS):
        for k in range(len(RATIOS)):
            Vr &= eps_greedy_freq(Qh[r, k], p_hat(cnt[r])) > m
    fr = np.array([[eps_greedy_freq(Qh[r, k], p_hat(cnt[r]))[Vr].min() for k in range(len(RATIOS))]
                   for r in range(D_RUNS)])
    print(f"robust V: |V| = {Vr.sum()}; smallest frequency of a robust-V pair over all runs {fr.min():.4f} (m = {m:.4f})")
    V = Vr
    cal = dict(np.load("net_calib.npz"))
    cpa, cpv, err = cal["t_agt"], cal["t_adv"], cal["err"]
    cntc = adversary_runs(4 * CAL_RUNS, cpv, PC, 10**6)        # same seeds as calib
    phi = np.abs(((P - p_hat(cntc)) * CT).sum(-1))            # (runs, K, N, M)
    assert np.allclose(phi.reshape(phi.shape[0], phi.shape[1], -1).min(-1), cal["phimin"]), "seed mismatch"
    pv = np.where(V, phi, 0.0).reshape(phi.shape[0], phi.shape[1], -1).max(-1)
    sel = cpv >= 1e4
    sv = np.polyfit(np.log(cpv[sel]), np.log(np.quantile(pv[:, sel], DELTA / 2, 0)), 1)[0]
    Bh = float(cal["B"])
    Ah = (np.sqrt(cpv[sel]) * np.quantile(pv[:, sel], DELTA / 2, 0)).min()
    Cv = (2 * (1 + GAMMA) * Bh / Ah) ** 2
    print(f"Phi_V: slope {sv:.3f} (t_adv >= 1e4); B_hat={Bh:.3f} A_hat={Ah:.4f} C_hat_V={Cv:.4g} "
          f"(Phi_min: C_hat={float(cal['B'])**0 * (2 * (1 + GAMMA) * Bh / float(cal['A'])) ** 2:.3g}); "
          f"certified t_agt at t_adv={T_ADV}: {(1 + RHO) ** 2 * Cv * T_ADV:.3g}")
    k = int(np.where(cpv == T_ADV)[0][0]); prs = {}
    for ra in RATIOS:
        j = int(np.argmin(np.abs(cpa - ra * T_ADV)))
        prs[int(ra)] = (pv[:, k][:, None] > 2 * (1 + GAMMA) * err[:, j][None, :]).mean()
        print(f"  t_adv={T_ADV} t_agt={cpa[j]:.2e}: Pr[IA with Phi_V] = {prs[int(ra)]:.3f}")
    np.savez("net_visitV.npz", V=V, freq=f, A=Ah, C=Cv, slope=sv)


def detect():
    cal = dict(np.load("net_calib.npz"))
    t_agt = RATIOS * T_ADV
    Qh = agent_runs(D_RUNS, t_agt, GAMMA, PC, R, SIGMA_W, 5 * 10**7)
    cnt = adversary_runs(D_RUNS, np.array([T_ADV], dtype=np.int64), PC, 6 * 10**7)[:, 0]
    PCatt = np.cumsum(p_hat(cnt), -1)
    PCsec = np.broadcast_to(PC, (D_RUNS,) + PC.shape).copy()
    pm = np.abs(((P - p_hat(cnt)) * CT).sum(-1)).reshape(D_RUNS, -1).min(-1)
    res = dict(t=D_CPS, t_agt=t_agt, t_adv=T_ADV)
    for k, ta in enumerate(t_agt):
        Qk = np.ascontiguousarray(Qh[:, k])
        j = int(np.argmin(np.abs(cal["t_agt"] - ta)))
        eps_bar = (1 + RHO) * np.quantile(cal["err"][:, j], 1 - DELTA / 2)   # calibrated on independent runs
        thr = (1 + GAMMA) * eps_bar
        err = np.abs(Qk - QS).reshape(D_RUNS, -1).max(-1)
        t0 = time.time()
        sec = detect_traces(Qk, PCsec, R, GAMMA, SIGMA_W, EPS_GREEDY, M_FRAC, D_CPS, 7 * 10**7 + 1000 * k)
        att = detect_traces(Qk, np.ascontiguousarray(PCatt), R, GAMMA, SIGMA_W, EPS_GREEDY, M_FRAC, D_CPS, 8 * 10**7 + 1000 * k)
        fa, miss = (sec > thr).mean(0), (att <= thr).mean(0)
        ia = (pm > 2 * (1 + GAMMA) * eps_bar) & (err < eps_bar)
        res[f"fa{k}"], res[f"miss{k}"], res[f"thr{k}"], res[f"ia{k}"] = fa, miss, thr, ia.mean()
        print(f"  t_agt/t_adv={ta // T_ADV}: thr={thr:.3g}, Pr[IA event]={ia.mean():.3f}, Pr[err<eps_bar]={(err < eps_bar).mean():.3f} | "
              f"at t_det=1e5: FA={fa[D_CPS == 100000][0]:.3f} miss={miss[D_CPS == 100000][0]:.3f}; "
              f"t_det=1e7: FA={fa[-1]:.3f} miss={miss[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez("net_detect.npz", **res)


@njit(cache=True)
def overflow_rate(policy, PCdummy, T, seed):
    """Fraction of slots in which an arriving packet is lost (buffer full), true plant, given a
    deterministic power policy over states (policy[x] in {0,1})."""
    np.random.seed(seed)
    b = 0; g = 0; lost = 0
    for _ in range(T):
        u = policy[b * 2 + g]
        if b > 0 and np.random.random() < S[g, u]:
            b -= 1
        if np.random.random() < LAM:
            if b == B:
                lost += 1
            else:
                b += 1
        g = 0 if np.random.random() < GE[g, 0] else 1
    return lost / T


def damage():
    pol = QS.argmax(1).astype(np.int64)
    nom = overflow_rate(pol, PC, 10**7, 1)
    att = overflow_rate(np.zeros(N, dtype=np.int64), PC, 10**7, 2)
    print(f"packet-loss rate (lost arrivals per slot): nominal greedy policy {nom:.4f}, attack (low power) {att:.4f}")
    np.save("net_damage.npy", np.array([nom, att]))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED = "#0b0b0b", "#898781"
    COLS = ["#9ec5f4", "#3987e5", "#0d366b"]          # sequential: more agent training -> darker
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "mathtext.fontset": "cm"})
    d = dict(np.load("net_detect.npz"))
    t = d["t"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
    for k, ra in enumerate(d["t_agt"] // d["t_adv"]):
        axs[0].plot(t, d[f"fa{k}"], color=COLS[k], lw=1.4)
        axs[1].plot(t, d[f"miss{k}"], color=COLS[k], lw=1.4, label=rf"$t_{{\rm agt}}={ra}\,t_{{\rm adv}}$")
    for ax, title in zip(axs, ("false-alarm rate", "miss rate")):
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xlabel(r"detection time $t_{\rm det}$")
    axs[1].legend(fontsize=6.3, loc="center right", handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(); fig.savefig("net_detection.png", dpi=300); plt.close(fig)
    print("saved net_detection.png")


if __name__ == "__main__":
    {"calib": calib, "visitV": visitV, "detect": detect, "damage": damage, "plot": plot}[sys.argv[1]]()
