"""Intermittent attacks (supplement, Theorem 11): numerical check on the Section VII single instance.

Uses the fixed Q_hat and P_hat of theory_validation.single (reproduced from the same seeds). The adversary
attacks during a fraction nu of each cycle, with cycle k of length CYCLE0 * k, so the number of switches up
to time t grows like sqrt(t) = o(t) (Assumption 3). The true plant is simulated alongside the reported
process: during an attack the adversary drives it at low power and reports next states from P_hat; when the
attack stops, the reported state jumps to the true state (a switching visit); during an attack-free period
the agent's actions drive the true plant and the true state is reported.

  python3 dynamic_attacks.py run     # BDPM limits (per-pair fractions) and Pr[breach] vs nu -> dynamic_attacks.npz
  python3 dynamic_attacks.py analyze # re-check of the per-pair limits (Definition 6) with z-tests -> dynamic_attacks.npz
  python3 dynamic_attacks.py plot    # dynamic_attacks.png
"""
import sys
import time

import numpy as np
from numba import njit, prange

import theory_validation as tv

N, M, GAMMA = tv.N, tv.M, tv.GAMMA
CYCLE0 = 100
NUS = np.round(np.linspace(0, 1, 21), 3)
RUNS = 50
T_DET = 10**7


def single_instance():
    """Q_hat and P_hat of theory_validation.single (same pools and seeds)."""
    Qs, Vs, C = tv.bellman_targets(GAMMA)
    cal = dict(np.load("theory_validation_exp12_partial.npz"))
    B, A, Cd = tv.certificates(cal, tv.E3_DELTA)
    t_agt = int(np.ceil((1 + tv.E3_RHO) ** 2 * Cd * tv.SI_T_ADV))
    eps_c = B / np.sqrt(t_agt)
    eps_bar = (1 + tv.E3_RHO) * eps_c
    Qh = tv.run_agent(tv.SI_POOL, np.array([t_agt], dtype=np.int64), GAMMA, 9 * 10**7)[:, 0]
    err = np.abs(Qh - Qs).reshape(tv.SI_POOL, -1).max(1)
    ia = np.where(err <= eps_c)[0][0]
    cnt = tv.run_adversary(tv.SI_POOL, np.array([tv.SI_T_ADV], dtype=np.int64), 9 * 10**7 + 5000)[:, 0]
    phiv = tv.phi_V(tv.phi_from_counts(cnt, C))
    iv = np.where(phiv > 2 * (1 + GAMMA) * eps_bar)[0][0]
    saved = dict(np.load("theory_validation_single.npz"))
    assert np.isclose(err[ia], saved["err"]) and np.isclose(phiv[iv], saved["phiV"]), "instance mismatch"
    return np.ascontiguousarray(Qh[ia]), tv.p_hat_from_counts(cnt[iv]), eps_bar


@njit(parallel=True, cache=True)
def intermittent_runs(Qh, pg_hat, pg, Rw, sigma_w, gamma, nu, cycle0, t_det, m, n_runs, seed0):
    """Algorithm 1 with uniform actions under an intermittent attack. pg_hat, pg: (N, M) Pr(next = Good).
    Returns per-run BDPM (N, M), visit counts, number of switching visits, and max retained |BDPM|."""
    bd = np.zeros((n_runs, N, M))
    vis = np.zeros((n_runs, N, M))
    nsw = np.zeros(n_runs)
    stat = np.zeros(n_runs)
    for r in prange(n_runs):
        np.random.seed(seed0 + r)
        V0 = max(Qh[0, 0], Qh[0, 1])
        V1 = max(Qh[1, 0], Qh[1, 1])
        D = np.zeros((N, M)); T = np.zeros((N, M))
        s = np.random.randint(N)          # reported state
        z = s                             # true plant state
        k = 1; pos = 0; sw = 0
        attacking = False
        for t in range(t_det):
            L = cycle0 * k
            on = pos < int(round(nu * L))
            a = np.random.randint(M)
            if on:
                xn = 0 if np.random.random() < pg_hat[s, a] else 1           # spoofed report
                z = 0 if np.random.random() < pg[z, 0] else 1                # plant at low power
            else:
                z = 0 if np.random.random() < pg[z, a] else 1                # plant under agent's action
                if attacking:
                    sw += 1                                                  # switching visit: jump to truth
                xn = z
            attacking = on
            v = V0 if xn == 0 else V1
            D[s, a] += Qh[s, a] - (Rw[s, a, xn] + sigma_w * np.random.standard_normal()) - gamma * v
            T[s, a] += 1.0
            s = xn
            if not on:
                z = s
            pos += 1
            if pos == L:
                pos = 0; k += 1
        best = 0.0
        for i in range(N):
            for j in range(M):
                if T[i, j] > 0:
                    bd[r, i, j] = D[i, j] / T[i, j]
                if T[i, j] > m * t_det:
                    best = max(best, abs(D[i, j]) / T[i, j])
        vis[r] = T; nsw[r] = sw; stat[r] = best
    return bd, vis, nsw, stat


def predicted_limit(Qh, ph, nu):
    """Theorem 11: Q_hat - Q* + gamma q_bar (v - v_hat)^T + nu Phi, q_bar = (1 - nu) p + nu p_hat."""
    Qs, Vs, C = tv.bellman_targets(GAMMA)
    vh = Qh.max(1)
    qbar = (1 - nu) * tv.P + nu * ph
    phi = ((tv.P - ph) * C).sum(-1)
    return Qh - Qs + GAMMA * (qbar * (Vs - vh)[None, None, :]).sum(-1) + nu * phi, phi


def stationary(P2):
    """Stationary distribution of the reported chain under uniform actions; P2: (N, M, N)."""
    K = P2.mean(1)
    w, v = np.linalg.eig(K.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def pair_fractions(ph, nu):
    """Per-pair attack fraction nu_{i,j} (Definition 6) for time fraction nu under uniform actions: the
    attacked share of visits to state i is nu pi_hat_i / (nu pi_hat_i + (1 - nu) pi_i)."""
    pi, pih = stationary(tv.P), stationary(ph)
    f = nu * pih / np.maximum(nu * pih + (1 - nu) * pi, 1e-300)
    return np.repeat(f[:, None], M, axis=1)


def realized_fraction(nu, t_det=T_DET, cycle0=CYCLE0):
    """Attacked share of the first t_det slots under the deterministic schedule (-> nu as t_det -> inf)."""
    t = 0; on = 0; k = 1
    while t < t_det:
        L = cycle0 * k
        n_on = int(round(nu * L))
        take = min(L, t_det - t)
        on += min(n_on, take)
        t += take; k += 1
    return on / t_det


def crossings(Qh, ph, eps_bar):
    """nu_c (predicted breach crossing) and nu* (smallest nu satisfying condition (39)), both computed with
    the per-pair attack fractions nu_{i,j} of Definition 6."""
    thr = (1 + GAMMA) * eps_bar
    _, phi = predicted_limit(Qh, ph, 1.0)
    grid = np.linspace(0, 1, 10001)
    mx = np.array([np.abs(predicted_limit(Qh, ph, pair_fractions(ph, g))[0]).max() for g in grid])
    mn = np.array([(pair_fractions(ph, g) * np.abs(phi)).max() for g in grid])   # max over V = all pairs
    nu_c = grid[np.argmax(mx > thr)]
    nu_star = grid[np.argmax(mn > (1 + GAMMA) * (eps_bar / (1 + tv.E3_RHO) + eps_bar))]
    return nu_c, nu_star, phi


def analyze():
    """Recompute Theorem 11 limits with the per-pair fractions nu_{i,j} from saved runs."""
    d = dict(np.load("dynamic_attacks.npz"))
    Qh, ph, eps_bar = single_instance()
    thr = (1 + GAMMA) * eps_bar
    nu_c, nu_star, _ = crossings(Qh, ph, eps_bar)
    print(f"pi = {stationary(tv.P).round(4).tolist()}, pi_hat = {stationary(ph).round(4).tolist()}")
    print(f"nu_c = {nu_c:.3f}, nu* = {nu_star:.3f}")
    MU, Z = [], []
    for q, nu in enumerate(d["nus"]):
        mu, _ = predicted_limit(Qh, ph, pair_fractions(ph, realized_fraction(nu)))
        MU.append(mu)
        bd = d["bd"][q]
        dev = np.abs(bd.mean(0) - mu).max()
        zz = (bd.mean(0) - mu) / (bd.std(0, ddof=1) / np.sqrt(bd.shape[0]))
        Z.append(zz.ravel()); z = np.abs(zz).max()
        print(f"  nu={nu:.2f} (realized {realized_fraction(nu):.4f}): max|mean BDPM - limit| = {dev:.2e}  (max |z| = {z:.1f})")
    from scipy import stats
    Z = np.concatenate(Z)
    print(f"{Z.size} z-scores: mean z^2 = {np.mean(Z**2):.3f}, chi2 p = {stats.chi2.sf(np.sum(Z**2), Z.size):.2f}, "
          f"KS p = {stats.kstest(Z, 'norm').pvalue:.2f}, max|z| = {np.abs(Z).max():.2f} "
          f"(Bonferroni p = {min(1, Z.size * 2 * stats.norm.sf(np.abs(Z).max())):.3f})")
    Zt = np.concatenate([((d["bd"][q].mean(0) - predicted_limit(Qh, ph, realized_fraction(nu))[0])
                          / (d["bd"][q].std(0, ddof=1) / np.sqrt(d["bd"].shape[1]))).ravel()
                         for q, nu in enumerate(d["nus"])])
    print(f"with the time fraction nu for every pair instead: mean z^2 = {np.mean(Zt**2):.3f}, max|z| = {np.abs(Zt).max():.2f}")
    d.update(mu=np.array(MU), nu_c=nu_c, nu_star=nu_star)
    np.savez("dynamic_attacks.npz", **d)


def run():
    Qh, ph, eps_bar = single_instance()
    thr = (1 + GAMMA) * eps_bar
    m = tv.M_FRAC / (N * M)
    nu_c, nu_star, phi = crossings(Qh, ph, eps_bar)
    print(f"threshold {thr:.3e}; Phi = {phi.round(5).tolist()}; per-pair attack fractions (Definition 6): "
          f"nu* (sufficient) = {nu_star:.3f}; nu_c (predicted crossing) = {nu_c:.3f}")
    res = dict(nus=NUS, thr=thr, nu_star=nu_star, nu_c=nu_c, phi=phi)
    BD, PR, MU, SW = [], [], [], []
    for q, nu in enumerate(NUS):
        t0 = time.time()
        bd, vis, nsw, stat = intermittent_runs(Qh, np.ascontiguousarray(ph[..., 0]), tv.P_GOOD, tv.R, tv.SIGMA_W,
                                                GAMMA, nu, CYCLE0, T_DET, m, RUNS, 10**9 + 1000 * q)
        mu, _ = predicted_limit(Qh, ph, pair_fractions(ph, realized_fraction(nu)))
        BD.append(bd); MU.append(mu); PR.append((stat > thr).mean()); SW.append(nsw.mean() / T_DET)
        dev = np.abs(bd.mean(0) - mu).max()
        se = (bd.std(0) / np.sqrt(RUNS)).max()
        print(f"  nu={nu:.2f}: Pr[breach]={PR[-1]:.2f}  max|mean BDPM - limit|={dev:.2e} (max s.e. {se:.1e})  "
              f"switching visits/t={SW[-1]:.1e} ({time.time()-t0:.0f}s)", flush=True)
    res.update(bd=np.array(BD), mu=np.array(MU), pr=np.array(PR), sw=np.array(SW))
    np.savez("dynamic_attacks.npz", **res)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"
    PAIRS = ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "mathtext.fontset": "cm"})
    d = dict(np.load("dynamic_attacks.npz"))
    nus, thr = d["nus"], float(d["thr"])
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.8))
    ax = axs[0]
    ax.axhspan(-thr, thr, color=BLUE, alpha=0.08, lw=0)
    names = ["(G,low)", "(G,high)", "(B,low)", "(B,high)"]
    for p in range(N * M):
        i, j = divmod(p, M)
        ax.plot(nus, d["mu"][:, i, j], color=PAIRS[p], lw=1.0)
        ax.plot(nus, d["bd"][:, :, i, j].mean(1), "o", ms=2.2, color=PAIRS[p], label=names[p])
    ax.set_xlabel(r"attack fraction $\nu$"); ax.set_ylabel(r"BDPM $\bar d_{i,j}$")
    ax.legend(fontsize=5.8, loc="lower left", handlelength=0.8, borderaxespad=0.1, labelspacing=0.2)
    ax = axs[1]
    ax.plot(nus, d["pr"], "o-", color=ORANGE, ms=2.5, lw=1.2)
    ax.axvline(float(d["nu_c"]), color=INK, ls="--", lw=0.9)
    ax.axvline(float(d["nu_star"]), color=INK, ls=":", lw=0.9)
    ax.text(float(d["nu_c"]) + 0.03, 0.45, r"$\nu_c$", ha="left", fontsize=7)
    ax.text(float(d["nu_star"]) + 0.02, 0.35, r"$\nu^\star$", ha="left", fontsize=7)
    ax.set_xlabel(r"attack fraction $\nu$"); ax.set_ylabel("Pr[breach declared]")
    ax.set_ylim(-0.03, 1.03)
    for a in axs:
        a.set_xticks([0, 0.5, 1])
    fig.tight_layout(); fig.savefig("dynamic_attacks.png", dpi=300); plt.close(fig)
    print("saved dynamic_attacks.png")


if __name__ == "__main__":
    {"run": run, "analyze": analyze, "plot": plot}[sys.argv[1]]()
