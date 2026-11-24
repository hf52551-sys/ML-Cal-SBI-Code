#!/usr/bin/env python3
"""
Experiment B: dose-response of ML-Cal-SBI to a *synthetic* engine error (OU model).

Idea: take the exact scheme posteriors pi_h, pi_{h/2} (grid, closed-form Euler AR(1)
likelihood), replace the engine by "exact posterior + injected error of known size
eps", run the full calibration pipeline (classifier ensemble -> reweighting), and
record how the effect of calibration changes with r = E_SBI / E_disc.

Injected error (standardised coordinates z = (theta - m) / sd of pi_{h/2}):
    theta' = m + sd * ( z * (1 + 0.5*eps*v) + eps*u ),   u, v ~ N(0, I_3)
i.e. a random mean shift of ~eps posterior sd and a random rescaling.
  mode "independent": different (u, v) for the two levels (independent training)
  mode "shared":      same (u, v) for both levels (shared training randomness)
The x-axis r is MEASURED (marginal W1 of engine samples vs exact pi_{h/2}, divided by
E_disc = marginal W1(pi_{h/2}, pi_0)), not the nominal eps.

Usage (one command: runs the experiment, then the analysis + figure)
  python synthetic_engine_dose_response.py
Other options
  python synthetic_engine_dose_response.py run --out results.csv --datasets 30 --workers 4
  python synthetic_engine_dose_response.py analyze --out results.csv
  # final confirmation with the MLP classifier of the paper (slow):
  python synthetic_engine_dose_response.py run --clf mlp --datasets 10 --out results_mlp.csv

Output columns: dataset, rep, mode, eps, esbi, edisc, r, w1_fine, w1_cal, w1_rich_oracle,
                w1_h2_oracle, rel_change, ess_frac, snr
rel_change = (w1_cal - w1_fine) / w1_fine   (negative = calibration helps)
"""
import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

trap = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------------------- model
LO = np.array([0.5, -1.0, 0.15])   # (kappa, mu, sigma) box, uniform prior
HI = np.array([4.0, 2.0, 0.8])
DELTA, NOBS, NG = 0.25, 20, 40
EDGES = [np.linspace(LO[j], HI[j], NG + 1) for j in range(3)]
CENT = [0.5 * (e[1:] + e[:-1]) for e in EDGES]
KK, MM, SS = np.meshgrid(*CENT, indexing="ij")
TT = [np.linspace(LO[j], HI[j], 3001) for j in range(3)]


def simulate(theta, rng):
    k, mu, s = theta
    a = np.exp(-k * DELTA)
    v = s ** 2 / (2 * k) * (1 - np.exp(-2 * k * DELTA))
    x = np.zeros(NOBS + 1)
    for i in range(NOBS):
        x[i + 1] = mu + (x[i] - mu) * a + np.sqrt(v) * rng.standard_normal()
    return x


def sums(x):
    x0, y = x[:-1], x[1:]
    return dict(sx=x0.sum(), sy=y.sum(), sxx=(x0 ** 2).sum(), sxy=(x0 * y).sum(), syy=(y ** 2).sum())


def loglik(S, a, v):
    """Gaussian AR(1) log-likelihood on the grid: y | x ~ N(mu + a (x - mu), v)."""
    c = MM * (1 - a)
    ss = (S["syy"] + a ** 2 * S["sxx"] + NOBS * c ** 2
          - 2 * a * S["sxy"] - 2 * c * S["sy"] + 2 * a * c * S["sx"])
    return -0.5 * NOBS * np.log(2 * np.pi * v) - ss / (2 * v)


def ll_exact(S):
    a = np.exp(-KK * DELTA)
    v = SS ** 2 / (2 * KK) * (1 - np.exp(-2 * KK * DELTA))
    return loglik(S, a, v)


def ll_euler(S, h):
    K = int(round(DELTA / h))
    a1 = 1 - KK * h
    a = a1 ** K
    v = SS ** 2 * h * (1 - a1 ** (2 * K)) / (1 - a1 ** 2)
    return loglik(S, a, v)


class Post:
    """Grid posterior with 1-D marginals/CDFs and exact sampling (uniform within cells)."""

    def __init__(self, ll):
        p = np.exp(ll - ll.max())
        p /= p.sum()
        self.pf = p.ravel()
        self.marg = [p.sum(axis=(1, 2)), p.sum(axis=(0, 2)), p.sum(axis=(0, 1))]
        self.cdf = [np.concatenate([[0.0], np.cumsum(m)]) for m in self.marg]
        w = (HI - LO) / NG
        self.mean = np.array([(self.marg[j] * CENT[j]).sum() for j in range(3)])
        var = np.array([(self.marg[j] * (CENT[j] - self.mean[j]) ** 2).sum() for j in range(3)]) + w ** 2 / 12
        self.sd = np.sqrt(var)

    def sample(self, n, rng):
        idx = rng.choice(self.pf.size, size=n, p=self.pf)
        i, j, k = np.unravel_index(idx, (NG, NG, NG))
        base = np.stack([EDGES[0][i], EDGES[1][j], EDGES[2][k]], axis=1)
        return base + rng.random((n, 3)) * (HI - LO) / NG


# ----------------------------------------------------------------------------- metrics
def w1_post_post(p1, p2):
    d = []
    for j in range(3):
        f1 = np.interp(TT[j], EDGES[j], p1.cdf[j])
        f2 = np.interp(TT[j], EDGES[j], p2.cdf[j])
        d.append(trap(np.abs(f1 - f2), TT[j]))
    return float(np.mean(d))


def w1_samples_post(th, w, post):
    """Marginal W1 (averaged over the 3 parameters) between (weighted) samples and a grid posterior."""
    d = []
    for j in range(3):
        o = np.argsort(th[:, j])
        x = th[o, j]
        cw = np.arange(len(x) + 1, dtype=float) if w is None else np.concatenate([[0.0], np.cumsum(w[o])])
        cw /= cw[-1]
        fs = cw[np.searchsorted(x, TT[j], side="right")]
        fg = np.interp(TT[j], EDGES[j], post.cdf[j])
        d.append(trap(np.abs(fs - fg), TT[j]))
    return float(np.mean(d))


# ----------------------------------------------------------------------------- engine
def engine_draw(post, n, rng, m, sd, u, v, eps):
    """Draw n samples from the synthetic engine (exact posterior + injected error), inside the box."""
    out, have = [], 0
    while have < n:
        th = post.sample(int(1.3 * (n - have)) + 100, rng)
        z = (th - m) / sd
        th2 = m + sd * (z * (1 + 0.5 * eps * v) + eps * u)
        ok = np.all((th2 >= LO) & (th2 <= HI), axis=1)
        out.append(th2[ok])
        have += int(ok.sum())
    return np.concatenate(out)[:n]


# ----------------------------------------------------------------------------- classifier
def scale(th):
    return 2 * (th - LO) / (HI - LO) - 1


class Ensemble:
    def __init__(self, kind, M, seed):
        self.kind, self.M, self.seed, self.models = kind, M, seed, []

    def fit(self, th_coarse, th_fine):
        X = scale(np.concatenate([th_coarse, th_fine]))
        y = np.concatenate([np.ones(len(th_coarse)), np.zeros(len(th_fine))])  # class 1 = coarse
        rng = np.random.default_rng(self.seed)
        for m in range(self.M):
            if self.kind == "mlp":
                from sklearn.neural_network import MLPClassifier
                clf = MLPClassifier(hidden_layer_sizes=(32, 32), activation="tanh", alpha=1e-3,
                                    batch_size=1024, max_iter=60, tol=1e-4, n_iter_no_change=10,
                                    random_state=self.seed + m)
                clf.fit(X, y)
            else:  # "quad": logistic regression on degree-2 features (fast, exact for Gaussian ratios)
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import PolynomialFeatures
                b = rng.integers(0, len(X), len(X))
                clf = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                    LogisticRegression(C=100.0, max_iter=300))
                clf.fit(X[b], y[b])
            self.models.append(clf)

    def logit(self, th):
        X = scale(th)
        out = []
        for clf in self.models:
            if self.kind == "mlp":
                p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
                out.append(np.log(p / (1 - p)))
            else:
                out.append(clf.decision_function(X))
        return np.array(out)  # (M, n)


def calibrate(ens, S0, ev, rho=100.0, beta=1):
    L0 = ens.logit(S0)
    snr = float(L0.mean(0).std() / np.sqrt(L0.var(0).mean()))
    center = L0.mean()
    Lev = ens.logit(ev).mean(0)
    logw = np.clip(-(Lev - center) / (2 ** beta - 1), -0.5 * np.log(rho), 0.5 * np.log(rho))
    w = np.exp(logw)
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2) / len(w))
    return w, ess, snr


# ----------------------------------------------------------------------------- one dataset
def run_dataset(args_tuple):
    d, a = args_tuple
    h, h2 = a["h"], a["h"] / 2
    rng_d = np.random.default_rng([a["seed"], d])
    theta = LO + (HI - LO) * rng_d.random(3)
    x = simulate(theta, rng_d)
    S = sums(x)
    ll0, llh, llh2 = ll_exact(S), ll_euler(S, h), ll_euler(S, h2)
    post0, posth, posth2 = Post(ll0), Post(llh), Post(llh2)
    postR = Post(2 * llh2 - llh)           # pi_R  ∝  pi_{h/2}^2 / pi_h   (beta = 1)
    edisc = w1_post_post(posth2, post0)
    w1_rich = w1_post_post(postR, post0)
    m, sd = posth2.mean, posth2.sd
    rows = []
    for rep in range(a["reps"]):
        for mi, mode in enumerate(a["modes"]):
            for ei, eps in enumerate(a["eps"]):
                rng = np.random.default_rng([a["seed"], d, rep, mi, ei])
                u1, v1 = rng.standard_normal(3), rng.standard_normal(3)
                u2, v2 = (u1, v1) if mode == "shared" else (rng.standard_normal(3), rng.standard_normal(3))
                Sc = engine_draw(posth, a["ntrain"], rng, m, sd, u1, v1, eps)
                Sf = engine_draw(posth2, a["ntrain"], rng, m, sd, u2, v2, eps)
                S0 = engine_draw(posth2, a["nheld"], rng, m, sd, u2, v2, eps)
                ev = engine_draw(posth2, a["neval"], rng, m, sd, u2, v2, eps)
                ens = Ensemble(a["clf"], 4, int(rng.integers(1 << 30)))
                ens.fit(Sc, Sf)
                w, ess, snr = calibrate(ens, S0, ev)
                esbi = w1_samples_post(ev, None, posth2)
                w1f = w1_samples_post(ev, None, post0)
                w1c = w1_samples_post(ev, w, post0)
                rows.append(dict(dataset=d, rep=rep, mode=mode, eps=eps, esbi=esbi, edisc=edisc,
                                 r=esbi / edisc, w1_fine=w1f, w1_cal=w1c, w1_rich_oracle=w1_rich,
                                 w1_h2_oracle=edisc, rel_change=(w1c - w1f) / w1f, ess_frac=ess, snr=snr))
    return rows


# ----------------------------------------------------------------------------- analysis
def crossing(xs, ys):
    """First sign change - -> + of ys along increasing xs (linear interpolation in log r)."""
    for i in range(len(xs) - 1):
        if ys[i] < 0 <= ys[i + 1]:
            t = -ys[i] / (ys[i + 1] - ys[i])
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return np.nan


def binned(df, nb):
    q = np.unique(np.quantile(df["r"], np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(q, df["r"], side="right") - 1, 0, len(q) - 2)
    g = df.groupby(b)
    return g["r"].median().values, g["rel_change"].mean().values


def analyze(path, nb=8, nboot=500, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = {"independent": "tab:red", "shared": "tab:blue"}
    for mode, g in df.groupby("mode"):
        g = g[g["r"] > 0].reset_index(drop=True)
        xr, ym = binned(g, nb)
        rstar = crossing(xr, ym)
        units = g["dataset"].values
        ud = np.unique(units)
        idx_by = {u: np.where(units == u)[0] for u in ud}
        boots, lo_hi = [], []
        for _ in range(nboot):  # bootstrap over datasets (unit of replication)
            pick = rng.choice(ud, len(ud))
            sub = g.iloc[np.concatenate([idx_by[u] for u in pick])]
            x_b, y_b = binned(sub, nb)
            boots.append(crossing(x_b, y_b))
        boots = np.array(boots, dtype=float)
        ok = boots[~np.isnan(boots)]
        ci = (np.quantile(ok, 0.025), np.quantile(ok, 0.975)) if len(ok) > 20 else (np.nan, np.nan)
        print(f"\n[{mode}]  n={len(g)}   r* = {rstar:.2f}  (95% CI {ci[0]:.2f} - {ci[1]:.2f};"
              f" {len(ok)}/{nboot} bootstrap draws with a crossing)")
        q = np.unique(np.quantile(g["r"], np.linspace(0, 1, nb + 1)))
        b = np.clip(np.searchsorted(q, g["r"], side="right") - 1, 0, len(q) - 2)
        print("  r range            n   mean rel.change   frac improved")
        for k, gg in g.groupby(b):
            print(f"  {gg['r'].min():6.2f}-{gg['r'].max():6.2f}   {len(gg):5d}   "
                  f"{gg['rel_change'].mean():+8.3f}         {(gg['rel_change'] < 0).mean():.2f}")
        ax.scatter(g["r"], g["rel_change"], s=4, alpha=0.12, color=cols.get(mode, "gray"))
        ax.plot(xr, ym, "o-", color=cols.get(mode, "gray"), label=f"{mode} (r*≈{rstar:.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(1, color="gray", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r = E_{SBI}/E_{disc}$ (measured)")
    ax.set_ylabel(r"$(W_1^{cal}-W_1^{fine})/W_1^{fine}$   (<0: calibration helps)")
    ax.legend()
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_curve.png"
    fig.savefig(out, dpi=200)
    print(f"\nfigure saved to {out}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "run", "analyze"],
                    help="all (default) = run then analyze")
    ap.add_argument("--out", default="dose_response.csv")
    ap.add_argument("--datasets", type=int, default=30)
    ap.add_argument("--reps", type=int, default=2, help="independent error draws per dataset and eps")
    ap.add_argument("--pair", type=int, default=4, help="coarse step h = 2^-pair, fine h/2")
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7])
    ap.add_argument("--modes", nargs="+", default=["independent", "shared"])
    ap.add_argument("--clf", choices=["quad", "mlp"], default="quad")
    ap.add_argument("--ntrain", type=int, default=20000)
    ap.add_argument("--nheld", type=int, default=5000)
    ap.add_argument("--neval", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=20261124)
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze(a.out)
        return
    cfg = dict(h=2.0 ** -a.pair, seed=a.seed, reps=a.reps, modes=a.modes, eps=a.eps, clf=a.clf,
               ntrain=a.ntrain, nheld=a.nheld, neval=a.neval)
    t0, rows = time.time(), []
    jobs = [(d, cfg) for d in range(a.datasets)]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(run_dataset, jobs), 1):
                rows += r
                print(f"dataset {i}/{a.datasets} done  ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)
    else:
        for i, j in enumerate(jobs, 1):
            rows += run_dataset(j)
            print(f"dataset {i}/{a.datasets} done  ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"saved {a.out}")
    if a.cmd == "all":
        analyze(a.out)


if __name__ == "__main__":
    main()