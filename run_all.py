"""
run_all.py  -  ملف واحد يشغّل كل شيء: التجربة ثم التحليل ثم يكتب التقرير.
التشغيل:   python run_all.py        (من Terminal داخل المجلد الذي فيه هذا الملف)
يمكن إيقافه بـ Ctrl+C وإعادة تشغيله لاحقاً بالأمر نفسه: يكمل من حيث وصل.
"""
# ======================= الإعدادات (غيّر هنا فقط إن لزم) =======================
MODE      = "sbi"            # "sbi" = التجربة الحقيقية ;  "exact" = اختبار سريع بمحرك دقيق
TRAIN     = "independent"    # "independent" = تدريب مستقل ;  "shared" = تدريب مشترك
PAIRS     = [(3, 4), (4, 5), (5, 6)]
SEEDS     = list(range(10))  # عشر بذور: 0 إلى 9
R         = 30               # عدد مجموعات البيانات لكل بذرة
NSIMS     = 100000           # عدد المحاكاات للتدريب
NPOST     = 20000            # عدد العيّنات من كل محرك
CLF_ITER  = 1000             # أقصى تكرار للمصنّف
N_CLF     = 4
MAX_EPOCHS, PATIENCE = 1000, 20
DATA_SEED = 20261113
OUT       = "E5_FINAL"       # مجلد النتائج (يُنشأ تلقائياً)
# ===============================================================================

import json, glob, time, traceback
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

DELTA, N_OBS, G = 0.25, 20, 40
LO = np.array([0.5, -1.0, 0.15]); HI = np.array([4.0, 2.0, 0.8])
BETA, RHO = 1, 100.0
AX = [LO[i] + (np.arange(G) + 0.5) * (HI[i] - LO[i]) / G for i in range(3)]
KK, MM, SS = np.meshgrid(*AX, indexing="ij")
EDGES = [np.linspace(LO[i], HI[i], G + 1) for i in range(3)]
XS = [np.linspace(LO[i], HI[i], 4001) for i in range(3)]


# ---------------------------------------------------------------- likelihoods / posteriors
def trans_params(kappa, mu, sigma, level):
    """Mean slope a, and variance v of one observation interval. level=None -> exact OU,
    level=k -> Euler with h=2^-k (K = Delta/h steps), exactly (Gaussian AR(1))."""
    if level is None:
        a = np.exp(-kappa * DELTA)
        v = sigma ** 2 * (1 - a ** 2) / (2 * kappa)
    else:
        h = 2.0 ** -level
        K = int(round(DELTA / h))
        r = 1 - kappa * h
        a = r ** K
        v = sigma ** 2 * h * (1 - r ** (2 * K)) / (1 - r ** 2)
    return a, v


def grid_posterior(path, level):
    ll = np.zeros_like(KK)
    for i in range(N_OBS):
        a, v = trans_params(KK, MM, SS, level)
        mean = MM + (path[i] - MM) * a
        ll += -0.5 * np.log(2 * np.pi * v) - (path[i + 1] - mean) ** 2 / (2 * v)
    p = np.exp(ll - ll.max())
    return p / p.sum()


def richardson_grid(p_h, p_h2):
    p = p_h2 ** 2 / np.maximum(p_h, 1e-300)
    return p / p.sum()


def sample_grid(p, N, rng):
    idx = rng.choice(p.size, size=N, p=p.ravel())
    ij = np.array(np.unravel_index(idx, p.shape)).T
    return LO + (ij + rng.random((N, 3))) * (HI - LO) / G


def grid_marginal_cdf(p, d):
    axes = tuple(j for j in range(3) if j != d)
    return np.interp(XS[d], EDGES[d], np.r_[0, np.cumsum(p.sum(axis=axes))])


def w1_samples_vs_grid(samples, weights, p):
    """mean over the 3 parameters of the marginal W1 between a (weighted) sample and a grid posterior"""
    out = []
    for d in range(3):
        o = np.argsort(samples[:, d])
        s = samples[o, d]
        w = np.ones(len(s)) if weights is None else weights[o]
        c = np.cumsum(w) / w.sum()
        Fn = np.r_[0, c][np.searchsorted(s, XS[d], side="right")]
        out.append(np.trapezoid(np.abs(grid_marginal_cdf(p, d) - Fn), XS[d]))
    return float(np.mean(out))


def w1_grid_vs_grid(p, q):
    return float(np.mean([np.trapezoid(np.abs(grid_marginal_cdf(p, d) - grid_marginal_cdf(q, d)), XS[d])
                          for d in range(3)]))


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(q, c, x[o])


def coverage(samples, weights, theta):
    w = np.ones(len(samples)) if weights is None else weights
    hit = [wquantile(samples[:, d], w, 0.025) <= theta[d] <= wquantile(samples[:, d], w, 0.975) for d in range(3)]
    return float(np.mean(hit))


# ---------------------------------------------------------------- data
def summaries(paths):
    x0, x1 = paths[:, :-1], paths[:, 1:]
    return np.stack([x0.sum(1), x1.sum(1), (x0 ** 2).sum(1), (x0 * x1).sum(1), (x1 ** 2).sum(1)], 1).astype(np.float32)


def simulate_scheme(theta, z, level):
    """paths of the Euler scheme at step 2^-level, observed every DELTA, x0=0 (law identical to
    running K Euler steps; z are the N x N_OBS standard normals of the observation intervals)"""
    a, v = trans_params(theta[:, 0], theta[:, 1], theta[:, 2], level)
    x = np.zeros((len(theta), N_OBS + 1))
    for i in range(N_OBS):
        x[:, i + 1] = theta[:, 1] + (x[:, i] - theta[:, 1]) * a + np.sqrt(v) * z[:, i]
    return x


def make_test_sets(cfg):
    rng = np.random.default_rng(cfg["data_seed"])
    sets = []
    for r in range(cfg["R"]):
        th = rng.uniform(LO, HI)
        a, v = trans_params(th[0], th[1], th[2], None)
        path = np.zeros(N_OBS + 1)
        for i in range(N_OBS):
            path[i + 1] = th[1] + (path[i] - th[1]) * a + np.sqrt(v) * rng.standard_normal()
        sets.append({"r": r, "theta": th, "path": path, "x_obs": summaries(path[None])[0]})
    return sets


def training_data(level, seed, cfg):
    """independent: separate simulation randomness per level; shared: same theta and same z"""
    key = (cfg["data_seed"] + 1) * 1_000_003 + seed * 1009 + (0 if cfg["train"] == "shared" else level)
    rng = np.random.default_rng(key)
    n = cfg["nsims"]
    theta = rng.uniform(LO, HI, size=(n, 3))
    z = rng.standard_normal((n, N_OBS))
    return theta.astype(np.float32), summaries(simulate_scheme(theta, z, level))


# ---------------------------------------------------------------- engines
class ExactEngine:
    def __init__(self, level, ref):
        self.level, self.ref = level, ref

    def sample(self, N, ds, rng):
        return sample_grid(self.ref[ds["r"]][self.level], N, rng)


class SbiEngine:
    def __init__(self, level, seed, cfg):
        import torch
        from sbi.inference import SNPE
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform
        self.torch = torch
        net_seed = seed * 1000 + (0 if cfg["train"] == "shared" else level)
        torch.manual_seed(net_seed)
        np.random.seed(net_seed)
        theta, x = training_data(level, seed, cfg)
        prior = BoxUniform(low=torch.tensor(LO, dtype=torch.float32), high=torch.tensor(HI, dtype=torch.float32))
        inf = SNPE(prior=prior, density_estimator=posterior_nn(model="maf", hidden_features=64, num_transforms=5))
        inf.append_simulations(torch.tensor(theta), torch.tensor(x))
        est = inf.train(max_num_epochs=cfg["max_epochs"], stop_after_epochs=cfg["patience"],
                        training_batch_size=256, show_train_summary=False)
        self.post = inf.build_posterior(est)
        try:  # training diagnostics; key names vary between sbi versions
            self.epochs = int(inf._summary["epochs_trained"][-1])
            self.val_loss = float(inf._summary["best_validation_log_prob"][-1])
        except Exception:
            self.epochs, self.val_loss = None, None

    def sample(self, N, ds, rng):
        x = self.torch.tensor(ds["x_obs"], dtype=self.torch.float32)
        return self.post.sample((N,), x=x, show_progress_bars=False).numpy().astype(np.float64)


# ---------------------------------------------------------------- calibration (Algorithm 1)
def calibrate(sc, sf, seed, cfg):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sf))
    A, B = idx[: len(sf) // 2], idx[len(sf) // 2:]
    X = np.vstack([sc[A], sf[A]])
    y = np.r_[np.ones(len(A)), np.zeros(len(A))]          # 1 = coarse, 0 = fine
    lam = 0.0
    for r in range(cfg["n_clf"]):
        clf = MLPClassifier((32, 32), activation="tanh", max_iter=cfg["clf_iter"], random_state=r).fit(X, y)
        p = np.clip(clf.predict_proba(sf[B])[:, 1], 1e-6, 1 - 1e-6)
        lam = lam + np.log(p / (1 - p)) / cfg["n_clf"]     # log pi_h / pi_h2 on fine draws
    lam = lam - lam.mean()                                  # centring on the held-out half
    w = np.exp(np.clip(-lam / (2 ** BETA - 1), -0.5 * np.log(RHO), 0.5 * np.log(RHO)))   # NOTE the minus sign
    return sf[B], w, float(w.sum() ** 2 / (w ** 2).sum() / len(w))


# ---------------------------------------------------------------- one (pair, seed)
def run_pair(pair, seed, getter, tests, ref, cfg):
    kc, kf = pair
    ec, ef = getter(kc), getter(kf)
    rows = []
    for ds in tests:
        rng = np.random.default_rng(cfg["data_seed"] + 7919 * seed + 31 * ds["r"] + kc)
        sc = ec.sample(cfg["npost"], ds, rng)
        sf = ef.sample(cfg["npost"], ds, rng)
        R0, pc, pf = ref[ds["r"]]["p0"], ref[ds["r"]][kc], ref[ds["r"]][kf]
        pr = richardson_grid(pc, pf)
        half, w, ess = calibrate(sc, sf, seed * 100003 + ds["r"], cfg)
        rows.append(dict(
            pair=f"{kc}_{kf}", seed=seed, r=ds["r"], theta_true=list(map(float, ds["theta"])),
            W_coarse=w1_samples_vs_grid(sc, None, R0),
            W_fine_full=w1_samples_vs_grid(sf, None, R0),
            W_fine_half=w1_samples_vs_grid(half, None, R0),
            W_calib=w1_samples_vs_grid(half, w, R0),
            O_coarse=w1_grid_vs_grid(pc, R0), O_fine=w1_grid_vs_grid(pf, R0), O_rich=w1_grid_vs_grid(pr, R0),
            E_SBI_fine=w1_samples_vs_grid(sf, None, pf), E_SBI_coarse=w1_samples_vs_grid(sc, None, pc),
            cov_fine=coverage(sf, None, ds["theta"]), cov_calib=coverage(half, w, ds["theta"]), ESS=ess))
    return rows


def analyze(folder):
    import numpy as np, pandas as pd
    from scipy import stats
    rows = []
    for f in sorted(glob.glob(f"{folder}/res_*.json")):
        rows += json.load(open(f))["rows"]
    if not rows:
        print("لا توجد نتائج للتحليل."); return
    df = pd.DataFrame(rows)
    df["diff"] = df.W_calib - df.W_fine_half
    df["improved"] = df["diff"] < 0
    rng = np.random.default_rng(0)

    def two_level_boot(sub, B=5000):
        seeds = sub.seed.unique(); g = {s: sub[sub.seed == s]["diff"].to_numpy() for s in seeds}
        out = []
        for _ in range(B):
            ss = rng.choice(seeds, len(seeds))
            out.append(np.mean([rng.choice(g[s], len(g[s])).mean() for s in ss]))
        return np.percentile(out, [2.5, 97.5])

    units = df.groupby(["pair", "seed"]).agg(
        W_fine_half=("W_fine_half", "mean"), W_calib=("W_calib", "mean"), diff=("diff", "mean"),
        improved=("improved", "mean"), E_SBI=("E_SBI_fine", "mean"), E_disc=("O_fine", "mean"),
        O_rich=("O_rich", "mean"), ESS=("ESS", "mean"), cov_fine=("cov_fine", "mean"),
        cov_calib=("cov_calib", "mean")).reset_index()
    units["ratio"] = units.E_SBI / units.E_disc
    units.to_csv(f"{folder}/summary_by_seed.csv", index=False)

    lines = ["=" * 70]
    for pair, u in units.groupby("pair"):
        n = len(u); d = u["diff"].to_numpy()
        lines.append(f"pair {pair}: seeds={n}  W_fine(half)={u.W_fine_half.mean():.4f}  W_calib={u.W_calib.mean():.4f}  "
                     f"oracle fine/rich={u.E_disc.mean():.4f}/{u.O_rich.mean():.4f}")
        if n >= 2:
            t = stats.ttest_1samp(d, 0.0)
            try: wp = stats.wilcoxon(d).pvalue
            except ValueError: wp = float("nan")
            lo, hi = two_level_boot(df[df.pair == pair])
            lines.append(f"   paired diff (calib - fine_half) = {d.mean():+.4f}  [two-level bootstrap 95%: {lo:+.4f}, {hi:+.4f}]")
            lines.append(f"   seed-level t-test p={t.pvalue:.4f} (df={n-1}); Wilcoxon p={wp:.4f}; seeds with diff<0: {(d<0).sum()}/{n}")
        else:
            lines.append(f"   paired diff = {d.mean():+.4f}  (بذرة واحدة فقط: لا اختبار إحصائي)")
        lines.append(f"   engine error E_SBI={u.E_SBI.mean():.4f} (= {u.ratio.mean():.1f} x E_disc); "
                     f"coverage {u.cov_fine.mean():.3f} -> {u.cov_calib.mean():.3f}; ESS/N={u.ESS.mean():.2f}")
    lines.append("=" * 70)
    if len(units) >= 5:
        rho = stats.spearmanr(units.ratio, units["diff"])
        lines.append(f"decision rule: Spearman(E_SBI/E_disc, harm) over {len(units)} units: rho={rho.statistic:+.2f}, p={rho.pvalue:.4f}")
    text = "\n".join(lines)
    print(text)
    open(f"{folder}/final_report.txt", "w", encoding="utf-8").write(text)
    print(f"\nتم حفظ التقرير في {folder}/final_report.txt وملخص البذور في {folder}/summary_by_seed.csv")


def main():
    cfg = dict(mode=MODE, train=TRAIN, R=R, nsims=NSIMS, npost=NPOST, clf_iter=CLF_ITER, n_clf=N_CLF,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, data_seed=DATA_SEED)
    pairs, seeds = PAIRS, SEEDS
    out = Path(OUT); out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, pairs=pairs, seeds=seeds), indent=1))
    print("config:", cfg, flush=True)

    tests = make_test_sets(cfg)
    levels = sorted({k for p in pairs for k in p})
    print("reference posteriors ...", flush=True)
    ref = {}
    for ds in tests:
        d = {"p0": grid_posterior(ds["path"], None)}
        for k in levels:
            d[k] = grid_posterior(ds["path"], k)
        ref[ds["r"]] = d

    t_start = time.time()
    for seed in seeds:
        todo = [p for p in pairs if not (out / f"res_{p[0]}_{p[1]}_{seed}.json").exists()]
        if not todo:
            print(f"seed {seed}: all done, skipping"); continue
        cache = {}

        def getter(k):
            if k not in cache:
                t0 = time.time()
                cache[k] = ExactEngine(k, ref) if cfg["mode"] == "exact" else SbiEngine(k, seed, cfg)
                print(f"  seed {seed}: engine level {k} ready ({time.time()-t0:.0f}s)", flush=True)
            return cache[k]

        for pair in todo:
            try:
                rows = run_pair(pair, seed, getter, tests, ref, cfg)
                meta = {k: getattr(cache.get(k), "epochs", None) for k in pair}
                (out / f"res_{pair[0]}_{pair[1]}_{seed}.json").write_text(json.dumps(dict(config=cfg, engine_epochs=meta, rows=rows)))
                d = np.mean([r["W_calib"] - r["W_fine_half"] for r in rows])
                print(f"seed {seed} pair {pair}: fine_half={np.mean([r['W_fine_half'] for r in rows]):.4f} "
                      f"calib={np.mean([r['W_calib'] for r in rows]):.4f} diff={d:+.4f}  "
                      f"(elapsed {(time.time()-t_start)/3600:.1f} h)", flush=True)
            except Exception:
                traceback.print_exc()

    print("\n====== انتهت التجربة. بدء التحليل ======")
    analyze(OUT)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()