import sys, os, time, json, pickle, numpy as np, torch
from scipy.stats import wasserstein_distance
from sklearn.neural_network import MLPClassifier
try:
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
except ImportError:
    from sbi.inference import SNPE as NPE
    from sbi.utils import posterior_nn
from sbi.utils import BoxUniform

# ---------------- settings (edit here) ----------------
ENGINE = sys.argv[1] if len(sys.argv) > 1 else "npe"
R      = 30            # 30 datasets

# =======================================================
# 1. تشغيل الزوج (3, 4) بناءً على طلب المحكم
LEVELS = (3, 4)        
# =======================================================
# 2. تقليل المحاكاات للسرعة (إذا كان جهازك قوياً، يمكنك زيادتها إلى 100,000)
N_SIMS = 50_000        
N_POST = 4000          
G      = 40             
Delta, n, BETA, RHO = 0.25, 20, 1, 100.0
LO = torch.tensor([0.5, -1.0, 0.15]); HI = torch.tensor([4.0, 2.0, 0.8])
dev = "cuda" if torch.cuda.is_available() else "cpu"

# =======================================================
# 3. تغيير البذرة العشوائية إلى 42 (البذرة الأولى كانت 0)
torch.manual_seed(42); np.random.seed(42)
# =======================================================

prior = BoxUniform(LO.to(dev), HI.to(dev))

# اسم ملف جديد ليحفظ المحركات الخاصة بـ 3,4 وبذرة 42
ENG_FILE = f"engines_{ENGINE}_{N_SIMS}_3-4_seed42.pkl"

def summaries(x):
    xp, xn = x[:, :-1], x[:, 1:]
    return torch.stack([xp.mean(1), xn.mean(1), (xp**2).mean(1), (xp*xn).mean(1), (xn**2).mean(1)], 1)

def sim_euler(theta, k):
    M = theta.shape[0]; K = int(Delta * 2**k); h = Delta / K
    kap, mu, sig = theta[:, 0], theta[:, 1], theta[:, 2]
    x = torch.zeros(M, n + 1, device=theta.device); cur = torch.zeros(M, device=theta.device)
    for i in range(n):
        for _ in range(K):
            cur = cur + kap*(mu - cur)*h + sig*h**0.5*torch.randn(M, device=theta.device)
        x[:, i + 1] = cur
    return x

def sim_exact(theta):
    M = theta.shape[0]; kap, mu, sig = theta[:, 0], theta[:, 1], theta[:, 2]
    a = torch.exp(-kap*Delta); sd = sig*torch.sqrt((1 - a**2)/(2*kap))
    x = torch.zeros(M, n + 1, device=theta.device)
    for i in range(n):
        x[:, i + 1] = mu + a*(x[:, i] - mu) + sd*torch.randn(M, device=theta.device)
    return x

cent = [np.linspace(float(LO[j]), float(HI[j]), 2*G + 1)[1::2] for j in range(3)]
TH = np.stack(np.meshgrid(*cent, indexing="ij"), -1).reshape(-1, 3)

def loglik_grid(x, level):
    k_, m_, s_ = TH.T; xp, xn = x[:-1][:, None], x[1:][:, None]
    if level is None:
        mm = m_ + (xp - m_)*np.exp(-k_*Delta); vv = s_**2/(2*k_)*(1 - np.exp(-2*k_*Delta))
    else:
        K = int(Delta*2**level); h = Delta/K; a = 1 - k_*h; aK = a**K
        mm = aK*(xp - m_) + m_; vv = s_**2*h*(1 - aK**2)/(1 - a**2)
    return (-0.5*np.log(2*np.pi*vv) - (xn - mm)**2/(2*vv)).sum(0)

def grid_post(x, level):
    l = loglik_grid(x, level); p = np.exp(l - l.max()); p /= p.sum(); return p.reshape(G, G, G)

def marginals(P): return [P.sum(axis=tuple(i for i in range(3) if i != j)) for j in range(3)]
def w1_samples(s, w, marg): return float(np.mean([wasserstein_distance(s[:, j], cent[j], w, marg[j]) for j in range(3)]))
def w1_grid(P, marg): return float(np.mean([wasserstein_distance(cent[j], cent[j], m, marg[j]) for j, m in enumerate(marginals(P))]))

def wquant(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o])/w.sum(); return v[o][min(np.searchsorted(c, q), len(v) - 1)]
def covered(s, w, truth):
    return float(np.mean([wquant(s[:, j], w, .025) <= truth[j] <= wquant(s[:, j], w, .975) for j in range(3)]))

def train_engine(k):
    print(f"Training NPE (MAF) for level h=2^-{k} (Seed 42)...")
    th = prior.sample((N_SIMS,)); x = summaries(sim_euler(th, k))
    inf = NPE(prior=prior, density_estimator=posterior_nn(model="maf", hidden_features=64, num_transforms=5), device=dev)
    inf.append_simulations(th, x).train(training_batch_size=512)
    return inf.build_posterior()

def draw(post, s_obs): return post.sample((N_POST,), x=s_obs, show_progress_bars=False).cpu().numpy()

def calibrate(sc, sf, n_clf=4):
    idx = np.random.permutation(len(sf)); A, B = idx[:len(sf)//2], idx[len(sf)//2:]
    X = np.vstack([sc[A], sf[A]]); y = np.r_[np.ones(len(A)), np.zeros(len(A))]
    lam = 0
    for r in range(n_clf):
        clf = MLPClassifier((32, 32), activation="tanh", max_iter=300, random_state=r).fit(X, y)
        p = np.clip(clf.predict_proba(sf[B])[:, 1], 1e-6, 1 - 1e-6)      
        lam = lam + np.log(p/(1 - p))/n_clf
    lam = lam - lam.mean()
    w = np.exp(np.clip(-lam/(2**BETA - 1), -0.5*np.log(RHO), 0.5*np.log(RHO)))
    return sf[B], w, float(w.sum()**2/(w**2).sum()/len(w))

if __name__ == "__main__":
    t0 = time.time()
    if os.path.exists(ENG_FILE):
        engines = pickle.load(open(ENG_FILE, "rb")); print("engines loaded from", ENG_FILE)
    else:
        engines = {k: train_engine(k) for k in LEVELS}
        try: pickle.dump(engines, open(ENG_FILE, "wb"))
        except Exception as e: print("could not save engines:", e)
        print(f"engines trained in {time.time()-t0:.0f}s", flush=True)
    
    t0 = time.time(); rows = []
    print(f"Running {R} datasets for level pair (3, 4) with Seed 42...")
    for r in range(R):
        th_star = prior.sample((1,)); x_obs = sim_exact(th_star)
        s_obs = summaries(x_obs); x_np = x_obs[0].cpu().numpy(); truth = th_star[0].cpu().numpy()
        marg0 = marginals(grid_post(x_np, None))
        
        kc, kf = 3, 4
        
        sc, sf = draw(engines[kc], s_obs), draw(engines[kf], s_obs)
        s_cal, w, ess = calibrate(sc, sf); one = np.ones(len(s_cal))
        Pc, Pf = grid_post(x_np, kc), grid_post(x_np, kf)
        Pr = np.clip(Pf**2/np.maximum(Pc, 1e-300), 0, None); Pr /= Pr.sum()
        rows.append(dict(r=r, kc=kc, kf=kf,
            W_coarse=w1_samples(sc, np.ones(len(sc)), marg0), W_fine=w1_samples(s_cal, one, marg0),
            W_calib=w1_samples(s_cal, w, marg0), O_coarse=w1_grid(Pc, marg0), O_fine=w1_grid(Pf, marg0),
            O_rich=w1_grid(Pr, marg0), cov_fine=covered(s_cal, one, truth), cov_calib=covered(s_cal, w, truth), ESS=ess))
        print(f"dataset {r+1}/{R}  {time.time()-t0:.0f}s", flush=True)
        
    json.dump(rows, open(f"results_{ENGINE}_3_4_seed42.json", "w"))
    
    # طباعة النتائج النهائية
    rs = rows; f = lambda key: np.array([q[key] for q in rs])
    d = f("W_calib") - f("W_fine")
    print(f"\n=== {ENGINE}: coarse h=2^-3, fine h2=2^-4, R={len(rs)} (Seed 42) ===")
    print(f"engine W1 : coarse {f('W_coarse').mean():.4f} | fine {f('W_fine').mean():.4f} | ML-Cal {f('W_calib').mean():.4f}")
    print(f"oracle W1 : coarse {f('O_coarse').mean():.4f} | fine {f('O_fine').mean():.4f} | Richardson {f('O_rich').mean():.4f}")
    print(f"paired W1(ML-Cal) - W1(fine): {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} ; improved in {np.mean(d < 0):.0%}")
    print(f"ESS/N {f('ESS').mean():.2f} ; coverage fine {f('cov_fine').mean():.3f} | ML-Cal {f('cov_calib').mean():.3f}")