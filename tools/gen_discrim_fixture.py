"""判別（R2）の参照値を、別実装（scikit-learn / statsmodels / scipy / 定義どおりの総当たり）で出す。
TypeScript 側とは独立に書いてあり、test/fixtures/discrim.json を作る。
使い方: python3 tools/gen_discrim_fixture.py
"""
import json
import numpy as np
import scipy, scipy.stats as st
import sklearn, statsmodels
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve

out = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__}}

def make(seed, n, rnd):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    z = 1.1 * X[:, 0] - 0.8 * X[:, 1] + 0.1
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    if rnd is not None:
        X = np.round(X, rnd)
    return X, y

def delong_brute(s, y):
    P, N = s[y == 1], s[y == 0]
    psi = lambda a, b: 1.0 if a > b else (0.5 if a == b else 0.0)
    v10 = np.array([np.mean([psi(p, q) for q in N]) for p in P])
    v01 = np.array([np.mean([psi(p, q) for p in P]) for q in N])
    auc = v10.mean()
    var = v10.var(ddof=1) / len(P) + v01.var(ddof=1) / len(N)
    return float(auc), float(np.sqrt(var))

def best_cut(s, y, direction, crit):
    u = np.unique(s)
    best = None
    for a, b in zip(u[:-1], u[1:]):
        cut = (a + b) / 2
        pred = (s > cut) if direction == "higher" else (s < cut)
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum()); tn = int(((~pred) & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
        val = (tp + tn) / len(y) if crit == "accuracy" else tp / (tp + fn) + tn / (tn + fp) - 1
        if best is None or val > best[0] + 1e-12:
            best = (val, cut, tp, fp, tn, fn)
    return {"cut": best[1], "tp": best[2], "fp": best[3], "tn": best[4], "fn": best[5]}

def calib_bins(p, y, bins):
    n = len(p); b = min(bins, n); order = np.argsort(p, kind="stable"); res = []
    for k in range(b):
        idx = order[(k * n) // b: ((k + 1) * n) // b]
        if len(idx):
            res.append({"n": int(len(idx)), "mean": float(p[idx].mean()), "obs": float(y[idx].mean()), "x": int(y[idx].sum())})
    return res

def dca(p, y, pts):
    n = len(y); prev = y.mean(); r = []
    for pt in pts:
        tp = int(((p >= pt) & (y == 1)).sum()); fp = int(((p >= pt) & (y == 0)).sum()); w = pt / (1 - pt)
        r.append({"pt": pt, "model": tp / n - fp / n * w, "all": float(prev - (1 - prev) * w)})
    return r

def logit_fit(Z, y):
    m = sm.Logit(y, sm.add_constant(Z, has_constant="add")).fit(method="newton", tol=1e-12, maxiter=200, disp=0)
    return m

def loo_preds(X, y, k, inside):
    n = len(y); m = X.shape[1]
    def select(idx):
        ps = []
        for j in range(m):
            a = X[idx][y[idx] == 1][:, j]; b = X[idx][y[idx] == 0][:, j]
            ps.append(st.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue)
        return [int(j) for j in sorted(range(m), key=lambda j: (ps[j], j))[:k]]
    outside = select(np.arange(n)) if not inside else None
    preds = []
    for i in range(n):
        tr = np.array([t for t in range(n) if t != i])
        cols = outside if outside is not None else select(tr)
        mu = X[tr][:, cols].mean(0); sd = X[tr][:, cols].std(0, ddof=1)
        f = logit_fit((X[tr][:, cols] - mu) / sd, y[tr])
        z = (X[i, cols] - mu) / sd
        preds.append(float(1 / (1 + np.exp(-(f.params[0] + z @ f.params[1:])))))
    return preds

def tree(X, y, stop, min_node, max_depth, cands):
    nodes = []
    def grow(idx, depth, bl):
        n = len(idx); pos = int(y[idx].sum())
        label = bl if bl is not None else (1 if pos * 2 >= n else 0)
        rec = {"depth": depth, "n": n, "pos": pos, "split": None, "label": int(label)}
        nodes.append(rec)
        if n == 0: return
        if bl is not None:
            wrong = (n - pos) if bl == 1 else pos
            if wrong / n < stop or n < min_node or depth >= max_depth: return
        best = None
        for j in cands:
            vals = np.unique(X[idx, j])
            for a, b in zip(vals[:-1], vals[1:]):
                cut = (a + b) / 2
                for d in ("higher", "lower"):
                    pred = (X[idx, j] > cut) if d == "higher" else (X[idx, j] < cut)
                    correct = int(((pred) & (y[idx] == 1)).sum() + ((~pred) & (y[idx] == 0)).sum())
                    if best is None or correct > best[0]: best = (correct, j, d, cut, pred)
        if best is None: return
        baseline = max(pos, n - pos) if bl is None else (pos if bl == 1 else n - pos)
        if best[0] <= baseline: return
        rec["split"] = {"feature": int(best[1]), "direction": best[2], "cut": float(best[3])}
        pred = best[4]
        grow(idx[pred], depth + 1, 1)
        grow(idx[~pred], depth + 1, 0)
    grow(np.arange(len(y)), 0, None)
    return nodes

cases = []
for seed, n, rnd in [(1, 60, 1), (2, 45, None), (3, 80, 1)]:
    X, y = make(seed, n, rnd)
    c = {"seed": seed, "X": X.tolist(), "y": y.tolist()}
    c["auc"] = [float(roc_auc_score(y, X[:, j])) for j in range(4)]
    fpr, tpr, thr = roc_curve(y, X[:, 0], drop_intermediate=False)
    c["roc0"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    c["delong"] = [dict(zip(("auc", "se"), delong_brute(X[:, j], y))) for j in range(4)]
    c["cut"] = {}
    for crit in ("accuracy", "youden"):
        for d in ("higher", "lower"):
            c["cut"][f"{crit}-{d}"] = best_cut(X[:, 0], y, d, crit)
    Z = (X[:, :2] - X[:, :2].mean(0)) / X[:, :2].std(0, ddof=1)
    f = logit_fit(Z, y)
    c["logit"] = {"beta": f.params.tolist(), "se": f.bse.tolist(), "p": f.pvalues.tolist(), "ll": float(f.llf), "pred": f.predict(sm.add_constant(Z)).tolist()}
    p = np.array(c["logit"]["pred"])
    c["calib"] = calib_bins(p, y, 10)
    lp = np.log(p / (1 - p)); g = logit_fit(lp[:, None], y)
    c["calib_slope"] = {"intercept": float(g.params[0]), "slope": float(g.params[1])}
    pts = [round(0.05 * i, 2) for i in range(1, 20)]
    c["dca"] = dca(p, y, pts)
    c["loo_outside"] = loo_preds(X, y, 2, False)
    c["loo_inside"] = loo_preds(X, y, 2, True)
    c["tree"] = tree(X, y, 0.15, 5, 6, [0, 1, 2, 3])
    cases.append(c)
out["cases"] = cases

cp = []
for x, n in [(0, 10), (10, 10), (25, 27), (3, 20), (34, 48), (1, 2), (50, 100)]:
    ci = st.binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    cp.append({"x": x, "n": n, "lower": float(ci.low), "upper": float(ci.high), "ppf_lower": 0.0 if x == 0 else float(st.beta.ppf(0.025, x, n - x + 1))})
out["cp"] = cp
out["beta"] = [{"x": x, "a": a, "b": b, "cdf": float(st.beta.cdf(x, a, b))} for x, a, b in [(0.3, 2, 5), (0.5, 0.5, 0.5), (0.9, 30, 3), (0.05, 1, 40), (0.7, 12.5, 4.5)]]

# P16 のモンテカルロ（別の乱数・同じ定義）
X, y = make(7, 70, 1)
nodes = tree(X, y, 0.15, 5, 6, [0, 1, 2, 3])
def mc(X, y, iters, seed):
    r = np.random.default_rng(seed)
    # 木を再構築して各節点の混同行列を持つ
    def grow(idx, depth, bl):
        n = len(idx); pos = int(y[idx].sum())
        node = {"label": bl if bl is not None else (1 if pos * 2 >= n else 0), "split": None}
        if n == 0: return node
        if bl is not None:
            wrong = (n - pos) if bl == 1 else pos
            if wrong / n < 0.15 or n < 5 or depth >= 6: return node
        best = None
        for j in range(4):
            vals = np.unique(X[idx, j])
            for a, b in zip(vals[:-1], vals[1:]):
                cut = (a + b) / 2
                for d in ("higher", "lower"):
                    pred = (X[idx, j] > cut) if d == "higher" else (X[idx, j] < cut)
                    correct = int(((pred) & (y[idx] == 1)).sum() + ((~pred) & (y[idx] == 0)).sum())
                    if best is None or correct > best[0]: best = (correct, j, d, cut, pred)
        baseline = max(pos, n - pos) if bl is None else (pos if bl == 1 else n - pos)
        if best is None or best[0] <= baseline: return node
        pred = best[4]; yy = y[idx]
        tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum()); tn = int((~pred & (yy == 0)).sum()); fn = int((~pred & (yy == 1)).sum())
        node["split"] = {"j": best[1], "d": best[2], "cut": best[3], "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        node["pos"] = grow(idx[pred], depth + 1, 1); node["neg"] = grow(idx[~pred], depth + 1, 0)
        return node
    root = grow(np.arange(len(y)), 0, None)
    se_l, sp_l, ac_l = [], [], []
    def collect(nd, acc):
        if nd["split"]: acc.append(nd); collect(nd["pos"], acc); collect(nd["neg"], acc)
    allnodes = []; collect(root, allnodes)
    for _ in range(iters):
        par = {id(nd): (r.beta(nd["split"]["tp"] + 1, nd["split"]["fn"] + 1), r.beta(nd["split"]["tn"] + 1, nd["split"]["fp"] + 1)) for nd in allnodes}
        tp = fn = tn = fp = 0
        for i in range(len(y)):
            nd = root
            while nd["split"]:
                se, sp = par[id(nd)]
                go_pos = (r.random() < se) if y[i] == 1 else (r.random() >= sp)
                nd = nd["pos"] if go_pos else nd["neg"]
            if y[i] == 1: tp += nd["label"] == 1; fn += nd["label"] != 1
            else: fp += nd["label"] == 1; tn += nd["label"] != 1
        se_l.append(tp / (tp + fn)); sp_l.append(tn / (tn + fp)); ac_l.append((tp + tn) / len(y))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"sens": q(se_l), "spec": q(sp_l), "acc": q(ac_l)}
out["mc"] = {"X": X.tolist(), "y": y.tolist(), "ref": mc(X, y, 20000, 5)}
json.dump(out, open("test/fixtures/discrim.json", "w"), allow_nan=False)
print("wrote", len(cases), "cases", out["versions"])
