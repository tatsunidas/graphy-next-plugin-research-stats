"""パイプライン（分割・品質管理・入れ子の交差検証）の参照値を、TypeScript 側とは別に書いて出す。
数値の部分は scipy（Mann–Whitney・Spearman）、scikit-learn（StandardScaler・AUC）、statsmodels（Logit・PHReg）を使う。
乱数（xoshiro128**）と分割の約束だけは src/core と同じ定義で書き直している（乱数の系列が違うと分割が比べられないため）。
使い方: python3 tools/gen_pipeline_fixture.py  → test/fixtures/pipeline.json
"""
import json
import numpy as np
import scipy, scipy.stats as st
import sklearn, statsmodels
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

M = 0xFFFFFFFF


class Rng:
    def __init__(self, seed):
        a = seed & M
        s = []
        for _ in range(4):
            a = (a + 0x9E3779B9) & M
            t = a ^ (a >> 16)
            t = (t * 0x21F0AAAD) & M
            t ^= t >> 15
            t = (t * 0x735A2D97) & M
            t ^= t >> 15
            s.append(t & M)
        self.s = s

    def u32(self):
        s = self.s
        rotl = lambda x, k: ((x << k) | (x >> (32 - k))) & M
        result = (rotl((s[1] * 5) & M, 7) * 9) & M
        t = (s[1] << 9) & M
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t
        s[3] = rotl(s[3], 11)
        return result

    def int(self, n):
        limit = 2**32 - (2**32 % n)
        x = self.u32()
        while x >= limit:
            x = self.u32()
        return x % n


def shuffle(ix, rng):
    a = list(ix)
    for i in range(len(a) - 1, 0, -1):
        j = rng.int(i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def folds(n, strata, K, rng):
    keys = sorted(set(strata)) if strata is not None else [0]
    out = [-1] * n
    pos = 0
    for key in keys:
        mem = [i for i in range(n) if (strata[i] if strata is not None else 0) == key]
        for i in shuffle(mem, rng):
            out[i] = pos % K
            pos += 1
    return out


def holdout(n, strata, frac, rng):
    keys = sorted(set(strata)) if strata is not None else [0]
    test = [False] * n
    for key in keys:
        mem = shuffle([i for i in range(n) if (strata[i] if strata is not None else 0) == key], rng)
        t = int(np.floor(len(mem) * frac + 0.5))  # JavaScript の Math.round と同じ（.5 は上へ）
        for q in range(t):
            test[mem[q]] = True
    return test


def impute(X, train, do):
    Xa = np.array([[np.nan if v is None else v for v in r] for r in X], dtype=float)
    m = Xa.shape[1]
    med, usable = [], []
    for j in range(m):
        col = Xa[train, j]
        pres = col[~np.isnan(col)]
        med.append(float(np.median(pres)) if do and len(pres) else None)
        usable.append(bool(len(pres) > 0) if do else bool(len(pres) == len(col) and len(col) > 0))
    out = Xa.copy()
    if do:
        for j in range(m):
            if med[j] is not None:
                out[np.isnan(out[:, j]), j] = med[j]
    return out, med, usable


def rank(Xi, usable, task, rows):
    ps = []
    for j in range(Xi.shape[1]):
        if not usable[j] or np.isnan(Xi[rows, j]).any():
            continue
        x = Xi[rows, j]
        if task["kind"] == "binary":
            y = task["y"][rows]
            if len(np.unique(x)) < 2:
                continue
            p = st.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
        else:
            if len(np.unique(x)) < 2:
                continue
            try:
                r = PHReg(task["time"][rows], x.reshape(-1, 1), status=task["event"][rows], ties="efron").fit(disp=0)
                p = float(r.pvalues[0])
            except Exception:
                continue
        if np.isfinite(p):
            ps.append((p, j))
    return [j for _, j in sorted(ps)]


def fit_topk(Xi, med, task, train, ranking, k):
    cols = ranking[:k]
    if not cols:
        return None
    sc = StandardScaler().fit(Xi[np.ix_(train, cols)])
    # StandardScaler は n で割る SD。src/core は n−1 なので、ここで n−1 に直す（定義の違いを明示して揃える）
    n = len(train)
    sd = np.sqrt(sc.var_ * n / (n - 1))
    if (sd <= 0).any():
        return None
    Z = (Xi[np.ix_(train, cols)] - sc.mean_) / sd
    if task["kind"] == "binary":
        try:
            res = sm.Logit(task["y"][train], sm.add_constant(Z, has_constant="add")).fit(disp=0, method="newton", maxiter=200, tol=1e-12)
        except Exception:
            return None
        if not res.mle_retvals["converged"] or np.abs(res.params).max() > 50:
            return None
        beta = res.params
        pred = lambda i: (None if np.isnan(Xi[i, cols]).any() else float(1 / (1 + np.exp(-(beta[0] + ((Xi[i, cols] - sc.mean_) / sd) @ beta[1:])))))
    else:
        try:
            res = PHReg(task["time"][train], Z, status=task["event"][train], ties="efron").fit(disp=0)
        except Exception:
            return None
        beta = res.params
        pred = lambda i: (None if np.isnan(Xi[i, cols]).any() else float(((Xi[i, cols] - sc.mean_) / sd) @ beta))
    return {"cols": cols, "predict": pred, "means": sc.mean_.tolist(), "sds": sd.tolist(), "medians": med}


def cindex(time, event, risk):
    conc = comp = 0.0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        for j in range(len(time)):
            if time[j] <= time[i]:
                continue
            comp += 1
            conc += 1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0.0)
    return None if comp == 0 else conc / comp


def score(task, rows, pred):
    ok = [(i, v) for i, v in zip(rows, pred) if v is not None]
    if task["kind"] == "binary":
        lab = [int(task["y"][i]) for i, _ in ok]
        if len(set(lab)) < 2:
            return None
        return float(roc_auc_score(lab, [v for _, v in ok]))
    ii = [i for i, _ in ok]
    return cindex(task["time"][ii], task["event"][ii], [v for _, v in ok])


def choose(X, task, T, spec, rng):
    fo = folds(len(T), None, spec["innerK"], rng)
    K = spec["kCandidates"]
    pooled = [[None] * len(T) for _ in K]
    per = [[] for _ in K]
    for f in range(spec["innerK"]):
        tin = [T[p] for p in range(len(T)) if fo[p] != f]
        pos = [p for p in range(len(T)) if fo[p] == f]
        tout = [T[p] for p in pos]
        Xi, med, us = impute(X, tin, spec["impute"])
        rk = rank(Xi, us, task, tin)
        for a, k in enumerate(K):
            m = fit_topk(Xi, med, task, tin, rk, k)
            pr = [None if m is None else m["predict"](i) for i in tout]
            for p, v in zip(pos, pr):
                pooled[a][p] = v
            per[a].append(None if m is None else score(task, tout, pr))
    scores = []
    for a in range(len(K)):
        if task["kind"] == "binary":
            scores.append(score(task, T, pooled[a]))
        else:
            v = [x for x in per[a] if x is not None]
            scores.append(sum(v) / len(v) if v else None)
    best = None
    for a, s in enumerate(scores):
        if s is None:
            continue
        if best is None or s > scores[best] + 1e-12 or (abs(s - scores[best]) <= 1e-12 and K[a] < K[best]):
            best = a
    return (None if best is None else K[best]), scores, fo


def nested(X, task, dev, spec, rng):
    rows, outer, inner, preds, pooled = [], [], [], [], []
    for r in range(spec["repeats"]):
        strata = None if spec["strata"] is None else spec["strata"]
        asg = folds(len(dev), strata, spec["outerK"], rng)
        outer.append(asg)
        inn = []
        pr = [None] * len(dev)
        for f in range(spec["outerK"]):
            T = [dev[p] for p in range(len(dev)) if asg[p] != f]
            pos = [p for p in range(len(dev)) if asg[p] == f]
            V = [dev[p] for p in pos]
            k, sc, fo = choose(X, task, T, spec, rng)
            inn.append(fo)
            row = {"k": k, "innerScores": sc, "cols": [], "score": None}
            if k is not None:
                Xi, med, us = impute(X, T, spec["impute"])
                m = fit_topk(Xi, med, task, T, rank(Xi, us, task, T), k)
                if m is not None:
                    row["cols"] = m["cols"]
                    v = [m["predict"](i) for i in V]
                    for p, x in zip(pos, v):
                        pr[p] = x
                    row["score"] = score(task, V, v)
            rows.append(row)
        inner.append(inn)
        preds.append(pr)
        pooled.append(score(task, dev, pr) if task["kind"] == "binary" else None)
    k, sc, fo = choose(X, task, dev, spec, rng)
    Xi, med, us = impute(X, dev, spec["impute"])
    fm = fit_topk(Xi, med, task, dev, rank(Xi, us, task, dev), k)
    final = {"k": k, "scores": sc, "folds": fo, "cols": fm["cols"], "means": fm["means"], "sds": fm["sds"], "medians": fm["medians"],
             "testPred": [fm["predict"](i) for i in range(len(X)) if i not in set(dev)]}
    return {"folds": rows, "outerAssign": outer, "innerAssign": inner, "pred": preds, "pooled": pooled, "final": final}


def binary_case(seed, n, m, signal, missing):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, m))
    z = sum(c * X[:, j] for j, c in enumerate(signal))
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    Xl = X.round(6).tolist()
    for _ in range(missing):
        Xl[int(r.integers(n))][int(r.integers(m))] = None
    return Xl, {"kind": "binary", "y": y}


def surv_case(seed, n, m, signal, missing):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, m))
    lp = sum(c * X[:, j] for j, c in enumerate(signal))
    t = r.exponential(1 / np.exp(lp))
    c = r.exponential(1.5, size=n)
    time = np.round(np.minimum(t, c) + 1e-3, 6)
    event = (t <= c).astype(int)
    Xl = X.round(6).tolist()
    for _ in range(missing):
        Xl[int(r.integers(n))][int(r.integers(m))] = None
    return Xl, {"kind": "survival", "time": time, "event": event}


out = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__}}

# 分割だけの参照
rng = Rng(2024)
strata = [i % 3 for i in range(23)]
out["splits"] = {"seed": 2024, "strata": strata, "folds5": folds(23, strata, 5, rng), "foldsPlain4": folds(23, None, 4, rng), "holdout30": holdout(23, strata, 0.3, rng)}

# Spearman（品質管理）
r0 = np.random.default_rng(7)
a = r0.normal(size=30); b = a * 0.5 + r0.normal(size=30) * 0.5; b[3] = np.round(b[3], 1)
ai = a.round(4).tolist(); bi = b.round(4).tolist(); ai[5] = None; bi[11] = None
mask = [i for i in range(30) if ai[i] is not None and bi[i] is not None]
out["spearman"] = {"x": ai, "y": bi, "rho": float(st.spearmanr([ai[i] for i in mask], [bi[i] for i in mask]).statistic)}

cases = []
for name, (X, task), spec_extra, seed in [
    ("binary", binary_case(11, 60, 10, [1.4, -1.0], 0), {"impute": False, "strata": "y"}, 101),
    ("binary-impute", binary_case(12, 60, 8, [1.2, 0, -0.9], 9), {"impute": True, "strata": "y"}, 102),
    ("survival", surv_case(13, 70, 8, [0.9, -0.6], 0), {"impute": False, "strata": None}, 103),
    ("survival-strat-impute", surv_case(14, 70, 6, [0.8], 6), {"impute": True, "strata": "event"}, 104),
]:
    n = len(X)
    rng = Rng(seed)
    lab = task["y"] if task["kind"] == "binary" else task["event"]
    test = holdout(n, lab.tolist(), 0.25, rng)
    dev = [i for i in range(n) if not test[i]]
    strata = None if spec_extra["strata"] is None else [int(lab[i]) for i in dev]
    spec = {"outerK": 5, "innerK": 4, "repeats": 2, "kCandidates": [1, 2, 4], "strata": strata, "impute": spec_extra["impute"]}
    res = nested(X, task, dev, spec, rng)
    tj = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in task.items()}
    cases.append({"name": name, "seed": seed, "X": X, "task": tj, "test": test, "dev": dev, "spec": spec, "expected": res})
out["cases"] = cases

with open("test/fixtures/pipeline.json", "w") as f:
    json.dump(out, f)
print("ok", [c["name"] for c in cases])
