"""生存解析の参照値を、別実装（statsmodels / scipy / 定義どおりの総当たり）で出す。
TypeScript 側の実装とは独立に書いてあり、test/fixtures/survival.json を作る。
使い方: python3 tools/gen_survival_fixture.py
"""
import json, sys
import numpy as np
import scipy, scipy.stats as st
import statsmodels
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.duration.hazard_regression import PHReg

out = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}}

def make(seed, n, tie_round):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 3))
    lam = np.exp(0.7 * x[:, 0] - 0.4 * x[:, 1])
    t = r.exponential(1 / lam)
    c = r.exponential(1.5, size=n)
    time = np.minimum(t, c)
    ev = (t <= c).astype(int)
    if tie_round:
        time = np.round(time, tie_round)
        time[time <= 0] = 10 ** -tie_round
    return time, ev, x

def cindex_bruteforce(time, ev, risk):
    conc = comp = 0.0
    n = len(time)
    for i in range(n):
        if ev[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:
                comp += 1
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] == risk[j]:
                    conc += 0.5
    return conc / comp

cases = []
for seed, n, rnd in [(1, 60, 1), (2, 100, 2), (3, 45, 1)]:
    time, ev, x = make(seed, n, rnd)
    case = {"seed": seed, "time": time.tolist(), "event": ev.tolist(), "x": x.tolist()}
    sf = SurvfuncRight(time, ev)
    case["km"] = {"times": sf.surv_times.tolist(), "surv": sf.surv_prob.tolist(), "se": sf.surv_prob_se.tolist(), "nrisk": sf.n_risk.tolist()}
    g2 = (x[:, 2] > np.median(x[:, 2])).astype(int)
    q = np.quantile(x[:, 2], [1 / 3, 2 / 3])
    g3 = (x[:, 2] > q[0]).astype(int) + (x[:, 2] > q[1]).astype(int)
    for name, g in (("lr2", g2), ("lr3", g3)):
        chi, p = survdiff(time, ev, g)
        case[name] = {"group": g.tolist(), "chi2": float(chi), "p": float(p)}
    for name, cols in (("cox1", [0]), ("cox3", [0, 1, 2])):
        m = PHReg(time, x[:, cols], status=ev, ties="efron")
        f = m.fit(disp=0)
        case[name] = {"cols": cols, "beta": f.params.tolist(), "se": f.bse.tolist(), "llf": float(f.llf), "ll0": float(m.loglike(np.zeros(len(cols))))}
        case[name]["risk"] = (x[:, cols] @ f.params).tolist()
        case[name]["cindex"] = cindex_bruteforce(time, ev, x[:, cols] @ f.params)
    # RMST: 平均は KM の階段関数の積分、SE はブートストラップ（式の実装とは別の経路）
    tau = float(np.quantile(time[ev == 1], 0.8))
    def rmst_of(tm, e):
        s = SurvfuncRight(tm, e)
        ts = np.concatenate([[0.0], s.surv_times]); sv = np.concatenate([[1.0], s.surv_prob])
        tot = 0.0
        for k in range(len(ts)):
            lo = ts[k]; hi = ts[k + 1] if k + 1 < len(ts) else np.inf
            a, b = max(lo, 0.0), min(hi, tau)
            if b > a:
                tot += sv[k] * (b - a)
        return tot
    rr = np.random.default_rng(100 + seed)
    boots = []
    for _ in range(2000):
        idx = rr.integers(0, n, n)
        boots.append(rmst_of(time[idx], ev[idx]))
    case["rmst"] = {"tau": tau, "value": rmst_of(time, ev), "bootSe": float(np.std(boots, ddof=1))}
    cases.append(case)
out["cases"] = cases

r = np.random.default_rng(9)
mw = []
for n1, n2, rnd in [(12, 15, None), (30, 25, 1), (8, 40, 0)]:
    a = r.normal(0, 1, n1); b = r.normal(0.5, 1, n2)
    if rnd is not None:
        a = np.round(a, rnd); b = np.round(b, rnd)
    res = st.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
    mw.append({"x": a.tolist(), "y": b.tolist(), "p": float(res.pvalue)})
out["mannwhitney"] = mw

ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986]
out["bh"] = {"p": ps, "q": st.false_discovery_control(ps, method="bh").tolist()}

dist = []
for x, df in [(0.5, 1), (3.84, 1), (10.0, 2), (25.0, 3), (1.0, 5), (60.0, 4)]:
    dist.append({"x": x, "df": df, "sf": float(st.chi2.sf(x, df))})
out["chi2"] = dist
out["normal"] = [{"z": z, "two": float(2 * st.norm.sf(abs(z))), "cdf": float(st.norm.cdf(z))} for z in (0.0, 0.5, 1.96, 3.0, 5.5, -2.2)]
out["quantile"] = [{"p": p, "z": float(st.norm.ppf(p))} for p in (0.975, 0.9, 0.5, 0.995)]
def clean(o):
    # NaN/Infinity は JSON にできない（statsmodels は最後の点の SE を NaN にする）。null で残す
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o

json.dump(clean(out), open("test/fixtures/survival.json", "w"), allow_nan=False)
print("wrote", len(cases), "cases", out["versions"])
