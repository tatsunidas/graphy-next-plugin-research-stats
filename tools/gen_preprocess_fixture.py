"""前処理の統計検定の参照値を scipy で出す（TypeScript の src/core/stats とは独立）。
Shapiro–Wilk / Levene・Brown–Forsythe / t 検定 / Wilcoxon の符号順位 / Mann–Whitney と、
検定の選び方の規則を scipy の関数で書き直したものの結果を、test/fixtures/preprocess.json に書く。
使い方: python3 tools/gen_preprocess_fixture.py   （固定シード。再実行しても同じ結果）
"""
import json, warnings
import numpy as np
import scipy, scipy.stats as st

warnings.simplefilter("ignore")
out = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__}}
R = np.random.default_rng(20261125)


def fin(v):
    v = float(v)
    return v if np.isfinite(v) else None


def sample(kind, n):
    if kind == "normal":
        return R.normal(10, 3, n)
    if kind == "lognormal":
        return R.lognormal(0, 1, n)
    if kind == "uniform":
        return R.uniform(0, 5, n)
    if kind == "outlier":
        v = R.normal(0, 1, n)
        v[0] = 12.0
        return v
    if kind == "ties":
        return np.round(R.normal(5, 2, n))
    if kind == "exp":
        return R.exponential(2, n)
    raise ValueError(kind)


KINDS = ["normal", "lognormal", "uniform", "outlier", "ties", "exp"]
SIZES = [3, 4, 5, 6, 7, 11, 12, 13, 20, 50, 51, 200, 1000]

# --- Shapiro–Wilk
shap = []
for n in SIZES:
    for kind in KINDS:
        x = sample(kind, n)
        if len(np.unique(x)) < 2:
            continue
        r = st.shapiro(x)
        shap.append({"kind": kind, "x": x.tolist(), "w": fin(r.statistic), "p": fin(r.pvalue)})
for n in (3, 5, 40):
    shap.append({"kind": "constant", "x": [2.5] * n, "w": None, "p": None})
for n in (2000, 5000):
    x = sample("lognormal", n)
    r = st.shapiro(x)
    shap.append({"kind": "lognormal", "x": x.tolist(), "w": fin(r.statistic), "p": fin(r.pvalue)})
out["shapiro"] = shap

# --- 2 群の素材（大きさ・分散・分布を変える）
PAIRS = []
for n1, n2 in [(3, 3), (4, 6), (5, 5), (8, 12), (12, 12), (20, 15), (30, 30), (50, 51), (100, 40), (200, 200), (7, 25), (60, 9)]:
    for k, (sd1, sd2, shift) in enumerate([(1, 1, 0), (1, 3, 0.5), (2, 2, 1.5), (1, 6, 0)]):
        PAIRS.append((R.normal(0, sd1, n1), R.normal(shift, sd2, n2), f"normal-{n1}-{n2}-{k}"))
for n1, n2 in [(10, 10), (25, 40), (60, 60)]:
    PAIRS.append((sample("lognormal", n1), sample("lognormal", n2) * 1.5, f"lognormal-{n1}-{n2}"))
    PAIRS.append((sample("ties", n1), sample("ties", n2) + 1, f"ties-{n1}-{n2}"))
    PAIRS.append((sample("outlier", n1), sample("normal", n2) / 3, f"outlier-{n1}-{n2}"))

# --- Levene（2 群と 3 群、median と mean）
lev = []
for x, y, tag in PAIRS:
    for c in ("median", "mean"):
        r = st.levene(x, y, center=c)
        lev.append({"tag": tag, "center": c, "groups": [x.tolist(), y.tolist()], "f": fin(r.statistic), "p": fin(r.pvalue)})
for n in (5, 10, 30, 100):
    for c in ("median", "mean"):
        g = [R.normal(0, 1, n), R.normal(0, 2, n + 3), R.lognormal(0, 1, n - 2 if n > 5 else n)]
        r = st.levene(*g, center=c)
        lev.append({"tag": f"k3-{n}", "center": c, "groups": [a.tolist() for a in g], "f": fin(r.statistic), "p": fin(r.pvalue)})
# 群の中の偏差がすべて同じ → F が定まらない（scipy は nan/inf）
lev.append({"tag": "degenerate", "center": "median", "groups": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], "f": None, "p": None})
out["levene"] = lev

# --- t 検定（2 標本 Student / Welch）
tt = []
for x, y, tag in PAIRS:
    for ev in (True, False):
        if not ev and (len(x) < 2 or len(y) < 2):
            continue
        r = st.ttest_ind(x, y, equal_var=ev)
        df = fin(r.df)
        tt.append({"tag": tag, "equalVar": ev, "x": x.tolist(), "y": y.tolist(), "t": fin(r.statistic), "df": df, "p": fin(r.pvalue)})
for ev in (True, False):
    # 分散 0 の 2 群は t が定まらない（scipy は nan）。df は Student だけ定義できる
    tt.append({"tag": "const", "equalVar": ev, "x": [1.0, 1.0, 1.0], "y": [1.0, 1.0, 1.0, 1.0], "t": None, "df": 5.0 if ev else None, "p": None})
out["ttestInd"] = tt

# --- 対応あり
def paired_sets():
    s = []
    for n in (3, 4, 5, 8, 12, 20, 30, 50, 51, 120, 400):
        for kind in ("normal", "lognormal", "ties", "outlier"):
            base = R.normal(10, 2, n)
            if kind == "normal":
                after = base + R.normal(0.5, 1, n)
            elif kind == "lognormal":
                after = base + R.lognormal(-0.5, 1, n)
            elif kind == "ties":
                base = np.round(base)
                after = np.round(base + R.normal(0.4, 1.2, n))
            else:
                after = base + R.normal(0.2, 0.5, n)
                after[0] += 9
            s.append((base, after, f"{kind}-{n}"))
    return s

pt = []
for x, y, tag in paired_sets():
    r = st.ttest_rel(x, y)
    pt.append({"tag": tag, "x": x.tolist(), "y": y.tolist(), "t": fin(r.statistic), "df": len(x) - 1, "p": fin(r.pvalue)})
pt.append({"tag": "const-diff", "x": [3.0, 4.0, 5.0], "y": [3.0, 4.0, 5.0], "t": None, "df": 2, "p": None})
out["ttestRel"] = pt

# --- Wilcoxon（method="auto", zero_method="wilcox", correction=False）
wc = []
def add_w(x, y, tag):
    try:
        r = st.wilcoxon(x, y, zero_method="wilcox", correction=False, method="auto")
        wc.append({"tag": tag, "x": list(map(float, x)), "y": list(map(float, y)), "stat": fin(r.statistic), "p": fin(r.pvalue)})
    except ValueError:
        wc.append({"tag": tag, "x": list(map(float, x)), "y": list(map(float, y)), "stat": None, "p": None})

for x, y, tag in paired_sets():
    add_w(x, y, tag)
for n in (4, 8, 10, 13, 14, 20, 30, 49, 50, 51, 60):
    base = R.normal(0, 1, n)
    after = base + R.normal(0.3, 1, n)
    k = max(1, n // 4)
    after[:k] = base[:k]  # 差が 0 のペア
    add_w(base, after, f"zeros-{n}")
for n in (6, 9, 13, 14, 25):
    base = np.round(R.normal(0, 2, n))
    after = np.round(base + R.normal(0.5, 1.5, n))
    add_w(base, after, f"roundtrip-{n}")
# 差がすべて 0 のとき scipy は p = 1 を返すが、検定する対象が無いので TypeScript 側は「計算できない」とする（意図した違い）
wc.append({"tag": "all-zero", "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "stat": None, "p": None, "intended": "no nonzero differences"})
add_w([5.0, 6.0, 7.0, 8.0], [4.0, 5.0, 6.0, 7.0], "all-same-diff")
out["wilcoxon"] = wc

# --- Mann–Whitney（asymptotic: 正規近似・同順位補正・連続性補正）
mw = []
for x, y, tag in PAIRS:
    r = st.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    mw.append({"tag": tag, "x": x.tolist(), "y": y.tolist(), "u1": fin(r.statistic), "p": fin(r.pvalue)})
out["mannwhitney"] = mw

# --- 検定の選び方（規則を scipy の関数で書いたもの）
def pick_unpaired(x, y, a):
    def nrm(v):
        if len(v) < 3 or np.ptp(v) == 0:
            return False
        return st.shapiro(v).pvalue >= a
    if not (nrm(x) and nrm(y)):
        return "mannwhitney", float(st.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").pvalue)
    lv = st.levene(x, y, center="median").pvalue
    if np.isfinite(lv) and lv >= a:
        return "student", float(st.ttest_ind(x, y, equal_var=True).pvalue)
    return "welch", float(st.ttest_ind(x, y, equal_var=False).pvalue)

def pick_paired(x, y, a):
    d = np.asarray(x) - np.asarray(y)
    if len(d) >= 3 and np.ptp(d) > 0 and st.shapiro(d).pvalue >= a:
        return "paired-t", float(st.ttest_rel(x, y).pvalue)
    r = st.wilcoxon(x, y, zero_method="wilcox", correction=False, method="auto")
    return "wilcoxon", float(r.pvalue)

sel = {"unpaired": [], "paired": []}
for x, y, tag in PAIRS:
    for a in (0.05, 0.01):
        t, p = pick_unpaired(x, y, a)
        sel["unpaired"].append({"tag": tag, "alpha": a, "x": x.tolist(), "y": y.tolist(), "test": t, "p": fin(p)})
for x, y, tag in paired_sets():
    for a in (0.05, 0.01):
        try:
            t, p = pick_paired(x, y, a)
        except ValueError:
            continue
        sel["paired"].append({"tag": tag, "alpha": a, "x": x.tolist(), "y": y.tolist(), "test": t, "p": fin(p)})
out["select"] = sel

json.dump(out, open("test/fixtures/preprocess.json", "w"), allow_nan=False)
print("wrote", {k: (len(v) if isinstance(v, list) else {kk: len(vv) for kk, vv in v.items()}) for k, v in out.items() if k != "versions"}, out["versions"])
