"""前処理の段（p 値による除外）の参照値を、scipy で独立に書いて出す。TypeScript 側（src/core/stats・pipeline）は使わない。
規則（Shapiro–Wilk・Levene（median）・Student／Welch／Mann–Whitney の選択、p ≥ しきい値で除外）を scipy の関数で書き直し、
同じ合成データ・同じ訓練の行（ここで層化 K 分割を作る: 外側 5 分割 × 2 反復、その 1 つの内側 4 分割、全体）で、
残る特徴量の一覧を、複数の（α、しきい値）で出す。対応ありは、前後の値の合成データで ttest_rel／wilcoxon の規則を同様に書く。
使い方: python3 tools/gen_preprocess_pipeline_fixture.py   （固定シード。再実行しても同じ結果）
  → test/fixtures/preprocess_pipeline.json
"""
import json, warnings
import numpy as np
import scipy, scipy.stats as st

warnings.simplefilter("ignore")
R = np.random.default_rng(20261126)
N0, N1 = 48, 48
y = np.array([0] * N0 + [1] * N1)
n = len(y)

cols, names = [], []


def add(kind, k, gen):
    for q in range(k):
        names.append(f"{kind}_{q + 1}")
        cols.append(gen())


add("shiftN", 8, lambda: np.where(y == 1, R.normal(1.2, 1, n), R.normal(0, 1, n)))
add("shiftLog", 4, lambda: np.exp(np.where(y == 1, R.normal(1.0, 0.8, n), R.normal(0, 0.8, n))))
add("noiseN", 20, lambda: R.normal(5, 2, n))
add("noiseLog", 8, lambda: R.lognormal(0, 1, n))
add("unequalVar", 4, lambda: np.where(y == 1, R.normal(0, 3, n), R.normal(0, 1, n)))
add("weak", 6, lambda: np.where(y == 1, R.normal(0.4, 1, n), R.normal(0, 1, n)))


def with_missing():
    v = np.where(y == 1, R.normal(1.0, 1, n), R.normal(0, 1, n))
    v[R.choice(n, 10, replace=False)] = np.nan
    return v


add("missing", 4, with_missing)
names.append("constant")
cols.append(np.full(n, 3.0))
X = np.column_stack(cols)
m = X.shape[1]


def pick_unpaired(a_, b_, alpha):
    def nrm(v):
        if len(v) < 3 or np.ptp(v) == 0:
            return False
        return st.shapiro(v).pvalue >= alpha
    if len(a_) == 0 or len(b_) == 0:
        return None, None
    if not (nrm(a_) and nrm(b_)):
        if np.ptp(np.concatenate([a_, b_])) == 0:
            return "mannwhitney", None
        return "mannwhitney", float(st.mannwhitneyu(a_, b_, alternative="two-sided", method="asymptotic").pvalue)
    lv = st.levene(a_, b_, center="median").pvalue
    if np.isfinite(lv) and lv >= alpha:
        return "student", float(st.ttest_ind(a_, b_, equal_var=True).pvalue)
    return "welch", float(st.ttest_ind(a_, b_, equal_var=False).pvalue)


def fin(v):
    return None if v is None or not np.isfinite(v) else float(v)


# --- 訓練の行（層化 K 分割）
def strat_folds(rows, k):
    rows = np.array(rows)
    assign = np.zeros(len(rows), dtype=int)
    for cls in (0, 1):
        idx = np.where(y[rows] == cls)[0]
        idx = R.permutation(idx)
        for q, i in enumerate(idx):
            assign[i] = q % k
    return rows, assign


train_sets = []  # (label, rows)
allrows = list(range(n))
train_sets.append(("final", allrows))
first_outer = None
for rep in range(2):
    rows, assign = strat_folds(allrows, 5)
    for f in range(5):
        tr = [int(i) for i in rows[assign != f]]
        train_sets.append((f"outer r{rep}f{f}", tr))
        if first_outer is None:
            first_outer = tr
rows, assign = strat_folds(first_outer, 4)
for f in range(4):
    train_sets.append((f"inner of outer r0f0, i{f}", [int(i) for i in rows[assign != f]]))

ALPHAS = [0.05, 0.01, 0.1]
COMBOS = [(0.05, 0.05), (0.05, 0.1), (0.01, 0.2), (0.1, 0.01), (0.05, 0.5), (0.1, 0.05)]

unp = []
for label, tr in train_sets:
    tr = np.array(tr)
    per_alpha = {}
    for a in ALPHAS:
        recs = []
        for j in range(m):
            a_ = X[tr[y[tr] == 0], j]
            b_ = X[tr[y[tr] == 1], j]
            a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]
            t, p = pick_unpaired(a_, b_, a)
            recs.append({"test": t, "p": fin(p)})
        per_alpha[str(a)] = recs
    kept = {}
    for a, thr in COMBOS:
        kept[f"{a}|{thr}"] = [j for j, r in enumerate(per_alpha[str(a)]) if r["p"] is not None and r["p"] < thr]
    unp.append({"label": label, "rows": [int(i) for i in tr], "perAlpha": per_alpha, "kept": kept})

# --- 対応あり（前後の値）
NP = 40
fb, fa, fnames = [], [], []


def addp(kind, k, gen):
    for q in range(k):
        fnames.append(f"{kind}_{q + 1}")
        b, a = gen()
        fb.append(b)
        fa.append(a)


def pair_norm_shift():
    b = R.normal(10, 3, NP)
    return b, b + R.normal(2.0, 1, NP)


def pair_norm_noise():
    b = R.normal(10, 3, NP)
    return b, b + R.normal(0, 1, NP)


def pair_skew_shift():
    b = R.lognormal(1, 0.6, NP)
    return b, b + R.lognormal(0, 0.8, NP)


def pair_skew_noise():
    b = R.lognormal(1, 0.6, NP)
    d = R.standard_t(2, NP)  # 重い裾（符号は対称）
    return b, b + d


def pair_ties():
    b = np.round(R.normal(10, 3, NP))
    return b, b + np.round(R.normal(0.2, 1.5, NP))


addp("shiftN", 6, pair_norm_shift)
addp("noiseN", 12, pair_norm_noise)
addp("shiftSkew", 4, pair_skew_shift)
addp("noiseHeavy", 8, pair_skew_noise)
addp("ties", 4, pair_ties)
B = np.column_stack(fb)
A = np.column_stack(fa)


def pick_paired(x, y_, alpha):
    d = np.asarray(x) - np.asarray(y_)
    if len(d) >= 3 and np.ptp(d) > 0 and st.shapiro(d).pvalue >= alpha:
        return "paired-t", float(st.ttest_rel(x, y_).pvalue)
    try:
        return "wilcoxon", float(st.wilcoxon(x, y_, zero_method="wilcox", correction=False, method="auto").pvalue)
    except ValueError:
        return "wilcoxon", None


pair_alpha = {}
for a in ALPHAS:
    pair_alpha[str(a)] = []
    for j in range(B.shape[1]):
        t, p = pick_paired(B[:, j], A[:, j], a)
        pair_alpha[str(a)].append({"test": t, "p": fin(p)})
pair_kept = {f"{a}|{thr}": [j for j, r in enumerate(pair_alpha[str(a)]) if r["p"] is not None and r["p"] < thr] for a, thr in COMBOS}

out = {
    "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    "unpaired": {
        "names": names, "y": y.tolist(),
        "X": [[None if np.isnan(v) else float(v) for v in row] for row in X],
        "combos": COMBOS, "alphas": ALPHAS, "sets": unp,
    },
    "paired": {
        "names": fnames, "before": B.tolist(), "after": A.tolist(),
        "combos": COMBOS, "alphas": ALPHAS, "perAlpha": pair_alpha, "kept": pair_kept,
    },
}
json.dump(out, open("test/fixtures/preprocess_pipeline.json", "w"), allow_nan=False)
print("sets", len(unp), "features", m, "paired features", B.shape[1], out["versions"])
for lab, a in [("unpaired final", unp[0]["kept"]), ("paired", pair_kept)]:
    print(lab, {k: len(v) for k, v in a.items()})
