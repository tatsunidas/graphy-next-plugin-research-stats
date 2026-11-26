"""ランダムサバイバルフォレストの参照値を作る（test/fixtures/rsf.json）。

乱数を使わない設定（bootstrap=False・max_features=None〈全部〉・木 1 本）で、scikit-survival の RandomSurvivalForest の予測（リスク）を
標本ごとに記録する。この設定でも、scikit-survival は特徴量を無作為に並べ替えるので、分岐の改善が同点の候補が複数あると、
乱数の種で木が変わる（scikit-survival 自身の文書にもある）。そこで、**乱数の種を 6 通り変えても予測が完全に同じ**になる設定だけを
参照値に採用する（同点が無く、木が一意に決まる設定）。時間は丸めて、同時刻（イベントと打ち切り）を含める。
"""
import json
import numpy as np
import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

out = {"versions": {"scikit-survival": sksurv.__version__, "numpy": np.__version__}, "cases": []}
for seed, n, p, rnd in [(1, 60, 5, 1), (2, 80, 8, 1), (3, 50, 4, 0)]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    t = np.round(rng.exponential(10, n) / np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1]), rnd) + 10 ** -rnd
    e = (rng.random(n) < 0.7)
    case = {"seed": seed, "X": X.tolist(), "time": t.tolist(), "event": e.astype(int).tolist(), "fits": []}
    for msplit, mleaf in [(6, 3), (10, 4), (8, 3), (12, 5), (16, 6)]:
        Xnew = rng.normal(size=(15, p))
        preds = []
        for rs in range(6):
            m = RandomSurvivalForest(n_estimators=1, bootstrap=False, max_features=None, min_samples_split=msplit, min_samples_leaf=mleaf, random_state=rs)
            m.fit(X, Surv.from_arrays(e, t))
            preds.append((m.predict(X), m.predict(Xnew)))
        if not all(np.array_equal(preds[0][0], q[0]) and np.array_equal(preds[0][1], q[1]) for q in preds[1:]):
            print("skip (tie-sensitive)", seed, msplit, mleaf)
            continue
        case["fits"].append({"minSplit": msplit, "minLeaf": mleaf, "pred_train": preds[0][0].tolist(), "Xnew": Xnew.tolist(), "pred_new": preds[0][1].tolist()})
    out["cases"].append(case)
# 乱数を使う本来の設定（ブートストラップ・max_features = sqrt・木 100 本）: 固定した分割（前半で作り、後半で C-index）を、20 シードで繰り返した分布
from sksurv.metrics import concordance_index_censored
c2 = out["cases"][1]
X2 = np.array(c2["X"]); t2 = np.array(c2["time"]); e2 = np.array(c2["event"]).astype(bool)
half = len(t2) // 2
cs = []
for rs in range(20):
    m = RandomSurvivalForest(n_estimators=100, bootstrap=True, max_features="sqrt", min_samples_split=6, min_samples_leaf=3, random_state=rs)
    m.fit(X2[:half], Surv.from_arrays(e2[:half], t2[:half]))
    cs.append(float(concordance_index_censored(e2[half:], t2[half:], m.predict(X2[half:]))[0]))
c2["dist"] = {"split": half, "trees": 100, "minLeaf": 3, "minSplit": 6, "cindex": cs}
nfits = sum(len(c["fits"]) for c in out["cases"])
assert nfits >= 6, nfits
json.dump(out, open("test/fixtures/rsf.json", "w"))
print("wrote", len(out["cases"]), "cases,", nfits, "tie-free settings")
