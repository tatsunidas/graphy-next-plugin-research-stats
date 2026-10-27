"""決定木・ランダムフォレスト・勾配ブースティング・SVM の参照値を、TypeScript 側とは別に scikit-learn で出す。
sklearn の木は特徴量を float32 にしてから分岐を探すので、X は float32 で表せる値にしておく（同じ値を比べるため）。
乱数を使わない設定（bootstrap なし・subsample=1・SVM）は数値で、乱数を使う設定はシードを変えた AUC の分布で照合する。
使い方: python3 tools/gen_ml_fixture.py  → test/fixtures/ml.json
"""
import json

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


rng = np.random.default_rng(2026)
n, p = 240, 10
X = rng.normal(size=(n, p))
X[:, 5] = X[:, 0] * 0.8 + rng.normal(size=n) * 0.6
eta = 1.2 * X[:, 0] - 1.0 * X[:, 1] + 0.9 * X[:, 2] * X[:, 3] + 0.8 * np.abs(X[:, 4]) - 0.5
y = (eta + rng.logistic(size=n) > 0).astype(int)
tr, te = np.arange(160), np.arange(160, 240)
# 標準化（訓練の行の平均・SD［n−1］）してから float32 に丸める
Z = f32((X - X[tr].mean(0)) / X[tr].std(0, ddof=1))
for j in range(p):
    assert len(np.unique(Z[:, j])) == n, "同じ値がある"
Ztr, Zte, ytr, yte = Z[tr], Z[te], y[tr], y[te]

out = {"versions": {"sklearn": sklearn.__version__, "numpy": np.__version__}, "X": Z.tolist(), "y": y.tolist(), "train": tr.tolist(), "test": te.tolist()}

def tree_arrays(m):
    """木の構造（ノードごとの特徴量・しきい値・子・値・重み）。葉の特徴量は -1。分類の値はクラス 1 の割合。"""
    t = m.tree_
    v = t.value[:, 0, 1] if t.value.shape[2] == 2 else t.value[:, 0, 0]
    return {"feature": [int(f) if f >= 0 else -1 for f in t.feature], "threshold": t.threshold.tolist(), "left": t.children_left.tolist(),
            "right": t.children_right.tolist(), "value": v.tolist(), "weight": t.weighted_n_node_samples.tolist()}


def invariant(make, X_, y_, Xt):
    """random_state を変えても予測が変わらないか（分岐の同点が結果に効いていないか）。"""
    ps = [make(s).fit(X_, y_).predict_proba(Xt)[:, 1] for s in range(10)]
    return all(np.array_equal(ps[0], q) for q in ps[1:]), ps


# 単独の木（重みなし・深さ上限なし／3）と、整数の重み（ブートストラップの回数を模す）。
# 小さいノードでは改善量が同点の分岐がよくあり、sklearn は内部の乱数の順で選ぶ。照合は「各ノードの分岐の改善量が最良と同じか」で行う。
w = np.random.default_rng(7).integers(0, 4, size=len(tr)).astype(float)
cart = {}
for name, depth, sw in [("full", None, None), ("depth3", 3, None), ("weighted", None, w)]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Ztr, ytr, sample_weight=sw)
    cart[name] = {"depth": depth, "tree": tree_arrays(m)}
out["cartWeights"] = w.tolist()
out["cart"] = cart
target = f32(np.sin(Ztr[:, 0] * 2) + Ztr[:, 1] ** 2 + 0.3 * rng.normal(size=len(tr)))
mr = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Ztr, target)
out["regTree"] = {"target": target.tolist(), "depth": 4, "tree": tree_arrays(mr)}

# 乱数を使わない GBM。同点が予測に効かない設定（random_state を変えても sklearn の予測が同じ）は数値で、
# 効く設定は sklearn の 10 個の random_state の AUC の範囲で照合する。
gb = []
for stages, depth, lr in [(100, 3, 0.1), (30, 2, 0.3), (20, 1, 0.5)]:
    make = lambda s: GradientBoostingClassifier(n_estimators=stages, max_depth=depth, learning_rate=lr, random_state=s)
    inv, ps = invariant(make, Ztr, ytr, Zte)
    m = make(0).fit(Ztr, ytr)
    gb.append({"stages": stages, "depth": depth, "lr": lr, "invariant": inv, "proba": m.predict_proba(Zte)[:, 1].tolist(), "raw": m.decision_function(Zte).tolist(),
               "importance": m.feature_importances_.tolist(), "aucRange": [min(roc_auc_score(yte, q) for q in ps), max(roc_auc_score(yte, q) for q in ps)]})
out["gbmDet"] = gb

# SVM。許容誤差は、sklearn 自身の tol=1e-10 と 1e-12 の決定値の差の 10 倍（解き切れていない分の大きさ）。
sv = []
for C, gamma in [(1.0, 0.05), (10.0, "scale"), (0.1, 0.2)]:
    m = SVC(C=C, kernel="rbf", gamma=gamma, tol=1e-10, shrinking=False).fit(Ztr, ytr)
    m2 = SVC(C=C, kernel="rbf", gamma=gamma, tol=1e-12, shrinking=False).fit(Ztr, ytr)
    g = 1.0 / (p * Ztr.var()) if gamma == "scale" else gamma
    sv.append({"C": C, "gamma": gamma, "gammaValue": float(g), "decision": m.decision_function(Zte).tolist(), "support": sorted(m.support_.tolist()),
               "selfDiff": float(np.max(np.abs(m.decision_function(Zte) - m2.decision_function(Zte)))),
               "nBound": int(np.sum(np.isclose(np.abs(m.dual_coef_[0]), C))), "auc": roc_auc_score(yte, m.decision_function(Zte))})
out["svm"] = sv

# 乱数を使う形: シード 20 個の AUC の分布。RF は特徴量の多いデータ（30 列、うち信号 5 列）で、
# 木ごとに特徴量を選ぶ誤った実装と区別できるようにする。
SEEDS = 20
r2 = np.random.default_rng(42)
n2, p2 = 200, 30
W = r2.normal(size=(n2, p2))
W[:, 5] = W[:, 0] * 0.8 + r2.normal(size=n2) * 0.6
eta2 = 1.2 * W[:, 0] - 1.0 * W[:, 1] + 0.8 * W[:, 2] * W[:, 3] + 0.7 * np.abs(W[:, 4])
y2 = (eta2 + r2.logistic(size=n2) > 0.5).astype(int)
tr2, te2 = np.arange(140), np.arange(140, 200)
W = f32((W - W[tr2].mean(0)) / W[tr2].std(0, ddof=1))
out["wide"] = {"X": W.tolist(), "y": y2.tolist(), "train": tr2.tolist(), "test": te2.tolist()}
# 重要度の定義の確認: RF は「木ごとに正規化した不純度の減少の平均を、もう一度正規化」（TS 側はこの定義で実装）
_rf = RandomForestClassifier(n_estimators=7, max_features="sqrt", random_state=3).fit(W[tr2], y2[tr2])
_per = np.array([e.tree_.compute_feature_importances(normalize=True) for e in _rf.estimators_])
assert np.allclose(_rf.feature_importances_, _per.mean(0) / _per.mean(0).sum(), atol=1e-15), "RF の重要度の定義が違う"
out["rfDist"] = {"trees": 200, "mtry": "sqrt", "auc": [roc_auc_score(y2[te2], RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=s).fit(W[tr2], y2[tr2]).predict_proba(W[te2])[:, 1]) for s in range(SEEDS)]}
out["gbmDist"] = {"stages": 100, "depth": 3, "lr": 0.1, "subsample": 0.7,
                  "auc": [roc_auc_score(yte, GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.7, random_state=s).fit(Ztr, ytr).predict_proba(Zte)[:, 1]) for s in range(SEEDS)]}

json.dump(out, open("test/fixtures/ml.json", "w"))
print("rf", np.mean(out["rfDist"]["auc"]), np.std(out["rfDist"]["auc"], ddof=1), "gbm", np.mean(out["gbmDist"]["auc"]), np.std(out["gbmDist"]["auc"], ddof=1),
      "svm auc", [s["auc"] for s in sv], "nSV", [len(s["support"]) for s in sv], "bound", [s["nBound"] for s in sv], "selfDiff", [s["selfDiff"] for s in sv],
      "gbm invariant", [g["invariant"] for g in gb], [g["aucRange"] for g in gb])
