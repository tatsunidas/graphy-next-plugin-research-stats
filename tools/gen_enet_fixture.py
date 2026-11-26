"""Elastic Net・Ridge の Cox（Breslow）の固定 λ での係数の参照値を作る（test/fixtures/enet.json）。

- Elastic Net（0 < α < 1）と LASSO（α = 1）: scikit-survival の CoxnetSurvivalAnalysis（目的関数 −(1/n)·l(β) + λ·[α‖β‖₁ + (1−α)/2·‖β‖²]、同時刻は Breslow）。
- Ridge（α = 0）: Coxnet は l1_ratio = 0 を受け付けないので、同じ目的関数を Newton 法で解く自前の参照（Breslow の部分尤度の勾配とヘッセ行列を定義どおりに書く）。
  この自前の参照が正しいことは、l1_ratio が小さい Coxnet（0.01 → 0.001）に近づくことで確かめる。
標準化は標本 SD（ddof = 1）。係数は標準化した尺度。
"""
import json
import numpy as np
import sksurv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv


def standardize(X):
    return (X - X.mean(0)) / X.std(0, ddof=1)


def breslow_parts(Z, t, e, beta):
    """−(1/n)·l(β) の値・勾配・ヘッセ行列（同時刻は Breslow）。"""
    n, p = Z.shape
    eta = Z @ beta
    w = np.exp(eta)
    f = 0.0; g = np.zeros(p); H = np.zeros((p, p))
    for tt in np.unique(t[e == 1]):
        ev = (t == tt) & (e == 1)
        rs = t >= tt
        d = int(ev.sum())
        S0 = w[rs].sum(); S1 = (w[rs, None] * Z[rs]).sum(0); S2 = (Z[rs].T * w[rs]) @ Z[rs]
        f -= (eta[ev].sum() - d * np.log(S0))
        g -= (Z[ev].sum(0) - d * S1 / S0)
        H += d * (S2 / S0 - np.outer(S1, S1) / S0 ** 2)
    return f / n, g / n, H / n


def ridge_cox(Z, t, e, lam):
    """−(1/n)·l(β) + (λ/2)‖β‖² の最小（Newton 法。半減つき）。"""
    p = Z.shape[1]
    beta = np.zeros(p)
    def obj(b):
        f, g, H = breslow_parts(Z, t, e, b)
        return f + lam / 2 * b @ b, g + lam * b, H + lam * np.eye(p)
    F, G, H = obj(beta)
    for _ in range(200):
        step = np.linalg.solve(H, G)
        h = 1.0
        while True:
            nb = beta - h * step
            F2, G2, H2 = obj(nb)
            if F2 <= F + 1e-16 * abs(F):
                break
            h /= 2
            assert h > 1e-12
        beta, F, G, H = nb, F2, G2, H2
        if np.abs(h * step).max() < 1e-14:
            break
    return beta


def make(seed, n, p, tie_round):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] * 0.7 + 0.3 * X[:, 1]  # 相関のある対
    eta = 0.8 * X[:, 0] - 0.6 * X[:, 2] + 0.3 * X[:, 3]
    t = np.round(rng.exponential(1.0, n) / np.exp(eta) * 10, tie_round) + 10 ** -tie_round
    c = np.round(rng.exponential(1.4, n) * 10, tie_round) + 10 ** -tie_round
    time = np.minimum(t, c); event = (t <= c).astype(int)
    return X, time, event


def coxnet(Z, t, e, lam, ratio):
    m = CoxnetSurvivalAnalysis(l1_ratio=ratio, alphas=[lam], fit_baseline_model=False, normalize=False, tol=1e-13, max_iter=10 ** 7)
    m.fit(Z, Surv.from_arrays(e.astype(bool), t))
    return m.coef_[:, 0]


out = {"versions": {"scikit-survival": sksurv.__version__, "numpy": np.__version__}, "cases": []}
for seed, n, p, tie in [(1, 60, 8, 0), (2, 80, 12, 1), (3, 45, 6, 2)]:
    X, t, e = make(seed, n, p, tie)
    Z = standardize(X)
    case = {"seed": seed, "X": X.tolist(), "time": t.tolist(), "event": e.tolist(), "fits": []}
    for ratio in (1.0, 0.5, 0.2):
        for lam in (0.2, 0.05, 0.01):
            case["fits"].append({"alpha": ratio, "lambda": lam, "beta": coxnet(Z, t, e, lam, ratio).tolist()})
    for lam in (0.5, 0.1, 0.02):
        case["fits"].append({"alpha": 0.0, "lambda": lam, "beta": ridge_cox(Z, t, e, lam).tolist()})
    # 自前の Ridge 参照の確認: l1_ratio を 0.001 まで小さくした Coxnet に近づく（同じ目的関数の極限）
    lam = 0.1
    r0 = ridge_cox(Z, t, e, lam)
    r1 = coxnet(Z, t, e, lam, 0.001)
    # Coxnet の l1_ratio = 0.001 は、λ·0.001 の L1 が残る（Ridge との差は L1 の分だけ）。差が小さいことを記録する
    case["ridge_vs_coxnet_0.001_max_abs_diff"] = float(np.abs(r0 - r1).max())
    out["cases"].append(case)
    assert case["ridge_vs_coxnet_0.001_max_abs_diff"] < 5e-3, case["ridge_vs_coxnet_0.001_max_abs_diff"]
json.dump(out, open("test/fixtures/enet.json", "w"))
print("wrote", len(out["cases"]), "cases;", [round(c["ridge_vs_coxnet_0.001_max_abs_diff"], 6) for c in out["cases"]])
