"""R5（少数の特徴量での予後モデル）の手順を、TypeScript を見ずに README・コメントの規則から書き直した独立の参照実装。

手順（README・`src/core/r5.ts` の記述）: 相関による除外 → Cox ブースティング → Cox の後退選択 → 最終 Cox
→ Harrell のブートストラップで楽観を補正した C-index（各標本で手順 1〜3 をすべてやり直す）。

目的: 時間とイベントの組を入れ替えた（特徴量と生存が無関係な）データで、補正 C-index が 0.5 を超える原因を切り分ける。
  - 参照も 0.5 を超える → この手順と補正の性質（標本が小さい・特徴量が多い）。実装の不具合とは言えない。
  - 実装だけが超える → 実装の補正の不具合。

使い方:
  RS_VERIFY_DATA=<.../LGN-NSCLC-EGFR/runs> python -I r5_ref.py perm  --seeds 30 --boot 200 --out out.json
  python -I r5_ref.py selftest
"""
import argparse, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- Cox（Efron）。定義どおりの Newton 法
def efron_cox(X, T, E, maxit=100):
    X = np.asarray(X, float); T = np.asarray(T, float); E = np.asarray(E, int)
    n, p = X.shape
    et = np.unique(T[E == 1])
    groups = [((T == t) & (E == 1), T >= t) for t in et]

    def parts(beta):
        r = np.exp(X @ beta)
        ll = 0.0; g = np.zeros(p); H = np.zeros((p, p))
        for ev, rs in groups:
            d = int(ev.sum()); Xe = X[ev]
            ll += float((Xe @ beta).sum()); g += Xe.sum(0)
            S0 = r[rs].sum(); T0 = r[ev].sum()
            S1 = r[rs] @ X[rs]; T1 = r[ev] @ Xe
            S2 = (X[rs] * r[rs, None]).T @ X[rs]; T2 = (Xe * r[ev, None]).T @ Xe
            for l in range(d):
                c = l / d
                A0 = S0 - c * T0; A1 = S1 - c * T1; A2 = S2 - c * T2
                ll -= math.log(A0); g -= A1 / A0
                H += A2 / A0 - np.outer(A1, A1) / A0 ** 2
        return ll, g, H

    beta = np.zeros(p)
    with np.errstate(all="ignore"):
        ll, g, H = parts(beta)
        for _ in range(maxit):
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return None
            h = 1.0
            while True:
                nb = beta + h * step
                try:
                    nll, ng, nH = parts(nb)
                except (ValueError, OverflowError, FloatingPointError):
                    nll, ng, nH = -np.inf, None, None  # 尤度が定義できない（発散）→ 歩幅を半分にする
                if np.isfinite(nll) and nll >= ll - 1e-13:
                    break
                h /= 2
                if h < 1e-12:
                    return None
            beta, ll, g, H = nb, nll, ng, nH
            if np.abs(h * step).max() < 1e-13:
                cov = np.linalg.inv(H)
                se = np.sqrt(np.diag(cov))
                if not np.all(np.isfinite(se)) or np.abs(beta).max() > 30:
                    return None  # 完全分離（係数の発散）
                return dict(beta=beta, se=se, ll=ll, p=2 * stats.norm.sf(np.abs(beta / se)))
    return None


# ---------------------------------------------------------------- Harrell の C（lifelines と同じ規則）
def harrell_c(T, E, risk):
    T = np.asarray(T, float); E = np.asarray(E, int); r = np.asarray(risk, float)
    comp = (E[:, None] == 1) & ((T[:, None] < T[None, :]) | ((T[:, None] == T[None, :]) & (E[None, :] == 0)))
    np.fill_diagonal(comp, False)
    den = comp.sum()
    if den == 0:
        return np.nan
    return (((r[:, None] > r[None, :]) & comp).sum() + 0.5 * ((r[:, None] == r[None, :]) & comp).sum()) / den


# ---------------------------------------------------------------- 手順
def corr_filter(Xr, thr):
    """値が 1 種類の列を除き、|Spearman| ≥ thr の組は、先に残した列を優先して後の列を除く（欠損の無いデータなので列の並び順）。"""
    cand = [j for j in range(Xr.shape[1]) if len(np.unique(Xr[:, j])) >= 2]
    rk = {j: stats.rankdata(Xr[:, j]) for j in cand}
    kept = []
    for j in cand:
        hit = False
        for k in kept:
            a, b = rk[j], rk[k]
            r = np.corrcoef(a, b)[0, 1]
            if np.isfinite(r) and abs(r) >= thr:
                hit = True; break
        if not hit:
            kept.append(j)
    return sorted(kept)


def cox_boost(T, E, Z, iters, nu):
    """成分ごとの尤度ベースの Cox ブースティング（線形の基底、Breslow）。各反復で、負の勾配（マルチンゲール残差）に
    各列の単回帰（切片なし）を当て、残差平方和が最小の 1 列だけを ν 倍して足す。変数重要度 = その列を更新して減った −ll の合計。"""
    n, q = Z.shape
    order = np.argsort(T, kind="stable")
    Ts, Es = T[order], E[order]
    ut = np.unique(Ts[Es == 1])
    ss = (Z ** 2).sum(0)

    def evalbr(eta):
        w = np.exp(eta - eta.max())
        c = eta.max()
        ll = 0.0; cum = np.zeros(n)
        H = 0.0
        # 時間の昇順に累積ハザード Σ d/S0 を作る
        S0s = np.array([w[T >= t].sum() for t in ut])
        ds = np.array([int(((T == t) & (E == 1)).sum()) for t in ut])
        for t, S0, d in zip(ut, S0s, ds):
            ll -= d * (math.log(S0) + c)
        ll += float((eta * E).sum())
        # 各人の累積ハザード: 自分の観察時間までのイベント時刻の寄与
        H_t = np.cumsum(ds / S0s)
        idx = np.searchsorted(ut, T, side="right") - 1
        Hi = np.where(idx >= 0, H_t[np.clip(idx, 0, None)], 0.0)
        g = E - np.exp(eta - c) * Hi  # w・S0 は exp(−c) 倍の尺度なので、exp(η)·H(T) と同じ
        return ll, g

    beta = np.zeros(q); imp = np.zeros(q); eta = np.zeros(n)
    ll, g = evalbr(eta)
    for _ in range(iters):
        xu = Z.T @ g
        sc = xu ** 2 / ss
        j = int(np.argmax(sc))
        step = nu * xu[j] / ss[j]
        beta[j] += step
        eta = eta + step * Z[:, j]
        nll, g = evalbr(eta)
        imp[j] += -(ll) - (-nll)
        ll = nll
    nz = [j for j in range(q) if beta[j] != 0]
    return sorted(nz, key=lambda j: (-imp[j], j))


def backward(T, E, X, rows, cand, pin, pout, maxv):
    """Cox（Efron）の後退選択。1 回に p 最大（> pout）を除き、除くたびに除いた変数を 1 つずつ戻した p を見て最小が pin 未満なら戻す。
    循環したら止める。最後に maxv を超えていれば p の大きい順に除く。失敗（係数の発散）は None。"""
    Tr, Er = T[rows], E[rows]

    def fit(cols):
        Z = X[np.ix_(rows, cols)]
        mu, sd = Z.mean(0), Z.std(0, ddof=1)
        if (sd <= 0).any():
            return None
        return efron_cox((Z - mu) / sd, Tr, Er)

    order = lambda cols: [c for c in cand if c in cols]
    cur = list(cand); removed = []; seen = {tuple(sorted(cur))}
    for _ in range(10 * len(cand) + 10):
        if not cur:
            break
        f = fit(cur)
        if f is None:
            return None
        w = int(np.argmax(f["p"]))
        if not f["p"][w] > pout:
            break
        out = cur[w]; cur = [c for c in cur if c != out]; removed.append(out)
        best = None
        for c in removed:
            if c == out:
                continue
            cc = order(cur + [c]); g = fit(cc)
            if g is None:
                continue
            p = g["p"][cc.index(c)]
            if p < pin and (best is None or p < best[1]):
                best = (c, p)
        if best:
            nxt = order(cur + [best[0]])
            if tuple(sorted(nxt)) in seen:
                break
            cur = nxt; removed.remove(best[0])
        seen.add(tuple(sorted(cur)))
    while len(cur) > maxv:
        f = fit(cur)
        if f is None:
            return None
        cur = [c for a, c in enumerate(cur) if a != int(np.argmax(f["p"]))]
    return cur


def procedure(T, E, X, rows, cfg):
    """rows（重複あり）で手順全体を行い、元の行 i のリスクを返す関数を返す。失敗は None。"""
    rows = np.asarray(rows)
    cols = corr_filter(X[rows], cfg["corr"])
    if not cols:
        return None
    mu, sd = X[np.ix_(rows, cols)].mean(0), X[np.ix_(rows, cols)].std(0, ddof=1)
    ok = sd > 0
    cols = [c for c, k in zip(cols, ok) if k]; mu, sd = mu[ok], sd[ok]
    Z = (X[np.ix_(rows, cols)] - mu) / sd
    rank = cox_boost(T[rows], E[rows], Z, cfg["iters"], cfg["nu"])
    cand = [cols[j] for j in rank]
    if not cand:
        return lambda idx: np.zeros(len(np.atleast_1d(idx)))
    sel = backward(T, E, X, rows, cand, cfg["pin"], cfg["pout"], cfg["maxv"])
    if sel is None:
        return None
    if not sel:
        return lambda idx: np.zeros(len(np.atleast_1d(idx)))
    Z2 = X[np.ix_(rows, sel)]; m2, s2 = Z2.mean(0), Z2.std(0, ddof=1)
    fit = efron_cox((Z2 - m2) / s2, T[rows], E[rows])
    if fit is None:
        return None
    return lambda idx: ((X[np.ix_(np.atleast_1d(idx), sel)] - m2) / s2) @ fit["beta"]


def corrected_c(T, E, X, cfg, B, rng):
    n = len(T); allr = np.arange(n)
    full = procedure(T, E, X, allr, cfg)
    if full is None:
        return None
    app = harrell_c(T, E, full(allr))
    opt = []; fail = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m = procedure(T, E, X, idx, cfg)
        if m is None:
            fail += 1; continue
        cb = harrell_c(T[idx], E[idx], m(idx)); ct = harrell_c(T, E, m(allr))
        if np.isnan(cb) or np.isnan(ct):
            fail += 1; continue
        opt.append(cb - ct)
    return dict(apparent=app, optimism=float(np.mean(opt)), corrected=app - float(np.mean(opt)), ok=len(opt), failed=fail)


# ---------------------------------------------------------------- データと実行
def load():
    import pandas as pd, re
    d = os.environ["RS_VERIFY_DATA"]
    A = pd.read_csv(os.path.join(d, "default56/nsclc.csv"))
    C = pd.read_csv(os.path.join(d, "skill_verify/clinical_surv56_subject.csv"))
    NONFEAT = re.compile(r"^(Provenance_|Roi_|Warning_|Filter_|Diagnostics_)")
    ID = {"CaseLabel", "SubjectCode", "Album", "StudyInstanceUID", "SourceSeriesUID", "MaskSeriesUID"}
    feats = [c for c in A.columns if c not in ID and not NONFEAT.match(c) and pd.api.types.is_numeric_dtype(A[c]) and A[c].notna().any()]
    M = A.merge(C, on="SubjectCode", how="inner")
    feats = [f for f in feats if M[f].notna().all()]  # R5 は欠損の無い特徴量だけ
    return M[feats].values.astype(float), M["OS_days"].values.astype(float), M["OS_event"].values.astype(int), feats


CFG = dict(corr=0.9, iters=100, nu=0.1, pin=0.1, pout=0.15, maxv=4)


def one_perm(args):
    k, B, scale_n = args
    X, T, E, _ = load()
    rng = np.random.default_rng(900 + k)
    perm = rng.permutation(len(T))
    Tp, Ep = T[perm], E[perm]
    r = corrected_c(Tp, Ep, X, CFG, B, rng)
    return dict(seed=k, **(r or {"failed_all": True}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["perm", "selftest"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_perm, [(k, a.boot, 0) for k in range(a.seeds)]))
    good = [r["corrected"] for r in res if "corrected" in r]
    print(json.dumps(res, indent=1))
    print(f"補正 C: n={len(good)} 平均={np.mean(good):.4f} SD={np.std(good, ddof=1):.4f} 範囲=[{min(good):.3f},{max(good):.3f}]")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def selftest():
    """部品を、既存の独立実装（lifelines）と比べる。"""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 80
    X = rng.normal(size=(n, 3)); T = np.round(rng.exponential(10, n) * np.exp(-0.5 * X[:, 0]), 1) + 0.1; E = (rng.random(n) < 0.7).astype(int)
    f = efron_cox(X, T, E)
    df = pd.DataFrame(X, columns=list("abc")); df["T"] = T; df["E"] = E
    c = CoxPHFitter().fit(df, "T", "E")
    assert np.allclose(f["beta"], c.params_.values, atol=1e-5), (f["beta"], c.params_.values)
    assert np.allclose(f["p"], c.summary["p"].values, atol=1e-5)
    r = rng.normal(size=n)
    assert abs(harrell_c(T, E, r) - concordance_index(T, -r, E)) < 1e-12
    # 反例: 完全分離は None
    Xs = np.r_[np.ones(10), np.zeros(10)][:, None]; Ts = np.arange(1, 21.0); Es = np.r_[np.ones(10), np.zeros(10)].astype(int)
    assert efron_cox(Xs, Ts, Es) is None
    # ブースティング: 効く列を最初に拾う
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    assert cox_boost(T, E, Z, 50, 0.1)[0] == 0
    print("selftest ok")


if __name__ == "__main__":
    main()
