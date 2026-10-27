"""LASSO（Cox・ロジスティック）・Cox（Breslow / Efron）・Schoenfeld 残差・後退選択の参照値を、TypeScript 側とは別に出す。
LASSO-Cox は statsmodels の PHReg.fit_regularized（ties="breslow"）、ロジスティック LASSO は scikit-learn（saga、切片は罰則なし）。
参照の解が本当に最小かを、ここで KKT 条件（|勾配| ≤ λ、係数が 0 でなければ勾配 = −λ·符号）で確かめて記録する。
使い方: python3 tools/gen_model_fixture.py  → test/fixtures/model.json
"""
import json
import warnings

import numpy as np
import scipy
import scipy.stats as st
import sklearn
import statsmodels
from sklearn.linear_model import LogisticRegression
from statsmodels.duration.hazard_regression import PHReg


def standardize(X):
    m = X.mean(axis=0)
    s = X.std(axis=0, ddof=1)
    return (X - m) / s


def surv_data(seed, n, p, signal):
    rng = np.random.default_rng(seed)
    scales = np.array([1.0, 10.0, 0.01, 3.0, 100.0, 0.5, 2.0, 7.0, 0.2, 40.0])[:p]
    X = rng.normal(size=(n, p)) * scales + rng.normal(size=p) * scales
    Z = standardize(X)
    eta = Z[:, : len(signal)] @ np.array(signal)
    t = rng.exponential(1.0 / np.exp(eta))
    c = rng.exponential(2.0, size=n)
    time = np.round(np.minimum(t, c) + 1e-3, 6)
    event = (t <= c).astype(int)
    assert len(np.unique(time)) == n, "時間に同点がある"
    return X, time, event


def breslow_score(Z, time, event, beta):
    """∂l/∂β（Breslow）を、ここで別に書いた式で計算する。"""
    eta = Z @ beta
    w = np.exp(eta)
    g = np.zeros(Z.shape[1])
    for i in np.where(event == 1)[0]:
        R = time >= time[i]
        g += Z[i] - (w[R][:, None] * Z[R]).sum(0) / w[R].sum()
    return g


def kkt(grad_loss, beta, lam):
    """grad_loss は −(1/n)l の勾配。最小なら 0 の係数で |g| ≤ λ、0 でない係数で g = −λ·sign(β)。違反の最大を返す。"""
    v = 0.0
    for g, b in zip(grad_loss, beta):
        v = max(v, abs(g + lam * np.sign(b)) if b != 0 else max(0.0, abs(g) - lam))
    return v


def lasso_cox_case():
    X, time, event = surv_data(11, 90, 6, [0.8, -0.6])
    n = len(time)
    Z = standardize(X)
    lmax = np.abs(breslow_score(Z, time, event, np.zeros(Z.shape[1]))).max() / n
    out = []
    for f in [0.6, 0.3, 0.1, 0.03]:
        lam = lmax * f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = PHReg(time, Z, status=event, ties="breslow").fit_regularized(alpha=lam, L1_wt=1.0, cnvrg_tol=1e-12, zero_tol=1e-10, maxiter=2000)
        beta = np.asarray(r.params)
        out.append({"lambda": lam, "beta": beta.tolist(), "kkt": kkt(-breslow_score(Z, time, event, beta) / n, beta, lam)})
    full = PHReg(time, Z, status=event, ties="breslow").fit()
    return {"X": X.tolist(), "time": time.tolist(), "event": event.tolist(), "lambdaMax": lmax, "fits": out, "breslowBeta": np.asarray(full.params).tolist()}


def lasso_logistic_case():
    rng = np.random.default_rng(7)
    n, p = 120, 6
    X = rng.normal(size=(n, p)) * np.array([1, 5, 0.1, 2, 50, 1]) + 3
    Z = standardize(X)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + 1.0 * Z[:, 0] - 0.7 * Z[:, 1])))).astype(int)
    pbar = y.mean()
    lmax = np.abs(Z.T @ (y - pbar)).max() / n
    out = []
    for f in [0.6, 0.3, 0.1, 0.03]:
        lam = lmax * f
        m = LogisticRegression(C=1.0 / (n * lam), l1_ratio=1.0, solver="saga", tol=1e-12, max_iter=1_000_000, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z, y)
        b0, beta = float(m.intercept_[0]), m.coef_[0]
        pr = 1 / (1 + np.exp(-(b0 + Z @ beta)))
        grad = -(Z.T @ (y - pr)) / n
        out.append({"lambda": lam, "b0": b0, "beta": beta.tolist(), "kkt": kkt(grad, beta, lam), "interceptGrad": float(abs((y - pr).sum() / n))})
    return {"X": X.tolist(), "y": y.tolist(), "lambdaMax": lmax, "fits": out}


def ties_case():
    """同時刻のイベントが多いデータ。Breslow と Efron で係数が違うことを使い、取り違えを検出する。"""
    rng = np.random.default_rng(31)
    n = 100
    X = rng.normal(size=(n, 3))
    Z = standardize(X)
    t = rng.exponential(1.0 / np.exp(0.9 * Z[:, 0] - 0.5 * Z[:, 1]))
    time = np.ceil(t * 5) / 5
    event = (rng.random(n) < 0.8).astype(int)
    b = PHReg(time, Z, status=event, ties="breslow").fit()
    e = PHReg(time, Z, status=event, ties="efron").fit()
    return {"X": X.tolist(), "time": time.tolist(), "event": event.tolist(), "breslowBeta": np.asarray(b.params).tolist(), "efronBeta": np.asarray(e.params).tolist(), "distinctTimes": int(len(np.unique(time)))}


def schoenfeld_case():
    X, time, event = surv_data(23, 70, 3, [0.7, 0.0, -0.5])
    Z = standardize(X)
    r = PHReg(time, Z, status=event, ties="breslow").fit()
    beta, V = np.asarray(r.params), np.asarray(r.cov_params())
    sr = r.schoenfeld_residuals
    ev = np.where(event == 1)[0]
    ev = ev[np.argsort(time[ev])]
    resid = sr[ev]
    d = len(ev)
    # 1 − KM(t−)
    order = np.argsort(time)
    km, at, g = 1.0, len(time), {}
    for i in order:
        g[i] = 1 - km
        if event[i] == 1:
            km *= 1 - 1 / at
        at -= 1
    gt = np.array([g[i] for i in ev])
    xx = gt - gt.mean()
    r2 = resid @ V * d
    tests = [float((xx @ r2[:, j]) ** 2 / (V[j, j] * d * (xx @ xx))) for j in range(Z.shape[1])]
    u = xx @ resid
    glob = float(d * u @ V @ u / (xx @ xx))
    return {
        "X": X.tolist(), "time": time.tolist(), "event": event.tolist(), "beta": beta.tolist(), "cov": V.tolist(),
        "rows": ev.tolist(), "resid": resid.tolist(), "scaled": (r2 + beta).tolist(),
        "chi2": tests, "p": [float(st.chi2.sf(c, 1)) for c in tests], "globalChi2": glob, "globalP": float(st.chi2.sf(glob, Z.shape[1])),
    }


def backward_run(X, time, event, p_in, p_out, max_vars):
    m = X.shape[1]

    def fit(cols):
        Z = standardize(X[:, cols])
        return np.asarray(PHReg(time, Z, status=event, ties="efron").fit().pvalues)

    order = lambda cs: [c for c in range(m) if c in cs]
    cur, removed, steps, seen = list(range(m)), [], [], {tuple(range(m))}
    while cur:
        p = fit(cur)
        w = int(np.argmax(p))
        if not p[w] > p_out:
            break
        out = cur[w]
        cur = [c for c in cur if c != out]
        removed.append(out)
        steps.append({"action": "remove", "col": out, "p": float(p[w])})
        best = None
        for c in removed:
            if c == out:
                continue
            nc = order(cur + [c])
            pc = fit(nc)[nc.index(c)]
            if pc < p_in and (best is None or pc < best[1]):
                best = (c, pc)
        if best:
            nxt = order(cur + [best[0]])
            if tuple(sorted(nxt)) in seen:
                steps.append({"action": "cycle", "col": best[0], "p": float(best[1])})
                break
            cur = nxt
            removed.remove(best[0])
            steps.append({"action": "add", "col": best[0], "p": float(best[1])})
        seen.add(tuple(sorted(cur)))
    pre_cap = list(cur)
    while len(cur) > max_vars:
        p = fit(cur)
        w = int(np.argmax(p))
        steps.append({"action": "cap", "col": cur[w], "p": float(p[w])})
        cur = [c for i, c in enumerate(cur) if i != w]
    return {"pIn": p_in, "pOut": p_out, "maxVars": max_vars, "steps": steps, "cols": cur, "preCap": pre_cap}


def backward_case():
    X, time, event = surv_data(5, 150, 8, [0.9, -0.7, 0.35])
    runs = [backward_run(X, time, event, 0.1, 0.15, 4), backward_run(X, time, event, 0.05, 0.3, 4), backward_run(X, time, event, 0.01, 0.05, 4)]
    efron_all = PHReg(time, standardize(X), status=event, ties="efron").fit()
    return {
        "X": X.tolist(), "time": time.tolist(), "event": event.tolist(), "runs": runs,
        "efronAllP": np.asarray(efron_all.pvalues).tolist(), "efronAllBeta": np.asarray(efron_all.params).tolist(),
    }


def main():
    out = {
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__},
        "lassoCox": lasso_cox_case(),
        "lassoLogistic": lasso_logistic_case(),
        "ties": ties_case(),
        "schoenfeld": schoenfeld_case(),
        "backward": backward_case(),
    }
    with open("test/fixtures/model.json", "w") as f:
        json.dump(out, f)
    for k in ["lassoCox", "lassoLogistic"]:
        print(k, "KKT 違反の最大", max(x["kkt"] for x in out[k]["fits"]), "非 0 の数", [sum(b != 0 for b in x["beta"]) for x in out[k]["fits"]])
    t = out["ties"]
    print("ties: 異なる時間", t["distinctTimes"], "Breslow", t["breslowBeta"], "Efron", t["efronBeta"])
    for r in out["backward"]["runs"]:
        print("backward", r["pIn"], r["pOut"], [(x["action"], x["col"], round(x["p"], 4)) for x in r["steps"]], r["cols"])


if __name__ == "__main__":
    main()
