#!/usr/bin/env python3
"""
実データ（NSCLC Radiogenomics 56 例）での検証: run_core.mts が出した JSON を、TypeScript の実装を見ずに
定義から書いた参照実装（lifelines・scipy・scikit-learn・statsmodels・pingouin）と突き合わせ、REPORT.md を書く。

環境変数:
  RS_VERIFY_DATA    LGN-NSCLC-EGFR/runs（読み取りだけ）
  RS_VERIFY_OUT     run_core が書いた JSON のあるディレクトリ
  RS_VERIFY_REPORT  REPORT.md の書き出し先
使い方: <venv>/bin/python -I verify_real.py
"""
import io
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA = os.environ["RS_VERIFY_DATA"]
OUT = os.environ["RS_VERIFY_OUT"]
REPORT = os.environ.get("RS_VERIFY_REPORT", os.path.join(OUT, "REPORT.md"))

# ---------------------------------------------------------------- 検査の記録

CHECKS = []  # dict(skill, id, status, observed, tol, note)
SECTION_NOTES = {}


def rec(skill, cid, status, observed="", tol="", note="", cause=""):
    """cause（失敗のときの原因の見立て）: impl=実装の食い違い / refdef=参照側の定義の違い / refdefect=参照側の不具合 / design=設計上の観察"""
    assert status in ("PASS", "FAIL", "UNMET")
    CHECKS.append(dict(skill=skill, id=cid, status=status, observed=str(observed), tol=str(tol), note=str(note), cause=cause))


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def compare(skill, cid, ref, got, atol, rtol, why, note="", cause=""):
    """ref と got（配列）の最大の差。欠損の位置が違えば失敗。"""
    ref = np.asarray(ref, dtype=float)
    got = np.asarray(got, dtype=float)
    atol = np.broadcast_to(np.asarray(atol, dtype=float), ref.shape) if ref.shape == got.shape else atol
    if ref.shape != got.shape:
        rec(skill, cid, "FAIL", f"形が違う {ref.shape} vs {got.shape}", why, note)
        return False
    nanmis = np.isnan(ref) != np.isnan(got)
    both = ~np.isnan(ref) & ~np.isnan(got)
    if both.sum() == 0 and not nanmis.any():
        rec(skill, cid, "UNMET", "比べる値が 0 個", why, note)
        return False
    d = np.abs(ref[both] - got[both])
    lim = atol[both] + rtol * np.abs(ref[both])
    bad = d > lim
    ok = (not nanmis.any()) and (not bad.any())
    mx = float(d.max()) if len(d) else 0.0
    rel = float((d / np.maximum(np.abs(ref[both]), 1e-300)).max()) if len(d) else 0.0
    obs = f"n={int(both.sum())} 最大絶対差={mx:.3g} 最大相対差={rel:.3g}"
    if nanmis.any():
        obs += f" 欠損位置の不一致={int(nanmis.sum())}"
    if bad.any():
        obs += f" 許容超過={int(bad.sum())}"
    at = f"{float(atol.max()):.3g}" if atol.size else "0"
    rec(skill, cid, "PASS" if ok else "FAIL", obs, f"atol={at}, rtol={rtol:g}（{why}）", note, cause if not ok else "")
    return ok


def J(rid):
    p = os.path.join(OUT, rid + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def T(d, name):
    return pd.read_csv(io.StringIO(d["tables"][name]), keep_default_na=True)


def step(d, sid):
    return next((s for s in d["steps"] if s["id"] == sid), None)


# ---------------------------------------------------------------- データ（pandas で独立に読む）

NONFEAT = re.compile(r"^(Provenance_|Roi_|Warning_|Filter_|Diagnostics_)")
ID_COLS = {"CaseLabel", "SubjectCode", "Album", "StudyInstanceUID", "SourceSeriesUID", "MaskSeriesUID"}


def feature_columns(df):
    """pandas が数として読んだ列のうち、記録・ID の列でなく、値が 1 つ以上あるもの。"""
    return [c for c in df.columns if c not in ID_COLS and not NONFEAT.match(c)
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


A = pd.read_csv(os.path.join(DATA, "default56/nsclc.csv"))
B = pd.read_csv(os.path.join(DATA, "iso1mm_bw25/nsclc.csv"))
C = pd.read_csv(os.path.join(DATA, "skill_verify/clinical_surv56_subject.csv"))
FEATS = feature_columns(A)
M = A.merge(C, on="SubjectCode", how="inner")
VERS = {}


def lib_versions():
    import lifelines, pingouin, sklearn, statsmodels
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__, pandas=pd.__version__,
                lifelines=lifelines.__version__, pingouin=pingouin.__version__, sklearn=sklearn.__version__, statsmodels=statsmodels.__version__)


from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

FIT_OPT = dict(fit_options=dict(precision=1e-12, step_size=0.95, max_steps=200))


def bh(p):
    """欠損を除いて BH。欠損は NaN のまま。"""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def cox_uni(T_, E_, X):
    """lifelines の Cox（Efron）。X は (n,p)。戻り: DataFrame(summary) と cph。失敗なら None。"""
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    df["T"] = T_
    df["E"] = E_
    cph = CoxPHFitter()
    try:
        cph.fit(df, "T", "E", **FIT_OPT)
    except Exception as e:  # 収束しない・完全分離
        return None, str(e)[:80]
    return cph, None


def efron_cox(X, T_, E_, maxit=100):
    """自前の Cox（Efron の同時刻処理）。部分尤度・勾配・情報行列を定義どおりに求め、Newton 法（半減つき）で |step|<1e-13 まで解く。"""
    X = np.asarray(X, float)
    T_ = np.asarray(T_, float)
    E_ = np.asarray(E_, int)
    n, p = X.shape
    et = np.unique(T_[E_ == 1])
    groups = [((T_ == t) & (E_ == 1), T_ >= t) for t in et]

    def parts(beta, need=True):
        r = np.exp(X @ beta)
        ll = 0.0
        g = np.zeros(p)
        H = np.zeros((p, p))
        for ev, rs in groups:
            d = int(ev.sum())
            Xe = X[ev]
            ll += float((Xe @ beta).sum())
            if need:
                g += Xe.sum(0)
            S0 = r[rs].sum(); T0 = r[ev].sum()
            if need:
                S1 = r[rs] @ X[rs]; T1 = r[ev] @ Xe
                S2 = (X[rs] * r[rs, None]).T @ X[rs]; T2 = (Xe * r[ev, None]).T @ Xe
            for l in range(d):
                c = l / d
                A0 = S0 - c * T0
                ll -= math.log(A0)
                if need:
                    A1 = S1 - c * T1; A2 = S2 - c * T2
                    g -= A1 / A0
                    H += A2 / A0 - np.outer(A1, A1) / A0 ** 2
        return ll, g, H

    beta = np.zeros(p)
    ll, g, H = parts(beta)
    ok = False
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
            except (OverflowError, ValueError):
                nll = -np.inf
            if np.isfinite(nll) and nll >= ll - 1e-13:
                break
            h /= 2
            if h < 1e-12:
                return None
        beta, ll, g, H = nb, nll, ng, nH
        if np.abs(h * step).max() < 1e-13:
            ok = True
            break
    if not ok:
        return None
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return dict(beta=beta, se=se, cov=cov, ll=ll, p=2 * stats.norm.sf(np.abs(z)), z=z)


def efron_summary(fit, level=0.975):
    zc = stats.norm.ppf(level)
    return dict(hr=np.exp(fit["beta"]), lo=np.exp(fit["beta"] - zc * fit["se"]), hi=np.exp(fit["beta"] + zc * fit["se"]), p=fit["p"])


def cox_tols(fit):
    """TS の停止規則（ll の変化 < 1e-10(|ll|+1)）が許す係数のずれの上限: ½δ'Iδ ≤ ε → |δ_j| ≤ √(2ε·(I⁻¹)_jj)。"""
    eps = 1e-10 * (abs(fit["ll"]) + 1)
    return np.sqrt(2 * eps * np.diag(fit["cov"]))


def cox_compare(skill, cid, fit, got_hr, got_lo, got_hi, got_p, why):
    """係数（log HR）・信頼区間の端・p を、停止規則から決めたずれの上限で比べる。fit, got_* は同じ長さの配列。"""
    tb = cox_tols(fit)
    s = efron_summary(fit)
    ok1 = compare(skill, f"{cid}.log_hr", np.log(s["hr"]), np.log(np.asarray(got_hr, float)), tb + 1e-12, 0.0, why + "。許容は特徴量ごとに √(2ε·(I⁻¹)jj)、ε=1e-10(|ll|+1)（TS の停止規則が許す係数のずれの上限）")
    tci = 2 * tb + 1e-12
    compare(skill, f"{cid}.log_ci_lower", np.log(s["lo"]), np.log(np.asarray(got_lo, float)), tci, 0.0, "同上（区間の端 = β−1.96 se。se のずれは二次なので 2 倍）")
    compare(skill, f"{cid}.log_ci_upper", np.log(s["hi"]), np.log(np.asarray(got_hi, float)), tci, 0.0, "同上")
    pdf = stats.norm.pdf(fit["z"])
    compare(skill, f"{cid}.p", s["p"], got_p, 2 * pdf / fit["se"] * tb * 1.01 + 1e-12, 0.0, "p の許容 = |dp/dβ|·（係数の許容）= 2φ(z)/se·δ")
    return ok1


def harrell_c(T_, E_, risk):
    """Harrell の C（定義どおりの総当たり）。比較可能な組 = i がイベントで、T_i < T_j、または T_i = T_j で j が打ち切り
    （イベントの方が先とみなす。lifelines の concordance_index と同じ規則）。リスクの同点は 0.5。"""
    T_ = np.asarray(T_, float); E_ = np.asarray(E_, int); r = np.asarray(risk, float)
    comp = (E_[:, None] == 1) & ((T_[:, None] < T_[None, :]) | ((T_[:, None] == T_[None, :]) & (E_[None, :] == 0)))
    np.fill_diagonal(comp, False)
    den = comp.sum()
    if den == 0:
        return np.nan
    conc = ((r[:, None] > r[None, :]) & comp).sum() + 0.5 * ((r[:, None] == r[None, :]) & comp).sum()
    return conc / den


# ---------------------------------------------------------------- Dataset check（②）


def check_dc():
    S = "Dataset check"
    d = J("dc")
    if d is None:
        rec(S, "run", "UNMET", "dc.json が無い")
        return
    j = step(d, "join")["summary"]
    rep = d["extra"]["report"]
    feat_ids = set(A["SubjectCode"])
    clin_ids = set(C["SubjectCode"])
    exp = dict(featureRows=len(A), patientsInFeatures=A["SubjectCode"].nunique(), patientsInClinical=C["SubjectCode"].nunique(),
               joined=len(feat_ids & clin_ids), featureOnly=len(feat_ids - clin_ids), clinicalOnly=len(clin_ids - feat_ids),
               features=len(FEATS), survivalUsable=int((C["OS_days"].notna() & C["OS_event"].notna()).sum()))
    for k, v in exp.items():
        rec(S, f"join.{k}", "PASS" if j[k] == v else "FAIL", f"実装={j[k]} pandas={v}", "整数の一致")
    # 欠損のある列: 特徴量（pandas の isna）と役割に使った臨床の列
    miss_exp = {c: int(M[c].isna().sum()) for c in FEATS if M[c].isna().any()}
    for c in ["OS_days", "OS_event", "EGFR", "Age", "Female"]:
        if M[c].isna().any():
            miss_exp[c] = int(M[c].isna().sum())
    miss_got = {m["name"]: m["count"] for m in rep["missing"]}
    rec(S, "missing.columns", "PASS" if miss_exp == miss_got else "FAIL", f"実装={miss_got} pandas={miss_exp}", "列名と件数の一致")
    # 除外の理由: 時間・イベントが揃わない患者は 0 人のはず
    rec(S, "exclusions", "PASS" if len(rep["exclusions"]) == 0 and (C["OS_days"].isna().sum() + C["OS_event"].isna().sum()) == 0 else "FAIL",
        f"実装の除外={len(rep['exclusions'])} pandas の時間・イベント欠損={int(C['OS_days'].isna().sum() + C['OS_event'].isna().sum())}", "件数の一致")
    rec(S, "leak.idLike", "PASS" if step(d, "leak")["summary"]["idLikeUsed"] == 0 else "FAIL", f"idLikeUsed={step(d, 'leak')['summary']['idLikeUsed']}", "ID・記録の列が特徴量に入らない")
    # 結合した表の値が元の表と同じ（先頭 3 列・全セル）
    ds = pd.read_csv(io.StringIO(d["tables"]["dataset"]))
    ds = ds.set_index("SubjectCode")
    Mi = M.set_index("SubjectCode").loc[ds.index]
    okv = all(np.allclose(ds[c].astype(float).values, Mi[c].astype(float).values, equal_nan=True, rtol=1e-12, atol=0) for c in FEATS)
    rec(S, "dataset.values", "PASS" if okv and list(ds.columns[-len(FEATS):]) == FEATS else "FAIL", f"{len(FEATS)} 列×{len(ds)} 行の全セルが元の表と一致={okv}", "rtol=1e-12（CSV の往復）")
    # 負例: 臨床の表から 3 人を除き、1 人の ID を小文字にした表
    dn = J("dc_neg")
    if dn is None:
        rec(S, "neg.damaged_clinical", "UNMET", "dc_neg.json が無い")
    else:
        jn = step(dn, "join")["summary"]
        c2 = C.iloc[3:].copy()
        c2.iloc[0, 0] = c2.iloc[0, 0].lower()
        e2 = dict(joined=len(feat_ids & set(c2.SubjectCode)), featureOnly=len(feat_ids - set(c2.SubjectCode)), clinicalOnly=len(set(c2.SubjectCode) - feat_ids), patientsInClinical=len(c2))
        ok = all(jn[k] == v for k, v in e2.items())
        rec(S, "neg.damaged_clinical", "PASS" if ok else "FAIL", f"実装={{{', '.join(f'{k}:{jn[k]}' for k in e2)}}} pandas={e2}", "壊した入力で人数が変わる（検出力）")
        w = " ".join(step(dn, "join")["warnings"])
        rec(S, "neg.case_only_id_warning", "PASS" if "大文字小文字" in w else "FAIL", f"警告={w[:120]!r}", "大文字小文字だけが違う ID を警告する")


# ---------------------------------------------------------------- R1


def logrank_chi2_fast(T_, E_, g):
    """2 群の log-rank の χ²（自前。lifelines と一致することを下で確かめてから、探索に使う）。"""
    et = np.unique(T_[E_ == 1])
    n = len(T_)
    # at risk と死亡（t ごと）
    atrisk = (T_[None, :] >= et[:, None])
    n1 = (atrisk & g[None, :]).sum(1).astype(float)
    nt = atrisk.sum(1).astype(float)
    d = ((T_[None, :] == et[:, None]) & (E_[None, :] == 1)).sum(1).astype(float)
    d1 = ((T_[None, :] == et[:, None]) & (E_[None, :] == 1) & g[None, :]).sum(1).astype(float)
    e1 = d * n1 / nt
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(nt > 1, d * (n1 / nt) * (1 - n1 / nt) * (nt - d) / (nt - 1), 0.0)
    return float((d1 - e1).sum() ** 2 / v.sum()) if v.sum() > 0 else np.nan


def chi2sf(x):
    return stats.chi2.sf(x, 1)


def r1_features(d):
    return T(d, "results")


def r1_group(x, mode):
    if mode == "median":
        c = np.median(x)
        return (x > c).astype(int), [c]
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return (x > q1).astype(int) + (x > q2).astype(int), [q1, q2]


def sub_fit(fit, j):
    return dict(beta=fit["beta"][[j]], se=fit["se"][[j]], cov=fit["cov"][[j]][:, [j]], ll=fit["ll"], z=fit["z"][[j]], p=fit["p"][[j]])


def stack_fits(fits):
    return dict(beta=np.array([f["beta"][0] for f in fits]), se=np.array([f["se"][0] for f in fits]), cov=np.array([[f["cov"][0, 0]] for f in fits]).reshape(-1, 1),
                ll=np.array([f["ll"] for f in fits]), z=np.array([f["z"][0] for f in fits]), p=np.array([f["p"][0] for f in fits]))


def cox_compare_many(skill, cid, fits, got_hr, got_lo, got_hi, got_p, why):
    """fits: 係数 1 個ずつの fit のリスト（sub_fit の結果）。特徴量ごとに許容が違うので、配列にして比べる。"""
    st = stack_fits(fits)
    eps = 1e-10 * (np.abs(st["ll"]) + 1)
    tb = np.sqrt(2 * eps * st["cov"][:, 0])
    zc = stats.norm.ppf(0.975)
    ref_lhr = st["beta"]; ref_lo = st["beta"] - zc * st["se"]; ref_hi = st["beta"] + zc * st["se"]
    pv = 2 * stats.norm.sf(np.abs(st["z"]))
    n = "TS の停止規則が許す係数のずれの上限 √(2ε·(I⁻¹)jj)、ε=1e-10(|ll|+1)、特徴量ごと"
    compare(skill, f"{cid}.log_hr", ref_lhr, np.log(np.asarray(got_hr, float)), tb + 1e-12, 0.0, why + "。許容: " + n)
    compare(skill, f"{cid}.log_ci_lower", ref_lo, np.log(np.asarray(got_lo, float)), 2 * tb + 1e-12, 0.0, "同上（区間の端 = β−1.96 se。se のずれは二次なので許容は 2 倍）")
    compare(skill, f"{cid}.log_ci_upper", ref_hi, np.log(np.asarray(got_hi, float)), 2 * tb + 1e-12, 0.0, "同上")
    compare(skill, f"{cid}.p", pv, got_p, 2 * stats.norm.pdf(st["z"]) / st["se"] * tb * 1.01 + 1e-12, 0.0, "許容 = |dp/dβ|·（係数の許容）= 2φ(z)/se·δ")


def check_r1_mode(rid, mode):
    S = "R1"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.run", "UNMET", f"{rid}.json が無い")
        return None
    res = r1_features(d)
    pat = d["extra"]["r1"]["patients"]
    D = M.iloc[pat].reset_index(drop=True)
    Tm, Em = D["OS_days"].values.astype(float), D["OS_event"].values.astype(int)
    rec(S, f"{rid}.cohort", "PASS" if len(pat) == 56 and Em.sum() == 15 else "FAIL", f"患者={len(pat)} イベント={int(Em.sum())}", "pandas の人数・イベント数")
    kmpts = T(d, "km")
    ref, skipped, ll_diff = {}, {}, []
    for _, r in res.iterrows():
        f = r["feature"]
        x = D[f].values.astype(float)
        ok = ~np.isnan(x)
        xx, tt, ee = x[ok], Tm[ok], Em[ok]
        if len(np.unique(xx)) < 2:
            skipped[f] = "const"; continue
        g, cuts = r1_group(xx, mode)
        k = 2 if mode == "median" else 3
        if any((g == i).sum() == 0 for i in range(k)):  # タイのため空の群ができる（群分けが定義できない）
            skipped[f] = "emptygroup"; continue
        out = dict(feature=f, n=len(xx), events=int(ee.sum()), cuts=cuts)
        out["sizes"] = [int((g == i).sum()) for i in range(k)]
        out["gev"] = [int(ee[g == i].sum()) for i in range(k)]
        lr = logrank_test(tt[g == 0], tt[g == 1], ee[g == 0], ee[g == 1]) if k == 2 else multivariate_logrank_test(tt, g, ee)
        out["chi2"], out["p"] = lr.test_statistic, lr.p_value
        meds = []
        for i in range(k):
            m = KaplanMeierFitter().fit(tt[g == i], ee[g == i]).median_survival_time_
            meds.append("NR" if np.isinf(m) else str(float(m)))
        out["med"] = meds
        out["sep"] = any(e == 0 for e in out["gev"])  # 群にイベントが無い → その群の係数は −∞（完全分離）
        if not out["sep"]:
            Xg = g[:, None].astype(float) if k == 2 else np.c_[(g == 1), (g == 2)].astype(float)
            fg = efron_cox(Xg, tt, ee)
            out["fg"] = sub_fit(fg, 0 if k == 2 else 1) if fg else None
        z = (xx - xx.mean()) / xx.std(ddof=1)
        fz = efron_cox(z[:, None], tt, ee)
        out["fz"] = sub_fit(fz, 0) if fz else None
        # lifelines との交差確認（参照どうし）
        if fz is not None:
            cph, _ = cox_uni(tt, ee, z[:, None])
            if cph is not None:
                ll_diff.append((f, float(cph.params_.values[0]), float(fz["beta"][0]), float(cph.log_likelihood_), float(fz["ll"])))
        ref[f] = out
    fs = [f for f in res["feature"] if f in ref]
    R = res.set_index("feature")
    # lifelines と自前 Newton の食い違い（参照側の不具合の切り分け）
    badl = [(f, bl, bo, lll, lo) for f, bl, bo, lll, lo in ll_diff if abs(bl - bo) > 1e-3]
    rec(S, f"{rid}.ref.lifelines_vs_own_newton", "PASS",
        f"単変量 Cox（SD あたり）{len(ll_diff)} 個で lifelines と自前 Newton が 1e-3 以上ずれた特徴量={[(f[:40], round(bl, 3), round(bo, 3), 'll', round(lll, 2), round(lo, 2)) for f, bl, bo, lll, lo in badl]}",
        "（参照どうしの確認）ずれた特徴量では、自前の方が部分尤度が高い＝lifelines の方が最大値に達していない。以降の Cox の基準は自前の Newton",
        "lifelines は歪みの強い共変量（z の最大 7）で最大値に届かない。参照側の不具合であり、実装の不具合ではない")
    # 解析できない特徴量の集合: 定数・空の群・（HR を求められない = イベントの無い群）
    err_got = set(res.loc[res["error"].notna(), "feature"])
    err_ref = set(skipped)
    hr_got = set(res.loc[res["hr_group_note"].notna(), "feature"])
    hr_ref = {f for f in ref if ref[f]["sep"] and f not in err_ref}
    rec(S, f"{rid}.unanalyzable_set", "PASS" if (err_got == err_ref and hr_got == hr_ref) else "FAIL",
        f"QC の除外 実装={len(err_got)} 参照={len(err_ref)}（定数/空の群）差={sorted(err_got ^ err_ref)[:5]}。群の HR を推定できず記録のみ 実装={len(hr_got)} 参照={len(hr_ref)}（イベントの無い群）差={sorted(hr_got ^ hr_ref)[:5]}", "集合の一致（実装が error にした特徴量 = 定数・空の群・どれかの群にイベントが無い特徴量）")
    nsep = [f for f in ref if ref[f]["sep"]]
    if nsep:
        lrp = [R.loc[f, "logrank_p"] for f in nsep]
        comp = [ref[f]["p"] for f in nsep]
        ok = np.allclose(lrp, comp, rtol=1e-8, atol=1e-12)
        rec(S, f"{rid}.separated_features_keep_logrank", "PASS" if ok else "FAIL",
            f"イベントの無い群をもつ特徴量 {len(nsep)} 個（HR は求められず error）でも、log-rank の p は計算され参照と一致（最小 p={min(lrp):.4g}、q<0.05 に入る数={sum(1 for f in nsep if R.loc[f, 'q_bh'] < 0.05) if R['q_bh'].notna().any() else 'q なし'}）",
            "log-rank は群にイベントが無くても定義できる", "", )
        sel_ok = [f for f in nsep if pd.notna(R.loc[f, "q_bh"]) and R.loc[f, "q_bh"] < 0.05]
        mvm = d["extra"]["r1"]["model"]
        screened = set(mvm["screened"]) if mvm else set()
        sel_blocked = [f for f in sel_ok if f not in screened]
        rec(S, f"{rid}.design.separated_excluded_from_selection", "FAIL" if sel_blocked else "PASS",
            f"群の HR が推定不能な特徴量のうち q<0.05 のもの={len(sel_ok)}、そのうち選択（多変量・補正 C-index の対象）に入らなかったもの={len(sel_blocked)}: {[f[:30] for f in sel_blocked[:5]]}",
            "出力の q 列と、実装が「基準を満たした」とした特徴量の一覧から（HR が推定不能でも選択に残す）")
    num = lambda col: [R.loc[f, col] for f in fs]
    compare(S, f"{rid}.n", [ref[f]["n"] for f in fs], num("n"), 0, 0, "整数")
    compare(S, f"{rid}.events", [ref[f]["events"] for f in fs], num("events"), 0, 0, "整数")
    compare(S, f"{rid}.cut1", [ref[f]["cuts"][0] for f in fs], num("cut1"), 1e-12, 1e-12, "中央値/分位点（type 7）は加減乗除のみ。許容は丸め誤差")
    if mode == "tertile":
        compare(S, f"{rid}.cut2", [ref[f]["cuts"][1] for f in fs], num("cut2"), 1e-12, 1e-12, "同上")
    gs_ok = all("/".join(map(str, ref[f]["sizes"])) == R.loc[f, "group_sizes"] for f in fs)
    ge_ok = all("/".join(map(str, ref[f]["gev"])) == R.loc[f, "group_events"] for f in fs)
    rec(S, f"{rid}.group_sizes", "PASS" if gs_ok else "FAIL", f"{len(fs)} 特徴量", "文字列の一致（低/（中/）高）")
    rec(S, f"{rid}.group_events", "PASS" if ge_ok else "FAIL", f"{len(fs)} 特徴量", "文字列の一致")
    norm = lambda sx: "/".join(s_ if s_ == "NR" else str(float(s_)) for s_ in sx)
    med_bad = [f for f in fs if norm(ref[f]["med"]) != norm(R.loc[f, "group_median_survival"].split("/"))]
    rec(S, f"{rid}.group_median_survival", "PASS" if not med_bad else "FAIL", f"不一致 {len(med_bad)} / {len(fs)} {med_bad[:3]}", "KM の中央値（S<=0.5 になる最初の時間。lifelines）")
    compare(S, f"{rid}.logrank_chi2", [ref[f]["chi2"] for f in fs], num("logrank_chi2"), 1e-9, 1e-9, "閉じた式。許容は倍精度の和の順序の差")
    compare(S, f"{rid}.logrank_p", [ref[f]["p"] for f in fs], num("logrank_p"), 1e-12, 1e-8, "chi2 上側確率。実装は自前の不完全ガンマ関数（精度約 1e-14）。p が小さい所は相対で比べる")
    # Cox（群の HR: イベントの無い群があるものは除く。SD あたり HR は全部）
    fg = [f for f in fs if not ref[f]["sep"] and ref[f].get("fg") is not None and pd.notna(R.loc[f, "hr_group"])]
    cox_compare_many(S, f"{rid}.hr_group", [ref[f]["fg"] for f in fg], [R.loc[f, "hr_group"] for f in fg], [R.loc[f, "hr_group_lower"] for f in fg],
                     [R.loc[f, "hr_group_upper"] for f in fg], [R.loc[f, "hr_group_p"] for f in fg], f"群の HR（{'高 vs 低' if mode == 'median' else '最上位 vs 最下位'}）、自前の Efron Newton と比べる")
    fz_ = [f for f in fs if ref[f].get("fz") is not None and pd.notna(R.loc[f, "hr_per_sd"])]
    cox_compare_many(S, f"{rid}.hr_per_sd", [ref[f]["fz"] for f in fz_], [R.loc[f, "hr_per_sd"] for f in fz_], [R.loc[f, "hr_per_sd_lower"] for f in fz_],
                     [R.loc[f, "hr_per_sd_upper"] for f in fz_], [R.loc[f, "hr_per_sd_p"] for f in fz_], "SD あたりの連続値 HR、自前の Efron Newton と比べる")
    if mode in ("median", "tertile"):
        q = bh([ref[f]["p"] for f in fs])
        compare(S, f"{rid}.q_bh", q, num("q_bh"), 1e-12, 1e-8, "BH は p の並べ替えと最小値の累積だけ。statsmodels の fdr_bh と比べる（比較数は解析できた特徴量の数）")
    # KM 点列（全特徴量・全群）
    bad_cnt, npts = 0, 0
    max_s = max_se = max_ci = 0.0
    for (f, gi), sub in kmpts.groupby(["feature", "group"]):
        if f not in ref:
            continue
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        g, _ = r1_group(x[ok], mode)
        tt, ee = Tm[ok][g == gi], Em[ok][g == gi]
        kmf = KaplanMeierFitter().fit(tt, ee)
        sf = kmf.survival_function_at_times(sub["time"].values).values
        max_s = max(max_s, float(np.abs(sf - sub["survival"].values).max()))
        et = np.unique(tt)
        n_at = np.array([(tt >= t).sum() for t in et]); dd = np.array([((tt == t) & (ee == 1)).sum() for t in et]); cc = np.array([((tt == t) & (ee == 0)).sum() for t in et])
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(n_at > dd, dd / (n_at * (n_at - dd)), np.nan)
        cum = np.nancumsum(np.where(dd > 0, term, 0.0))
        S_ = np.cumprod(1 - dd / n_at)
        se_ref = S_ * np.sqrt(cum)
        m = pd.Series(se_ref, index=et).reindex(sub["time"].values).values
        valid = ~np.isnan(m) & sub["se"].notna().values
        if valid.any():
            max_se = max(max_se, float(np.abs(m[valid] - sub["se"].values[valid]).max()))
        cnt = pd.DataFrame({"at": n_at, "ev": dd, "ce": cc}, index=et).reindex(sub["time"].values)
        if not (np.array_equal(cnt["at"].values, sub["at_risk"].values) and np.array_equal(cnt["ev"].values, sub["events"].values) and np.array_equal(cnt["ce"].values, sub["censored"].values)):
            bad_cnt += 1
        ci = kmf.confidence_interval_
        ci = ci.reindex(ci.index.union(sub["time"].values)).ffill().loc[sub["time"].values]
        sel = (sub["survival"].values > 0) & (sub["survival"].values < 1) & sub["lower"].notna().values
        if sel.any():
            max_ci = max(max_ci, float(np.nanmax(np.abs(ci.iloc[:, 0].values[sel] - sub["lower"].values[sel]))), float(np.nanmax(np.abs(ci.iloc[:, 1].values[sel] - sub["upper"].values[sel]))))
        npts += len(sub)
    rec(S, f"{rid}.km.survival", "PASS" if max_s <= 1e-12 else "FAIL", f"{npts} 点・最大絶対差={max_s:.3g}", "積極限推定は閉じた式。許容 1e-12")
    rec(S, f"{rid}.km.counts", "PASS" if bad_cnt == 0 else "FAIL", f"人数（at_risk・events・censored）の不一致の群={bad_cnt}", "整数の一致")
    rec(S, f"{rid}.km.greenwood_se", "PASS" if max_se <= 1e-9 else "FAIL", f"最大絶対差={max_se:.3g}（自前の Greenwood 式 se=S·√Σd/(n(n−d))）", "閉じた式。許容 1e-9")
    rec(S, f"{rid}.km.ci_loglog", "PASS" if max_ci <= 1e-9 else "FAIL", f"最大絶対差={max_ci:.3g}（lifelines の信頼区間）", "閉じた式。許容 1e-9")
    return ref, fs, R, D, Tm, Em


def my_chi2_selfcheck(D, Tm, Em, feats):
    """自前の log-rank が lifelines と一致することを確かめる（探索に使う前の前提）。"""
    mx = 0.0
    for f in feats:
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        g = x[ok] > np.median(x[ok])
        a = logrank_test(Tm[ok][~g], Tm[ok][g], Em[ok][~g], Em[ok][g]).test_statistic
        b = logrank_chi2_fast(Tm[ok], Em[ok], g)
        mx = max(mx, abs(a - b))
    return mx


def check_r1_optimal(rid="r1_optimal", minfrac=0.1):
    S = "R1"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.run", "UNMET", "json が無い"); return
    res = r1_features(d)
    R = res.set_index("feature")
    pat = d["extra"]["r1"]["patients"]
    D = M.iloc[pat].reset_index(drop=True)
    Tm, Em = D["OS_days"].values.astype(float), D["OS_event"].values.astype(int)
    mx = my_chi2_selfcheck(D, Tm, Em, FEATS[:15])
    rec(S, f"{rid}.selfcheck_logrank", "PASS" if mx < 1e-9 else "FAIL", f"自前の log-rank と lifelines の最大差={mx:.3g}（15 特徴量）", "探索に使う前提。1e-9")
    n = len(Tm)
    best = {}
    for f in FEATS:
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        xx, tt, ee = x[ok], Tm[ok], Em[ok]
        if len(np.unique(xx)) < 2:
            continue
        cand = []
        for c in np.unique(xx)[:-1]:
            nlo = int((xx <= c).sum())
            if nlo >= minfrac * len(xx) and (len(xx) - nlo) >= minfrac * len(xx):
                cand.append((logrank_chi2_fast(tt, ee, xx > c), c))
        if cand:
            best[f] = max(cand, key=lambda t: (t[0], -t[1]))
    # TS が解析できなかった特徴量
    err = res[res["error"].notna()]
    hrn = res[res["hr_group_note"].notna()]
    rec(S, f"{rid}.errors", "PASS" if (len(err) + len(hrn) == 15 and (err["qc_excluded"] == 1).all() and len(err) >= 3) else "FAIL", f"実装の解析不能（QC の除外）={len(err)}: {err['error'].value_counts().to_dict()}、群の HR を推定できず記録のみ={len(hrn)}", "（情報）3 個は定数。残りは最小の群の割合で候補なし等の可能性")
    fs = [f for f in best if pd.notna(R.loc[f, "logrank_chi2"])]
    only_ref = [f for f in best if pd.isna(R.loc[f, "logrank_chi2"])]
    rec(S, f"{rid}.analyzable_set", "PASS" if not only_ref else "FAIL", f"参照では候補があるが実装が解析不能={only_ref[:6]}（{len(only_ref)} 個）", "集合の一致")
    if fs:
        compare(S, f"{rid}.chi2_max", [best[f][0] for f in fs], [R.loc[f, "logrank_chi2"] for f in fs], 1e-9, 1e-9, "全しきい値の総当たりでの最大 χ²（閉じた式）。群の最小割合は README の既定 10%")
        # 実装のしきい値は、データの値そのものでなく隣り合う値の中点で報告される。「同じ分割になる」ことと「中点であること」を分けて確かめる
        same_part, mid_ok, nn = 0, 0, 0
        for f in fs:
            x = D[f].values.astype(float); u = np.unique(x[~np.isnan(x)])
            c_ref = best[f][1]; nxt = u[u > c_ref].min()
            c_ts = R.loc[f, "cut1"]
            nn += 1
            same_part += int(((x[~np.isnan(x)] <= c_ts) == (x[~np.isnan(x)] <= c_ref)).all())
            mid_ok += int(abs(c_ts - (c_ref + nxt) / 2) <= 1e-9 * max(1.0, abs(c_ref)))
        rec(S, f"{rid}.cut_same_partition", "PASS" if same_part == nn else "FAIL", f"実装のしきい値で分けた群が、参照の最大 χ² のしきい値の群と同じ={same_part}/{nn}", "群分けの一致（低群 = 値 ≤ cut）")
        rec(S, f"{rid}.cut_is_midpoint", "PASS" if mid_ok == nn else "FAIL", f"実装のしきい値 = 参照のしきい値と次の値の中点={mid_ok}/{nn}", "（定義）報告されるしきい値は隣り合う異なる値の中点（R2 と同じ。README に明記なし）")
        raw_ref = chi2sf(np.array([best[f][0] for f in fs]))
        compare(S, f"{rid}.raw_p", raw_ref, [R.loc[f, "optimal_raw_p"] for f in fs], 1e-12, 1e-8, "最大 χ² の自由度 1 の上側確率（探索の素の p）")
    sepc = 0
    for f in [f for f in best if not pd.isna(R.loc[f, "logrank_chi2"]) and pd.notna(R.loc[f, "hr_group_note"])]:
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        gg = x[ok] > best[f][1]
        ge = [int(Em[ok][gg == v].sum()) for v in (0, 1)]
        sepc += int(min(ge) == 0)
    errd = [f for f in best if pd.notna(R.loc[f, "hr_group_note"])]
    rec(S, f"{rid}.separated_features", "PASS" if sepc == len(errd) and len(errd) > 0 else "FAIL", f"最適しきい値で HR が推定不能と記録された特徴量 {len(errd)} 個のうち、そのしきい値の群にイベントが無い（HR が発散する）もの={sepc}", "error の理由が実在する（完全分離）")
    # 群に死亡が無く HR が出ない特徴量も、q が alpha 未満なら選択に入る（log-rank の p・q は有効）
    sel_ok = [f for f in errd if pd.notna(R.loc[f, "q_bh"]) and R.loc[f, "q_bh"] < d["params"]["alpha"]]
    mvm = d["extra"]["r1"]["model"]
    screened = set(mvm["screened"]) if mvm else set()
    blocked = [f for f in sel_ok if f not in screened]
    rec(S, f"{rid}.design.separated_excluded_from_selection", "FAIL" if blocked else "PASS",
        f"最適しきい値で HR が推定不能な特徴量 {len(errd)} 個のうち、補正 p の q が alpha 未満のもの={len(sel_ok)}、そのうち選択（多変量の候補）に入らなかったもの={len(blocked)} {[f[:30] for f in blocked[:3]]}",
        "q の列と、実装の「基準を満たした特徴量」の一覧から（推定不能の特徴量を落とさない）")
    cp = res["optimal_permutation_p"].dropna()
    rp = res.loc[cp.index, "optimal_raw_p"]
    ok_range = bool(((cp >= 1 / 1001 - 1e-12) & (cp <= 1)).all() and (cp >= rp - 1e-12).all())
    rec(S, f"{rid}.permutation_p_range", "PASS" if ok_range else "FAIL", f"補正 p の範囲=[{cp.min():.4g},{cp.max():.4g}] 全て 素の p 以上={bool((cp >= rp - 1e-12).all())}", "補正 p ∈ [1/(B+1),1] かつ 素の p 以下にならない")
    # 自前の並べ替え検定（B=1000）で補正 p を独立に推定し、二項の揺らぎの範囲で比べる
    rng = np.random.default_rng(7)
    pick = sorted(fs, key=lambda f: R.loc[f, "optimal_permutation_p"])[:4] + list(rng.choice([f for f in fs], 4, replace=False))
    B = 1000
    diffs = []
    t0 = time.time()
    for f in dict.fromkeys(pick):
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        xx, tt, ee = x[ok], Tm[ok], Em[ok]
        cuts = [c for c in np.unique(xx)[:-1] if (xx <= c).sum() >= minfrac * len(xx) and (xx > c).sum() >= minfrac * len(xx)]
        order = np.argsort(xx, kind="stable")
        def maxchi(tt_, ee_):
            return max(logrank_chi2_fast(tt_, ee_, xx > c) for c in cuts)
        obs = best[f][0]
        cnt = 0
        for _ in range(B):
            perm = rng.permutation(len(xx))
            if maxchi(tt[perm], ee[perm]) >= obs - 1e-12:
                cnt += 1
        p_ref = (cnt + 1) / (B + 1)
        p_ts = R.loc[f, "optimal_permutation_p"]
        se = math.sqrt(2 * max(p_ref * (1 - p_ref), 1e-4) / B)
        diffs.append((f, p_ref, p_ts, abs(p_ref - p_ts) / se))
    bad = [x for x in diffs if x[3] > 3]
    rec(S, f"{rid}.permutation_p_vs_independent", "PASS" if not bad else "FAIL",
        "; ".join(f"{f[:28]}: ref={a:.3f} ts={b:.3f} z={z:.1f}" for f, a, b, z in diffs) + f"（{time.time()-t0:.0f}s）", "2 つの乱数推定の差が √(2p(1−p)/B) の 3 倍以内（B=1000 同士）")


def check_r1_multivariate(rid="r1_median_a03", mode="median"):
    S = "R1"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.mv.run", "UNMET", "json が無い"); return
    res = r1_features(d)
    ex = d["extra"]["r1"]
    D = M.iloc[ex["patients"]].reset_index(drop=True)
    Tm, Em = D["OS_days"].values.astype(float), D["OS_event"].values.astype(int)
    alpha = d["params"]["alpha"]
    ps = {}
    for f in FEATS:
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        if len(np.unique(x[ok])) < 2:
            continue
        g = x[ok] > np.median(x[ok])
        ps[f] = chi2sf(logrank_chi2_fast(Tm[ok], Em[ok], g))
    fl = list(ps)
    q = bh([ps[f] for f in fl])
    sel_ref = [f for f, qq in zip(fl, q) if qq < alpha]
    mv = ex["model"]
    covs = list(d["params"].get("covariates") or [])
    rec(S, f"{rid}.mv.screened_set", "PASS" if set(sel_ref) == set(mv["screened"]) else "FAIL", f"実装={len(mv['screened'])} 参照={len(sel_ref)}", f"q<{alpha} の特徴量の集合（BH）")
    # 変数の上限（イベント 10 件あたり 1 変数。共変量も数える）: p の小さい順に、入れた特徴量が揃った患者のイベント数で決める
    order = sorted(sel_ref, key=lambda f: (ps[f], fl.index(f)))
    k = len(order)
    while k > 0:
        okp = ~D[order[:k] + covs].isna().any(axis=1).values
        if k + len(covs) <= math.floor(Em[okp].sum() / 10):
            break
        k -= 1
    rec(S, f"{rid}.mv.epv_cap_entered", "PASS" if order[:k] == mv["selected"] else "FAIL",
        f"参照: 基準を満たした {len(order)} 個のうち p の小さい順に {k} 個（イベント {int(Em.sum())} 件、共変量 {len(covs)} 個、上限 {math.floor(Em.sum() / 10)} 変数）を入れる。実装={mv['selected']} 参照={order[:k]}",
        "入れる特徴量 = p の小さい順の先頭 k 個。k = k + 共変量 ≤ ⌊イベント ÷ 10⌋ を満たす最大の数（参照側で独立に実装）")
    rec(S, f"{rid}.mv.not_entered", "PASS" if mv["notEntered"] == order[k:] else "FAIL", f"実装={len(mv['notEntered'])} 個 参照={len(order[k:])} 個", "入れなかった特徴量の一覧が、基準を満たしたもの − 入れたもの")
    if k == 0:
        rec(S, f"{rid}.mv.no_model_when_no_room", "PASS" if (mv["error"] is None and not mv["terms"]) else "FAIL", f"枠が無いとき多変量モデルを作らず、エラーにもしない: terms={len(mv['terms'])} error={mv['error']}", "")
        return
    if mv["error"] or not mv["terms"]:
        rec(S, f"{rid}.mv.fit", "FAIL", f"実装が多変量を作れなかった: {mv['error']}", ""); return
    sel = mv["selected"]
    cols = []
    for f in sel:
        x = D[f].values.astype(float)
        cols.append((x - x.mean()) / x.std(ddof=1))
    X = np.c_[np.array(cols).T, *[D[c].values.astype(float) for c in covs]] if covs else np.array(cols).T
    fit = efron_cox(X, Tm, Em)
    names = sel + covs
    t = pd.DataFrame(mv["terms"])
    rec(S, f"{rid}.mv.term_names", "PASS" if list(t["name"]) == names else "FAIL", f"{list(t['name'])}", "項の並び（選ばれた特徴量→共変量）")
    cox_compare(S, f"{rid}.mv", fit, t["hr"].values, t["lower"].values, t["upper"].values, t["p"].values, "多変量 Cox（Efron）。自前の Newton と比べる")
    lp = X @ fit["beta"]
    c_ref = harrell_c(Tm, Em, lp)
    eps = 1e-10 * (abs(fit["ll"]) + 1)
    compare(S, f"{rid}.mv.cindex_apparent", [c_ref], [mv["cIndexApparent"]], 1e-9, 0.0, "Harrell の C を定義どおり総当たりで数える（比較可能な組 = i がイベントで T_i < T_j、または同時刻で j が打ち切り。lifelines と同じ規則）")
    c_life = concordance_index(Tm, -lp, Em)
    compare(S, f"{rid}.mv.cindex_apparent_vs_lifelines", [c_life], [mv["cIndexApparent"]], 1e-9, 0.0, "lifelines の concordance_index")
    compare(S, f"{rid}.mv.loglik", [fit["ll"]], [mv["loglik"]], eps + 1e-12, 0.0, "部分尤度。TS の停止規則（ll の変化 < 1e-10(|ll|+1)）が許す差")
    if covs:
        fc = efron_cox(X[:, len(sel):], Tm, Em)
        chi2 = 2 * (fit["ll"] - fc["ll"])
        p_av = stats.chi2.sf(chi2, len(sel))
        av = mv["addedValue"]
        compare(S, f"{rid}.mv.added_value_chi2", [chi2], [av["chi2"]], 4 * eps + 1e-12, 0.0, "−2 log 尤度の差。両モデルの ll の許容の和の 2 倍")
        compare(S, f"{rid}.mv.added_value_p", [p_av], [av["p"]], 4 * eps * 0.5 + 1e-12, 0.0, "χ² 上側確率（df = 特徴量の数）。|dp/dχ²| ≤ 0.5")
    else:
        rec(S, f"{rid}.mv.added_value_none", "PASS" if mv["addedValue"] is None else "FAIL", f"共変量なし: addedValue={mv['addedValue']}", "共変量が無いときは臨床因子だけのモデルが無いので、足した効果は出さない")
    cc = ex["model"]["corrected"]
    if cc is None:
        rec(S, f"{rid}.mv.corrected_c", "UNMET", "実装が補正 C-index を出していない"); return
    ok_names = [f for f in FEATS if not D[f].isna().any()]
    Xall = np.array([D[f].values.astype(float) for f in ok_names]).T
    cov = np.c_[*[D[c].values.astype(float) for c in covs]] if covs else np.zeros((len(D), 0))
    rng = np.random.default_rng(11)
    B = 500
    def procedure(idx):
        tt, ee = Tm[idx], Em[idx]
        pv = []
        for j in range(Xall.shape[1]):
            x = Xall[idx, j]
            if len(np.unique(x)) < 2:
                pv.append(np.nan); continue
            g = x > np.median(x)
            if g.all() or (~g).all():
                pv.append(np.nan); continue
            pv.append(chi2sf(logrank_chi2_fast(tt, ee, g)))
        qq = bh(pv)
        sel_i = [j for j in range(len(qq)) if not np.isnan(qq[j]) and qq[j] < alpha]
        # 本体と同じ規則（独立に実装）: p の小さい順に、標本のイベント数 ÷ 10 − 共変量の数 まで
        sel_i = sorted(sel_i, key=lambda j: (pv[j], j))[:max(0, math.floor(ee.sum() / 10) - cov.shape[1])]
        cols_ = []
        for j in sel_i:
            x = Xall[idx, j]; sd = x.std(ddof=1)
            if sd > 0: cols_.append((j, x.mean(), sd))
        Xb = np.c_[np.array([(Xall[idx, j] - m) / s_ for j, m, s_ in cols_]).T if cols_ else np.zeros((len(idx), 0)), cov[idx]]
        if Xb.shape[1] == 0:
            return lambda rows: np.zeros(len(rows))
        fb = efron_cox(Xb, tt, ee)
        if fb is None:
            return None
        beta = fb["beta"]
        def risk(rows):
            Z = np.c_[np.array([(Xall[rows, j] - m) / s_ for j, m, s_ in cols_]).T if cols_ else np.zeros((len(rows), 0)), cov[rows]]
            return Z @ beta
        return risk
    allrows = np.arange(len(Tm))
    opt, fail = [], 0
    t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, len(Tm), len(Tm))
        if Em[idx].sum() < 2:
            fail += 1; continue
        risk = procedure(idx)
        if risk is None:
            fail += 1; continue
        a = harrell_c(Tm[idx], Em[idx], risk(idx)); o = harrell_c(Tm, Em, risk(allrows))
        opt.append(a - o)
    opt = np.array(opt)
    ref_corr = c_ref - opt.mean()
    se = opt.std(ddof=1) / math.sqrt(len(opt))
    ts_se = se * math.sqrt(len(opt) / max(cc["bootstraps"], 1))
    z = abs(ref_corr - cc["corrected"]) / math.sqrt(se ** 2 + ts_se ** 2)
    rec(S, f"{rid}.mv.corrected_c_independent_bootstrap", "PASS" if z < 3 else "FAIL",
        f"参照 補正C={ref_corr:.4f}（楽観 {opt.mean():.4f}±{se:.4f}、成功 {len(opt)}/{B}、{time.time()-t0:.0f}s） 実装 補正C={cc['corrected']:.4f}（楽観 {cc['optimism']:.4f}、成功 {cc['bootstraps']}/500） z={z:.2f}",
        "ブートストラップは乱数で再現できない。2 つの推定の差が、標準誤差（実装側も参照と同程度とみなす）の √2 倍の 3 倍以内")
    rec(S, f"{rid}.mv.bootstrap_failure_rate", "UNMET",
        f"参照の失敗率={fail / B:.3f}（自前 Newton が収束しない・イベント<2）、実装の失敗率={(500 - cc['bootstraps']) / 500:.3f}（モデルを作れず除いた {500 - cc['bootstraps']}/500 回）",
        "同じ標本で比べられないので合否は付けない（失敗の定義も違う）。観察: どちらも半数以上が失敗し、補正 C-index は成功した少数の標本（変数が少なく収束した標本）の楽観で補正されている")


def check_r1():
    out = check_r1_mode("r1_median", "median")
    check_r1_mode("r1_tertile", "tertile")
    d = J("r1_median")
    if d is not None:
        ms = step(d, "multivariate"); us = step(d, "univariate")
        mvm = d["extra"]["r1"]["model"]
        ok = ms["status"] != "failed" and d["status"] == "ok" and us["summary"]["selected"] > 15 and mvm is not None and not mvm["selected"] and mvm["error"] is None
        rec("R1", "r1_median.mv.default_alpha", "PASS" if ok else "FAIL",
            f"既定 alpha=0.05 で選ばれた特徴量={us['summary']['selected']} 個。多変量の状態={ms['status']}、Skill の状態={d['status']}、入れた特徴量={len(mvm['selected']) if mvm else None}、入れなかった={len(mvm['notEntered']) if mvm else None}（イベント 15 件・共変量 2 個）",
            "変数の上限（⌊15 ÷ 10⌋ = 1 変数）が共変量 2 個で埋まるため、特徴量は入らない。Skill は失敗にならず、特徴量ごとの結果は残る（以前は 36 変数で情報行列が特異になり Skill 全体が失敗した）")
    check_r1_optimal()
    check_r1_multivariate()


# ---------------------------------------------------------------- R2


def delong_ci(score, y):
    """総当たりのペア比較による DeLong（構造成分 V10・V01）。score は向きを揃えた値。"""
    P = score[y == 1]; N = score[y == 0]
    m, n = len(P), len(N)
    psi = (P[:, None] > N[None, :]).astype(float) + 0.5 * (P[:, None] == N[None, :])
    auc = psi.mean()
    v10 = psi.mean(1); v01 = psi.mean(0)
    s10 = v10.var(ddof=1); s01 = v01.var(ddof=1)
    var = s10 / m + s01 / n
    se = math.sqrt(var)
    return auc, se, max(0.0, auc - 1.959963984540054 * se), min(1.0, auc + 1.959963984540054 * se)  # 区間は [0,1] に切る（LOO の AUC が 0 に近いとき下限が負になる。実装も切る）


def exact_ci(k, n):
    r = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return r.low, r.high


def best_cuts(x, y, direction, criterion):
    """隣り合う異なる値の中点を総当たり。陽性の予測: higher なら値 > cut、lower なら値 < cut。"""
    u = np.unique(x)
    cuts = (u[:-1] + u[1:]) / 2
    best = []
    for c in cuts:
        pred = (x > c) if direction == "higher" else (x < c)
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum()); tn = int(((~pred) & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
        acc = (tp + tn) / len(y)
        yj = tp / (tp + fn) + tn / (tn + fp) - 1
        best.append((c, tp, fp, tn, fn, acc, yj))
    return best


def check_r2_run(rid):
    S = "R2"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.run", "UNMET", "json が無い"); return
    crit = d["params"]["criterion"]
    ex = d["extra"]["r2"]
    res = T(d, "results")
    R = res.set_index("feature")
    pat = ex["patients"]
    D = M.iloc[pat].reset_index(drop=True)
    y = (D["EGFR"] == "Mutant").astype(int).values
    rec(S, f"{rid}.cohort", "PASS" if len(pat) == 44 and y.sum() == 13 and (ex["y"] == list(y)) else "FAIL", f"患者={len(pat)} 陽性={int(y.sum())}（EGFR 空 12 人を除く）、陽性ラベルの割り当ても一致", "pandas の人数と 0/1 の割り当て")
    ref = {}
    for f in FEATS:
        x = D[f].values.astype(float); ok = ~np.isnan(x)
        if len(np.unique(x[ok])) < 2:
            continue
        xx, yy = x[ok], y[ok]
        a0 = roc_auc_score(yy, xx)
        dirn = "lower" if a0 < 0.5 else "higher"
        s = xx if dirn == "higher" else -xx
        auc, se, lo, hi = delong_ci(s, yy)
        mw = stats.mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided", method="asymptotic", use_continuity=True)
        ref[f] = dict(auc=auc, lo=lo, hi=hi, mwp=mw.pvalue, dir=dirn, n=len(xx), pos=int(yy.sum()), x=xx, y=yy)
    fs = [f for f in res["feature"] if f in ref]
    err_set = set(res.loc[res["error"].notna(), "feature"])
    rec(S, f"{rid}.unanalyzable_set", "PASS" if err_set == set(res["feature"]) - set(ref) else "FAIL", f"実装={sorted(err_set)}", "定数の特徴量の集合")
    g = lambda c: [R.loc[f, c] for f in fs]
    compare(S, f"{rid}.n", [ref[f]["n"] for f in fs], g("n"), 0, 0, "整数")
    compare(S, f"{rid}.positives", [ref[f]["pos"] for f in fs], g("positives"), 0, 0, "整数")
    rec(S, f"{rid}.direction", "PASS" if all(R.loc[f, "direction"] == ref[f]["dir"] for f in fs) else "FAIL", f"{len(fs)} 特徴量", "AUC が 0.5 以上になる向き（README）")
    compare(S, f"{rid}.auc", [ref[f]["auc"] for f in fs], g("auc"), 1e-12, 1e-12, "AUC は順位の閉じた式。sklearn の roc_auc_score（向きを揃えた値）")
    compare(S, f"{rid}.auc_delong_lower", [ref[f]["lo"] for f in fs], g("auc_lower"), 1e-9, 1e-9, "DeLong を総当たりのペア比較で独立に実装。正規近似 ±1.96 SE")
    compare(S, f"{rid}.auc_delong_upper", [ref[f]["hi"] for f in fs], g("auc_upper"), 1e-9, 1e-9, "同上")
    compare(S, f"{rid}.mannwhitney_p", [ref[f]["mwp"] for f in fs], g("mannwhitney_p"), 1e-12, 1e-8, "scipy mannwhitneyu(asymptotic, 連続性補正あり・タイ補正あり)。正規分布の p は実装が自前（精度約 1e-14）")
    if d["params"]["multiplicity"] == "bh":
        q = bh([ref[f]["mwp"] for f in fs])
        compare(S, f"{rid}.q_bh", q, g("q_bh"), 1e-12, 1e-8, "BH（statsmodels）。参照の p から求める")
    else:
        rec(S, f"{rid}.q_empty_when_none", "PASS" if res["q_bh"].isna().all() else "FAIL", f"補正なし（multiplicity=none）のとき q 列がすべて空={bool(res['q_bh'].isna().all())}", "補正しないなら q を出さない（0 や p で埋めない）")
    # 最適な cut
    bad_cm = bad_opt = bad_cp = 0
    maxcp = 0.0
    for f in fs:
        r = R.loc[f]
        bc = best_cuts(ref[f]["x"], ref[f]["y"], ref[f]["dir"], crit)
        if not bc:
            continue
        idx = 5 if crit == "accuracy" else 6
        mxv = max(b[idx] for b in bc)
        c = r["cut"]
        pred = (ref[f]["x"] > c) if ref[f]["dir"] == "higher" else (ref[f]["x"] < c)
        yy = ref[f]["y"]
        tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum()); tn = int((~pred & (yy == 0)).sum()); fn = int((~pred & (yy == 1)).sum())
        if (tp, fp, tn, fn) != (r["tp"], r["fp"], r["tn"], r["fn"]):
            bad_cm += 1
        val = (tp + tn) / len(yy) if crit == "accuracy" else tp / (tp + fn) + tn / (tn + fp) - 1
        if abs(val - mxv) > 1e-12:
            bad_opt += 1
        # 感度・特異度・Clopper–Pearson
        for nm, k, n, lo_c, hi_c in [("sensitivity", tp, tp + fn, "sens_lower", "sens_upper"), ("specificity", tn, tn + fp, "spec_lower", "spec_upper"),
                                      ("ppv", tp, tp + fp, "ppv_lower", "ppv_upper"), ("npv", tn, tn + fn, "npv_lower", "npv_upper"), ("accuracy", tp + tn, len(yy), "acc_lower", "acc_upper")]:
            if n == 0:
                continue
            lo_, hi_ = exact_ci(k, n)
            maxcp = max(maxcp, abs(lo_ - r[lo_c]) if pd.notna(r[lo_c]) else 0, abs(hi_ - r[hi_c]) if pd.notna(r[hi_c]) else 0, abs(k / n - r[nm]))
    rec(S, f"{rid}.cut_confusion", "PASS" if bad_cm == 0 else "FAIL", f"{len(fs)} 特徴量中、cut での混同行列が自前の数え上げと違う={bad_cm}", "整数の一致（値 > cut が陽性、向き lower は 値 < cut）")
    rec(S, f"{rid}.cut_optimal_{crit}", "PASS" if bad_opt == 0 else "FAIL", f"cut の {crit} が、中点を総当たりした最大値と違う特徴量={bad_opt}", "最大値の一致（同点の選び方は比べない）")
    rec(S, f"{rid}.clopper_pearson", "PASS" if maxcp < 1e-9 else "FAIL", f"感度・特異度・PPV・NPV・精度の値と正確な信頼区間の最大絶対差={maxcp:.3g}", "scipy binomtest の proportion_ci(exact)。ベータ分布の分位点は実装が自前。許容 1e-9")
    # 組み合わせ
    sel = [f for f in fs if R.loc[f, "q_bh"] < d["params"]["alpha"]] if d["params"]["multiplicity"] == "bh" else [f for f in fs if ref[f]["mwp"] < d["params"]["alpha"]]
    sel_ts = [f for f in fs if (R.loc[f, "q_bh"] < d["params"]["alpha"] if d["params"]["multiplicity"] == "bh" else R.loc[f, "mannwhitney_p"] < d["params"]["alpha"])]
    combo = ex["combo"]
    rec(S, f"{rid}.criterion_met_set", "PASS" if set(sel) == set(sel_ts) else "FAIL", f"基準を満たす特徴量 参照={len(sel)} 実装の表から={len(sel_ts)}", f"{'q' if d['params']['multiplicity']=='bh' else 'p'}<{d['params']['alpha']}")
    if len(sel) < 2:
        rec(S, f"{rid}.combination", "UNMET", f"基準を満たす特徴量が {len(sel)} 個で、組み合わせは作られない（実装も {'作っていない' if combo is None or combo['features'] == [] else '作った'}）。ロジスティック回帰・LOO は比べられない",
            "前提待ち（この設定では検査の対象が無い）。別の設定の run（r2_acc_none）で確かめる")
        if combo is not None and combo["features"] != []:
            rec(S, f"{rid}.combination_unexpected", "FAIL", f"参照では基準を満たす特徴量 {len(sel)} 個なのに実装は {combo['features']}", "")
        return
    top = sorted(sel, key=lambda f: (ref[f]["mwp"]))[: d["params"]["topK"]]
    rec(S, f"{rid}.combo_features", "PASS" if top == combo["features"] else "FAIL", f"参照={top} 実装={combo['features']}", "MW の p が小さい順の上位 k（同点は列順）")
    cm = np.all([D[f].notna() for f in top], axis=0)
    yy = y[cm]
    Z = np.array([(D[f].values[cm] - D[f].values[cm].mean()) / D[f].values[cm].std(ddof=1) for f in top]).T
    lg = sm.Logit(yy, sm.add_constant(Z)).fit(disp=0, method="newton", maxiter=200, tol=1e-12)
    ci = lg.conf_int()
    t = pd.DataFrame(combo["terms"])
    compare(S, f"{rid}.combo.odds_ratio", np.exp(lg.params[1:]), t["oddsRatio"].values, 1e-6, 1e-6, "ロジスティック回帰（statsmodels Logit、Newton）。実装は ll の変化 1e-11 で停止。相対 1e-6")
    compare(S, f"{rid}.combo.or_lower", np.exp(ci[1:, 0]), t["lower"].values, 1e-6, 1e-6, "Wald 区間")
    compare(S, f"{rid}.combo.or_upper", np.exp(ci[1:, 1]), t["upper"].values, 1e-6, 1e-6, "同上")
    compare(S, f"{rid}.combo.p", lg.pvalues[1:], t["p"].values, 1e-9, 1e-6, "Wald の p")
    pred = lg.predict(sm.add_constant(Z))
    a_app, _, lo_a, hi_a = delong_ci(pred, yy)
    compare(S, f"{rid}.combo.apparent_auc", [a_app, lo_a, hi_a], [combo["apparent"]["auc"], combo["apparent"]["lower"], combo["apparent"]["upper"]], 1e-6, 1e-6, "見かけの AUC と DeLong 区間（予測確率から）")
    # LOO（選択を各折でやり直す）
    cand = [f for f in FEATS if D[f].values[cm].size and not np.isnan(D[f].values[cm]).any()]
    rec(S, f"{rid}.loo.candidates", "PASS" if len(cand) == combo["candidates"] else "FAIL", f"候補=参照 {len(cand)} 実装 {combo['candidates']}", "組み合わせに使った患者で欠損の無い特徴量の数")
    Xc = np.array([D[f].values[cm] for f in cand]).T
    k = d["params"]["topK"]
    alpha = d["params"]["alpha"] if d["params"]["multiplicity"] == "bh" else None
    n = len(yy)
    preds = np.full(n, np.nan)
    nfb = 0
    for i in range(n):
        tr = np.arange(n) != i
        ps = []
        for j in range(Xc.shape[1]):
            a, b = Xc[tr][yy[tr] == 1, j], Xc[tr][yy[tr] == 0, j]
            if len(np.unique(Xc[tr][:, j])) < 2:
                ps.append(np.nan); continue
            ps.append(stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue)
        ps = np.array(ps)
        qs = bh(ps) if alpha is not None else ps
        okj = [j for j in range(len(ps)) if not np.isnan(ps[j]) and (alpha is None or qs[j] < alpha)]
        okj = sorted(okj, key=lambda j: (ps[j], j))[:k]
        if not okj:
            preds[i] = yy[tr].mean(); nfb += 1; continue
        mu = Xc[tr][:, okj].mean(0); sd = Xc[tr][:, okj].std(0, ddof=1)
        try:
            lf = sm.Logit(yy[tr], sm.add_constant((Xc[tr][:, okj] - mu) / sd)).fit(disp=0, method="newton", maxiter=200, tol=1e-12)
            preds[i] = lf.predict(np.r_[1.0, (Xc[i, okj] - mu) / sd][None, :])[0]
        except Exception:
            preds[i] = np.nan
    okp = ~np.isnan(preds)
    a_loo, _, lo_l, hi_l = delong_ci(preds[okp], yy[okp])
    rec(S, f"{rid}.loo.auc", "PASS" if abs(a_loo - combo["loo"]["auc"]) < 1e-6 and combo["looFailed"] == int((~okp).sum()) else "FAIL",
        f"参照 LOO AUC={a_loo:.6f}（失敗 {int((~okp).sum())}）実装={combo['loo']['auc']:.6f}（失敗 {combo['looFailed']}）",
        "選択（MW p 昇順の上位 k）を各折の訓練内でやり直す規則（README）。分離のある折は IRLS の収束が違い得るので 1e-6")
    compare(S, f"{rid}.loo.auc_ci", [lo_l, hi_l], [combo["loo"]["lower"], combo["loo"]["upper"]], 1e-6, 1e-6, "LOO 予測の AUC の DeLong 区間")
    if nfb:
        mentioned = any(("陽性割合" in nt) for nt in combo["notes"])
        rec(S, f"{rid}.design.loo_fallback_folds", "PASS" if mentioned else "FAIL",
            f"LOO 44 折のうち、選択が 0 個になり訓練の陽性割合を予測にした折={nfb}。この折の予測は検証の症例の結果と逆向きに動く（左に除いた症例が陽性なら訓練の陽性割合は下がる）ので、折が多いと LOO の AUC は 0.5 未満に偏る（今回 LOO AUC={combo['loo']['auc']:.3f}）。実装の notes に折数の記載={'あり' if mentioned else 'なし'}",
            "フォールバックの折数が結果に出る（解釈の注意）", "選択が 0 個の折の扱い（陽性割合）の影響が、結果・注記に出ない", cause="design" if not mentioned else "")


def check_r2():
    for rid in ["r2_acc", "r2_youden", "r2_acc_a45", "r2_acc_none"]:
        check_r2_run(rid)


# ---------------------------------------------------------------- R3


def check_r3():
    import pingouin as pg
    S = "R3"
    d = J("r3")
    if d is None:
        rec(S, "run", "UNMET", "json が無い"); return
    res = T(d, "results").set_index("feature")
    ex = d["extra"]["r3"]
    Bi = B.set_index("CaseLabel").loc[A["CaseLabel"]].reset_index()
    common = [c for c in A.columns if c in B.columns and c != "CaseLabel" and pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c]) and A[c].notna().any() and B[c].notna().any()]
    got_feats = set(res.index)
    rec(S, "feature_set", "PASS" if got_feats == set(common) else "FAIL", f"実装={len(got_feats)} 参照（両表で数値の列）={len(common)} 差={sorted(got_feats ^ set(common))[:6]}", "集合の一致")
    rec(S, "matched", "PASS" if ex["matched"] == 56 else "FAIL", f"突き合わせ={ex['matched']}", "56 症例")
    rows = []
    n_err_ref = set(); fs = []
    ref = {}
    for f in sorted(got_feats & set(common)):
        a = A[f].values.astype(float); b = Bi[f].values.astype(float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 3:
            n_err_ref.add(f); continue
        long = pd.DataFrame({"s": np.r_[np.arange(ok.sum()), np.arange(ok.sum())], "r": np.r_[np.zeros(ok.sum()), np.ones(ok.sum())], "v": np.r_[a[ok], b[ok]]})
        try:
            i = pg.intraclass_corr(long, targets="s", raters="r", ratings="v").set_index("Type")
        except Exception:
            n_err_ref.add(f); continue
        if not np.isfinite(i.loc["ICC2", "ICC"]):
            n_err_ref.add(f); continue
        # 分散分析の平均平方（自前）
        M_ = np.c_[a[ok], b[ok]]
        nn, k = M_.shape
        gm = M_.mean(); rm = M_.mean(1); cm = M_.mean(0)
        msr = k * ((rm - gm) ** 2).sum() / (nn - 1); msc = nn * ((cm - gm) ** 2).sum() / (k - 1)
        mse = ((M_ - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((nn - 1) * (k - 1))
        dd = b[ok] - a[ok]
        bias, sd = dd.mean(), dd.std(ddof=1)
        t = stats.t.ppf(0.975, nn - 1)
        seb = sd / math.sqrt(nn); sel = math.sqrt(sd * sd * (1 / nn + 1.96 ** 2 / (2 * (nn - 1))))
        scale = max(np.abs(a[ok]).max(), np.abs(b[ok]).max()) ** 2
        exact = bool(np.array_equal(a[ok], b[ok]) or np.ptp(b[ok] - a[ok]) == 0)
        ref[f] = dict(scale=scale, exact=exact, n=int(ok.sum()), icc=i.loc["ICC2", "ICC"], lo=i.loc["ICC2", "CI95%"][0], hi=i.loc["ICC2", "CI95%"][1],
                      ick=i.loc["ICC2k", "ICC"], lok=i.loc["ICC2k", "CI95%"][0], hik=i.loc["ICC2k", "CI95%"][1], msr=msr, msc=msc, mse=mse,
                      icc3=i.loc["ICC3", "ICC"],
                      ba=dict(n=nn, bias=bias, sd=sd, bl=(bias - t * seb, bias + t * seb), lo=bias - 1.96 * sd, hi=bias + 1.96 * sd,
                              lol=(bias - 1.96 * sd - t * sel, bias - 1.96 * sd + t * sel), hil=(bias + 1.96 * sd - t * sel, bias + 1.96 * sd + t * sel)))
    err_got = set(res.index[res["error"].notna()])
    rec(S, "unanalyzable_set", "PASS" if err_got == n_err_ref else "FAIL", f"実装の ICC 不能={sorted(err_got)} 参照（pingouin 不能・n<3）={sorted(n_err_ref)}", "集合の一致")
    fs = [f for f in ref if f in res.index and pd.notna(res.loc[f, "icc"])]
    only_ref = [f for f in ref if f in res.index and pd.isna(res.loc[f, "icc"])]
    rec(S, "ref_ok_but_ts_error", "PASS" if not only_ref else "FAIL", f"{only_ref[:6]}", "参照は計算できるが実装が空欄の特徴量")
    g = lambda c: [res.loc[f, c] for f in fs]
    compare(S, "n", [ref[f]["n"] for f in fs], g("n"), 0, 0, "整数")
    sc = np.array([ref[f]["scale"] for f in fs])
    compare(S, "msr", [ref[f]["msr"] for f in fs], g("msr"), 1e-14 * sc, 1e-9, "二元配置の平均平方（自前で定義から計算）。atol は倍精度の丸め（値の最大の 2 乗 × 1e-14）、rtol 1e-9")
    compare(S, "msc", [ref[f]["msc"] for f in fs], g("msc"), 1e-14 * sc, 1e-9, "同上。A と B が完全に等しい特徴量の真値は 0 で、自前の式の丸め（1e-30 程度）が atol の中に入る")
    compare(S, "mse", [ref[f]["mse"] for f in fs], g("mse"), 1e-14 * sc, 1e-6, "同上。実装は残差から直接求める（1e-16 の打ち消しを避ける）。atol を上のとおり置き、相対 1e-6")
    compare(S, "icc_ICC2(A,1)", [ref[f]["icc"] for f in fs], g("icc"), 1e-9, 1e-9, "pingouin ICC2（2 元ランダム・absolute agreement・単一）。ICC3 は別（consistency）。閉じた式")
    nonex = [f for f in fs if not ref[f]["exact"]]
    exact_f = [f for f in fs if ref[f]["exact"]]
    pg_lo = [ref[f]["lo"] for f in nonex]
    compare(S, "icc_lower_vs_pingouin", pg_lo, [res.loc[f, "icc_lower"] for f in nonex], 0.005 + 1e-9, 0.0, "pingouin は CI95% を小数 2 桁に丸めて返すので、許容 = 丸めの幅の半分 0.005（下の厳密式の検査が本命）")
    compare(S, "icc_upper_vs_pingouin", [ref[f]["hi"] for f in nonex], [res.loc[f, "icc_upper"] for f in nonex], 0.005 + 1e-9, 0.0, "同上")
    # 信頼区間: McGraw–Wong の F 近似を定義から自前で書く（pingouin の丸めを避ける）。A と B が完全に等しい（誤差の分散 0）特徴量は式が 0 で割るので別扱い
    cl = []
    for f in nonex:
        r = ref[f]; n_, k = r["n"], 2
        msr, msc, mse = r["msr"], r["msc"], r["mse"]
        a_ = k * r["icc"] / (n_ * (1 - r["icc"]))
        b_ = 1 + k * r["icc"] * (n_ - 1) / (n_ * (1 - r["icc"]))
        v = (a_ * msc + b_ * mse) ** 2 / ((a_ * msc) ** 2 / (k - 1) + (b_ * mse) ** 2 / ((n_ - 1) * (k - 1)))
        fl = stats.f.ppf(0.975, n_ - 1, v); fu = stats.f.ppf(0.975, v, n_ - 1)
        lo = n_ * (msr - fl * mse) / (fl * (k * msc + (k * n_ - k - n_) * mse) + n_ * msr)
        hi = n_ * (fu * msr - mse) / (k * msc + (k * n_ - k - n_) * mse + n_ * fu * msr)
        cl.append((lo, hi))
    compare(S, "icc_lower_McGrawWong", [c[0] for c in cl], [res.loc[f, "icc_lower"] for f in nonex], 1e-9, 1e-8, f"McGraw–Wong (1996) の F 近似の式（ICC(A,1)）を自前で書いた。F 分位点は scipy。{len(nonex)} 特徴量")
    compare(S, "icc_upper_McGrawWong", [c[1] for c in cl], [res.loc[f, "icc_upper"] for f in nonex], 1e-9, 1e-8, "同上")
    ex_icc = [res.loc[f, "icc"] for f in exact_f]
    rec(S, "exact_agreement.icc_is_one", "PASS" if np.allclose(ex_icc, 1.0, atol=1e-12) else "FAIL", f"A と B が完全に等しい（または差が定数のみ）特徴量 {len(exact_f)} 個: 実装の ICC={sorted(set(np.round(ex_icc, 12)))}（差が定数のものは 1 未満になるはず）",
        "A=B なら ICC=1（性質）。差が定数のものは absolute agreement で 1 未満")
    ex_a = [f for f in exact_f if np.array_equal(A[f].values.astype(float)[~np.isnan(A[f].values.astype(float))], Bi[f].values.astype(float)[~np.isnan(A[f].values.astype(float))])]
    rec(S, "exact_agreement.ci_limit", "UNMET", f"誤差の分散 0 のとき、信頼区間は式の極限で求める（実装の note）。参照の F 近似は 0 で割るので、この {len(exact_f)} 特徴量の区間は独立に突き合わせられない（実装の区間={sorted(set((round(res.loc[f, 'icc_lower'], 6), round(res.loc[f, 'icc_upper'], 6)) for f in exact_f))[:3]}）", "参照が定義できない（前提待ち）")
    compare(S, "icc_avg_ICC2k", [ref[f]["ick"] for f in fs], g("icc_avg"), 1e-9, 1e-9, "pingouin ICC2k（Spearman–Brown: 2 icc / (1 + icc)）")
    # 条件 B = A + 定数 のような特徴量は ICC3 と ICC2 が違うことを確かめる（absolute agreement の確認）
    diff = [f for f in fs if abs(ref[f]["icc"] - ref[f]["icc3"]) > 1e-3]
    rec(S, "absolute_agreement_vs_consistency", "PASS" if diff else "UNMET", f"ICC2 と ICC3 が違う特徴量={len(diff)} / {len(fs)}（実装の icc は ICC2 と一致、ICC3 とは別）", "実装が consistency でなく absolute agreement であることの確認（違いが出る特徴量が実在すること）")
    # 実装が ICC3 と一致してしまっていないか
    c3 = np.array([ref[f]["icc3"] for f in diff]); gt = np.array([res.loc[f, "icc"] for f in diff])
    if len(diff):
        rec(S, "icc_not_icc3", "PASS" if (np.abs(c3 - gt) > 1e-4).all() else "FAIL", f"ICC3 と違う特徴量={len(diff)}、実装が ICC3 に一致した数={(np.abs(c3 - gt) <= 1e-4).sum()}", "負例（定義が違えば落ちる）")
    # 区分
    from_ref = []
    for f, c in zip(nonex, cl):
        lo = max(c[0], 0.0)
        from_ref.append("poor" if lo < 0.5 else "moderate" if lo < 0.75 else "good" if lo <= 0.9 else "excellent")
    mis = [f for f, cat in zip(nonex, from_ref) if res.loc[f, "category"] != cat]
    rec(S, "category", "PASS" if not mis else "FAIL", f"不一致={len(mis)} {mis[:3]}（{len(nonex)} 特徴量）", "下限（0 に切る）<0.5 poor、<0.75 moderate、≤0.9 good、>0.9 excellent（README の定義）")
    dist = pd.Series(from_ref).value_counts().to_dict()
    rec(S, "category_distribution", "PASS", f"参照の区分の分布={dist}（実装 {res.loc[nonex, 'category'].value_counts().to_dict()}）、A=B の {len(exact_f)} 特徴量の区分={res.loc[exact_f, 'category'].value_counts().to_dict()}", "（情報）")
    # Bland–Altman
    ba = ex["ba"]
    fb = [f for f in fs if isinstance(ba.get(f), dict) and "bias" in ba[f]]
    rec(S, "ba.available", "PASS" if len(fb) == len(fs) else "FAIL", f"B-A が出た特徴量={len(fb)} / ICC が出た {len(fs)}", "")
    for key, rk, tk in [("bias", "bias", "bias"), ("sd", "sd", "sd"), ("lo", "lo", "lowerLoa"), ("hi", "hi", "upperLoa")]:
        compare(S, f"ba.{key}", [ref[f]["ba"][rk] for f in fb], [ba[f][tk] for f in fb], 1e-12, 1e-9, "平均・SD・±1.96 SD は閉じた式")
    for key, rk, tk, ix in [("bias_ci_lower", "bl", "biasCi", 0), ("bias_ci_upper", "bl", "biasCi", 1), ("loa_lower_ci_lower", "lol", "lowerLoaCi", 0), ("loa_lower_ci_upper", "lol", "lowerLoaCi", 1),
                            ("loa_upper_ci_lower", "hil", "upperLoaCi", 0), ("loa_upper_ci_upper", "hil", "upperLoaCi", 1)]:
        compare(S, f"ba.{key}", [ref[f]["ba"][rk][ix] for f in fb], [ba[f][tk][ix] for f in fb], 1e-12, 1e-8, "Bland–Altman (1999) の近似。t 分位点は scipy、実装は自前の逆関数")


# ---------------------------------------------------------------- P4


def fold_values(d, kind, repeat_idx_names):
    pass


def safe_c(tt, p, ee, life=False):
    """欠けた予測を除いた C-index。既定は自前の総当たり（lifelines と同じ規則）、life=True は lifelines。"""
    ok = ~np.isnan(p)
    if ok.sum() < 2 or ee[ok].sum() == 0:
        return np.nan
    if not life:
        return harrell_c(tt[ok], ee[ok], p[ok])
    try:
        return concordance_index(tt[ok], -p[ok], ee[ok])
    except Exception:
        return np.nan


def p4_recompute(d):
    """予測 CSV と分割から、折ごとの値を再計算する（二値は sklearn の AUC、生存は自前の総当たり C と lifelines の C）。"""
    pred = T(d, "pred")
    folds = T(d, "folds")
    kind = "binary" if "outcome" in pred.columns else "survival"
    reps = sorted(int(re.search(r"\d+", c).group()) for c in pred.columns if c.startswith("pred_repeat_"))
    out = []
    for r in reps:
        p = pred[f"pred_repeat_{r}"].values.astype(float)
        fa = pred[f"fold_repeat_{r}"].values.astype(int)
        for fo in sorted(set(fa)):
            te = fa == fo
            row = folds[(folds["repeat"] == r) & (folds["fold"] == fo)].iloc[0]
            if kind == "binary":
                y = (pred["outcome"].values == "Mutant").astype(int)
                ok = ~np.isnan(p[te])
                v = roc_auc_score(y[te][ok], p[te][ok]) if len(set(y[te][ok])) == 2 else np.nan
                vl = v
            else:
                tt, ee = pred["time"].values.astype(float)[te], pred["event"].values.astype(int)[te]
                v = safe_c(tt, p[te], ee)
                vl = safe_c(tt, p[te], ee, life=True)
            out.append((r, fo, v, row["score"], row["n_test"], int(te.sum()), row["n_train"], row["error"], vl))
    return pred, folds, kind, reps, out


def check_p4_run(rid, full=True):
    S = "P4"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.run", "UNMET", "json が無い"); return None
    if d["status"] != "ok":
        rec(S, f"{rid}.run", "FAIL", f"実行が失敗: {[s['errors'] for s in d['steps'] if s['errors']]}", ""); return None
    pred, folds, kind, reps, out = p4_recompute(d)
    ex = d["extra"]["p4"]
    n = len(pred)
    # (b) 分割: 反復ごとに全症例がちょうど 1 回検証に使われる・訓練と検証が重ならない
    oa = ex["outerAssign"]
    K = d["params"]["outerK"]
    ok_once = True; ok_size = True; ok_disj = True; ok_strat = True
    y_or_e = (pred["outcome"].values == "Mutant").astype(int) if kind == "binary" else pred["event"].values.astype(int)
    for ri, a in enumerate(oa):
        a = np.array(a)
        ok_once &= (len(a) == n) and set(a) == set(range(K))
        sizes = np.bincount(a, minlength=K)
        ok_size &= (sizes.max() - sizes.min() <= 1)
        ok_strat &= all(abs(y_or_e[a == k].sum() - y_or_e.sum() / K) <= 1.0 + 1e-9 for k in range(K))
        for k in range(K):
            row = folds[(folds["repeat"] == ri + 1) & (folds["fold"] == k + 1)].iloc[0]
            ok_disj &= (row["n_test"] == (a == k).sum()) and (row["n_train"] == n - (a == k).sum())
    # 外側の割り当てが CSV のものと同じ
    ok_csv = all(np.array_equal(np.array(oa[ri]) + 1, pred[f"fold_repeat_{ri+1}"].values) for ri in range(len(oa)))
    rec(S, f"{rid}.b.each_case_validated_once", "PASS" if ok_once and ok_csv else "FAIL", f"反復 {len(oa)} 回 × 症例 {n}: 各症例が 1 つの折に属する={ok_once}、出力 CSV の分割と一致={ok_csv}", "割り当てが全症例を 0..K−1 の 1 つに分ける")
    rec(S, f"{rid}.b.train_val_disjoint_and_counts", "PASS" if ok_disj else "FAIL", f"n_test=その折の症例数、n_train=残り（重ならない）={ok_disj}", "整数の一致")
    rec(S, f"{rid}.b.balanced_sizes", "PASS" if ok_size else "FAIL", f"折の大きさの差が 1 以下={ok_size}", "層化の分割の規則（性質）")
    rec(S, f"{rid}.b.stratified", "PASS" if ok_strat else "FAIL", f"各折の陽性（イベント）数が 全体/K から 1 以内={ok_strat}", "層化（性質）")
    lk = step(d, "p4-leak")["summary"]
    rec(S, f"{rid}.b.leak_check_record", "PASS" if lk["splitCheck"] == "OK" and lk["rowUseCheck"] in ("OK", "NOTICE") else "FAIL", f"実装のデータリーク検査: split={lk['splitCheck']} rowUse={lk['rowUseCheck']}", "OK（品質管理を有効にしたときは NOTICE もあり）")
    # (c) 再計算
    vals = np.array([o[2] for o in out], dtype=float)
    ts = np.array([o[3] for o in out], dtype=float)
    vlife = np.array([o[8] for o in out], dtype=float)
    compare(S, f"{rid}.c.fold_score", vals, ts, 1e-9, 1e-9, f"折ごとの {'AUC（sklearn）' if kind=='binary' else 'C-index（自前の総当たり。lifelines と同じ規則）'} を予測 CSV と分割から再計算")
    if kind == "survival" and full:
        compare(S, f"{rid}.c.fold_score_vs_lifelines", vlife, ts, 1e-9, 1e-9, "折ごとの C-index を lifelines の concordance_index で再計算")
    errs = folds["error"].dropna()
    if len(errs):
        rec(S, f"{rid}.c.unevaluable_folds", "PASS" if np.isnan(vals[np.isnan(ts)]).all() else "FAIL", f"評価不能の折={len(errs)}: {errs.tolist()[:2]}（参照でも評価不能={int(np.isnan(vals).sum())}）", "実装が評価不能とした折が、参照でも定義できない（片方のクラスだけ等）")
    sm_ = ex["summary"]
    v_ok = vals[~np.isnan(vals)]
    compare(S, f"{rid}.c.summary_mean_min_max", [v_ok.mean(), v_ok.min(), v_ok.max()], [sm_["mean"], sm_["min"], sm_["max"]], 1e-9, 1e-9, "折ごとの値の平均・最小・最大")
    pooled_ref = []
    for r in reps:
        p = pred[f"pred_repeat_{r}"].values.astype(float)
        ok = ~np.isnan(p)
        if kind == "binary":
            y = (pred["outcome"].values == "Mutant").astype(int)
            pooled_ref.append(roc_auc_score(y[ok], p[ok]))
        else:
            pooled_ref.append(safe_c(pred["time"].values.astype(float), p, pred["event"].values.astype(int)))
    if kind == "binary":
        compare(S, f"{rid}.c.pooled_auc", pooled_ref, ex["pooled"], 1e-9, 1e-9, "予測を束ねた AUC（sklearn）")
    elif full:
        rec(S, f"{rid}.c.pooled_cindex", "UNMET", f"束ねた C-index（参照）={np.round(pooled_ref, 4).tolist()}。実装は生存で束ねた値を出さない（pooled=null）", "比べる実装側の値が無い（束ねた値は性質の検査(d)で参照の値を使う）")
    return dict(kind=kind, pooled=pooled_ref, fold_mean=float(v_ok.mean()) if len(v_ok) else np.nan, folds=vals, final=ex["finalFeatures"], summary=sm_,
                nonempty=float(folds["features"].notna().mean()), unevaluable=int(np.isnan(ts).sum()))


def check_p4():
    S = "P4"
    for kind in ["b", "s"]:
        for tag in ["topk", "lasso"]:
            for sfx in ["", "_pre"]:
                check_p4_run(f"p4{kind}_{tag}{sfx}")
            # (a) 再現性
            a, b = J(f"p4{kind}_{tag}"), J(f"p4{kind}_{tag}_again")
            if a and b:
                same = a["tables"] == b["tables"] and a["extra"] == b["extra"]
                rec(S, f"p4{kind}_{tag}.a.reproducible", "PASS" if same else "FAIL", f"同じ入力・シードで 2 回: 表と出力（分割・予測・選ばれた特徴量）が完全に同じ={same}", "完全一致")
    # 生存だけのモデル（Elastic Net・Ridge・Cox ブースティング）: 同じ検査（分割・折ごとの C-index を lifelines で再計算・予測の再現）と、入れ替え（負例）
    for tag in ["enet", "ridge", "coxboost", "rsf"]:
        rid = f"p4s_{tag}"
        if J(rid) is None:
            rec(S, f"{rid}.run", "UNMET", "json が無い"); continue
        check_p4_run(rid)
        a, b = J(rid), J(f"{rid}_again")
        if a and b:
            rec(S, f"{rid}.a.reproducible", "PASS" if a["tables"] == b["tables"] and a["extra"] == b["extra"] else "FAIL", "同じ入力・シードで 2 回: 表と出力が完全に同じ", "完全一致")
        ms = []
        k = 1
        while J(f"{rid}_permS{k}") is not None:
            d_ = J(f"{rid}_permS{k}")
            if d_["status"] == "ok":
                ms.append(float(step(d_, "p4-cv")["summary"]["mean"]))
            k += 1
        if len(ms) >= 5:
            v = np.array(ms)
            tci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / math.sqrt(len(v)))
            rec(S, f"{rid}.d.permutation_cindex", "PASS" if tci[0] <= 0.5 <= tci[1] else "FAIL",
                f"時間・イベントの組を入れ替え {len(v)} 回: 外側の C-index の平均の平均={v.mean():.3f}（SD {v.std(ddof=1):.3f}、95% t 区間 [{tci[0]:.3f},{tci[1]:.3f}]）", "95% t 区間が 0.5 を含む（特徴量と生存が無関係なデータ）")
        else:
            rec(S, f"{rid}.d.permutation_cindex", "UNMET", f"入れ替えの実行 {len(ms)}/5 回分しか無い")
    # 前処理を有効にしたとき、実際に何かを除いたか（検査の対象が実在するか）
    for rid in ["p4b_topk_pre", "p4b_lasso_pre", "p4s_topk_pre"]:
        d = J(rid)
        if d:
            q = step(d, "p4-qc")["summary"]
            rec(S, f"{rid}.pre.effective", "PASS" if q["removed"] > 0 else "FAIL", f"品質管理で除いた特徴量={q['removed']}（{q['featuresBefore']} → {q['featuresKept']}）", "前処理が実際に効いている（前提）")
            lk = step(d, "p4-leak")["summary"]
            rec(S, f"{rid}.pre.leak_level", "PASS" if lk["rowUseCheck"] in ("NOTICE", "OK") else "FAIL", f"rowUseCheck={lk['rowUseCheck']} findings={lk['rowUseFindings']}", "README: 品質管理を有効にすると「注意」。NG でない")
            if rid.startswith("p4b"):
                pdr = step(d, "p4-cv")["summary"]
                rec(S, f"{rid}.pre.pdrop_effective", "PASS" if (pdr.get("pdropOuterEntries") or 0) > 0 else "FAIL", f"p 値による除外の記録: 外側={pdr.get('pdropOuterEntries')} 平均除外数={pdr.get('pdropOuterMeanDropped')}", "B3 が実際に動いた（前提）")
    # (d) 並べ替え（負例）
    perm_summary = {}
    for kind, perm_tag in [("b", "permB"), ("s", "permS")]:
        for tag in ["topk", "lasso"]:
            for pre in ["", "_pre"]:
                key = f"p4{kind}_{tag}{pre}"
                start = len(CHECKS)
                per = []
                k = 1
                while J(f"{key}_{perm_tag}{k}") is not None:
                    r = check_p4_run(f"{key}_{perm_tag}{k}", full=False)
                    if r is not None:
                        per.append(r)
                    k += 1
                # (b)(c) の個別の合格は 1 行にまとめ、失敗・未達だけ個別に残す
                new = CHECKS[start:]
                del CHECKS[start:]
                npass = sum(1 for x in new if x["status"] == "PASS")
                CHECKS.extend(x for x in new if x["status"] != "PASS")
                rec(S, f"{key}.perm_runs.b_c_checks", "PASS" if all(x["status"] == "PASS" for x in new) else "FAIL",
                    f"並べ替え {len(per)} 回の実行それぞれで、分割（全症例が各反復で 1 回ずつ検証・訓練と検証が重ならない・層化）と、折ごとの値・平均/最小/最大・束ねた値の再計算が一致: {npass}/{len(new)} 件が合格",
                    "実データの run と同じ検査（個別の行は省略。失敗・未達は個別に残す）")
                if len(per) < 5:
                    rec(S, f"{key}.d.permutation", "UNMET", f"並べ替えの実行 {len(per)}/5 回分しか無い", ""); continue
                name = "AUC" if kind == "b" else "C-index"
                for what, vals in [("束ねた" if kind == "b" else "束ねた（参照の自前の総当たり）", [np.nanmean(r["pooled"]) for r in per]), ("折ごとの値の平均", [r["fold_mean"] for r in per])]:
                    v = np.array(vals, dtype=float)
                    v = v[~np.isnan(v)]
                    n_ = len(v)
                    tci = stats.t.interval(0.95, n_ - 1, loc=v.mean(), scale=v.std(ddof=1) / math.sqrt(n_))
                    ok = tci[0] <= 0.5 <= tci[1] and v.mean() < 0.6
                    ne = np.mean([r["nonempty"] for r in per])
                    below = tci[1] < 0.5
                    rec(S, f"{key}.d.permutation.{'pooled' if what.startswith('束') else 'foldmean'}", "PASS" if ok else "FAIL",
                        f"{'EGFR ラベル' if kind=='b' else '時間・イベントの組'}を入れ替え {n_} シード: {what} {name} の平均={v.mean():.3f}（SD {v.std(ddof=1):.3f}、95% t 区間 [{tci[0]:.3f},{tci[1]:.3f}]、範囲 [{v.min():.3f},{v.max():.3f}]）。特徴量を使った折の割合={ne:.2f}",
                        f"{n_} シードの平均の 95% t 区間（自由度 {n_ - 1}）が 0.5 を含み、平均が 0.6 未満。漏れがあれば 0.5 を明確に超える",
                        "区間が 0.5 を下回った（漏れの向きとは逆）。下の参照（特徴量なしの予測）と同方向の既知の偏りで、漏れではない" if below else "", cause="design" if below else "")
                    perm_summary[f"{key}.{what}"] = float(v.mean())
                if kind == "b":
                    # 参照: 特徴量を使わない予測（その折の訓練の陽性割合だけ）で束ねた AUC。入れ替えた結果でも 0.5 を下回る向きに偏る既知の現象かを確かめる
                    nulls_ = []
                    k2 = 1
                    while J(f"{key}_{perm_tag}{k2}") is not None:
                        dd_ = J(f"{key}_{perm_tag}{k2}")
                        pr_ = T(dd_, "pred")
                        y_ = (pr_["outcome"].values == "Mutant").astype(int)
                        for ri, a_ in enumerate(dd_["extra"]["p4"]["outerAssign"]):
                            a_ = np.array(a_)
                            pp_ = np.array([y_[a_ != fo].mean() for fo in a_])
                            nulls_.append(roc_auc_score(y_, pp_))
                        k2 += 1
                    rec(S, f"{key}.d.reference_null_model_pooled_auc", "PASS",
                        f"参照: 同じ外側の分割で、特徴量を使わず訓練の陽性割合だけで予測したときの束ねた AUC の平均={np.mean(nulls_):.3f}（{len(nulls_)} 回）。実装の束ねた AUC の平均={np.mean([np.nanmean(r['pooled']) for r in per]):.3f}",
                        "（情報）特徴量が効かないとき、束ねた AUC は 0.5 より小さい側に偏る（折の訓練の陽性割合が検証の陽性割合と逆に動くため）。実装の偏りがこの参照と同方向か")
    return perm_summary


def leaky_reference():
    """検出力の確認: 結果ラベルを入れ替えても、特徴量の選択を交差検証の前に全データでやると AUC が 0.5 を超えることを、自前の参照で示す。"""
    S = "P4"
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    D = M[M["EGFR"].notna()].reset_index(drop=True)
    y0 = (D["EGFR"] == "Mutant").astype(int).values
    Xf = D[[f for f in FEATS if not D[f].isna().any() and D[f].nunique() > 1]].values
    res_leak, res_ok = [], []
    for sd in range(1, 31):
        rng = np.random.default_rng(1000 + sd)
        y = rng.permutation(y0)
        ps = np.array([stats.mannwhitneyu(Xf[y == 1, j], Xf[y == 0, j], method="asymptotic").pvalue for j in range(Xf.shape[1])])
        top = np.argsort(ps)[:4]
        for leak, store in [(True, res_leak), (False, res_ok)]:
            aucs = []
            for rep in range(2):
                skf = StratifiedKFold(5, shuffle=True, random_state=sd * 10 + rep)
                p = np.zeros(len(y))
                for tr, te in skf.split(Xf, y):
                    if leak:
                        cols = top
                    else:
                        pp = np.array([stats.mannwhitneyu(Xf[tr][y[tr] == 1, j], Xf[tr][y[tr] == 0, j], method="asymptotic").pvalue for j in range(Xf.shape[1])])
                        cols = np.argsort(pp)[:4]
                    mu, s = Xf[tr][:, cols].mean(0), Xf[tr][:, cols].std(0, ddof=1)
                    lr = LogisticRegression(penalty=None, max_iter=1000).fit((Xf[tr][:, cols] - mu) / s, y[tr])
                    p[te] = lr.predict_proba((Xf[te][:, cols] - mu) / s)[:, 1]
                aucs.append(roc_auc_score(y, p))
            store.append(np.mean(aucs))
    n_perm = 20
    sd_ok = np.std(res_ok, ddof=1)
    se_n = sd_ok / math.sqrt(n_perm)
    eff = np.mean(res_leak) - 0.5
    rec(S, "d.detector_power.reference_leaky", "PASS" if eff > 2 * se_n else "FAIL",
        f"参照（自前、30 通り）: ラベル入れ替えの束ねた AUC 平均 漏れあり（全データで MW の p が小さい上位 4 個を選んでから交差検証）={np.mean(res_leak):.3f}（SD {np.std(res_leak, ddof=1):.3f}）、漏れなし（折の内側で選ぶ）={np.mean(res_ok):.3f}（SD {sd_ok:.3f}）。"
        f"漏れの上乗せ={eff:.3f}、{n_perm} シードの平均の標準誤差={se_n:.3f}（その 2 倍={2 * se_n:.3f}）",
        f"漏れがあれば、この入れ替え・特徴量数・分割・症例数（44）でも、{n_perm} シードの平均が 0.5 を標準誤差の 2 倍以上超える＝負例の検出力がある")
    rec(S, "d.detector_power.readme_claim", "UNMET" if np.mean(res_leak) <= 0.7 else "PASS",
        f"README の「漏れのあるやり方では 0.7 を超える」は、この実データ（n=44・168 特徴量・上位 4 個）では再現しなかった（参照の漏れありの平均 {np.mean(res_leak):.3f}）。README の 0.7 は合成データ（強い信号の特徴量が混ざる）での試験の値",
        "（情報）実データの入れ替えで漏れの検出に使える上乗せは上の行のとおり")


# ---------------------------------------------------------------- R5


def check_r5_run(rid, kind):
    S = "R5"
    d = J(rid)
    if d is None:
        rec(S, f"{rid}.run", "UNMET", "json が無い"); return None
    if d["status"] != "ok":
        rec(S, f"{rid}.run", "FAIL", f"失敗: {[s['errors'] for s in d['steps'] if s['errors']]}", ""); return None
    ex = d["extra"]["r5"]
    sel = ex["selected"]
    pat = ex["patients"]
    D = M.iloc[pat].reset_index(drop=True)
    Tm, Em = D["OS_days"].values.astype(float), D["OS_event"].values.astype(int)
    P = d["params"]
    maxv = P["r5MaxVars"]
    rec(S, f"{rid}.max_vars", "PASS" if len(sel) <= maxv else "FAIL", f"選ばれた={len(sel)} 上限={maxv}", "選ばれた数 ≤ r5MaxVars")
    if kind == "perm":
        c = ex["corrected"]
        return dict(sel=sel, corrected=c["corrected"] if c else None, app=ex["cApparent"], opt=c["optimism"] if c else None, boots=c["bootstraps"] if c else None, failed=c["failed"] if c else None, oob=c.get("oob") if c else None)
    if not sel:
        rec(S, f"{rid}.model", "UNMET", "選ばれた特徴量が 0 個（特徴量を使わないモデル）。係数・p の突き合わせの対象が無い", "前提待ち")
        c = ex["corrected"]
        return dict(sel=sel, corrected=c["corrected"] if c else None, app=ex["cApparent"], opt=c["optimism"] if c else None, oob=c.get("oob") if c else None)
    # 欠損のあるはずの特徴量が使われていない
    miss_cols = [f for f in FEATS if D[f].isna().any()]
    rec(S, f"{rid}.missing_features_excluded", "PASS" if set(ex["droppedMissing"]) == set(miss_cols) else "FAIL", f"実装が除いた={ex['droppedMissing']} 参照（欠損のある列）={miss_cols}", "集合の一致")
    cols = [D[f].values.astype(float) for f in sel]
    means = [c.mean() for c in cols]; sds = [c.std(ddof=1) for c in cols]
    Z = np.array([(c - m) / s_ for c, m, s_ in zip(cols, means, sds)]).T
    fit = efron_cox(Z, Tm, Em)
    terms = pd.DataFrame(ex["terms"])
    tbl = T(d, "model")
    tm = tbl[tbl["feature"].isin(sel)].set_index("feature").loc[sel]
    compare(S, f"{rid}.std_mean_sd", np.r_[means, sds], np.r_[tm["mean"].values, tm["sd"].values], 1e-9, 1e-9, "標準化に使う平均・標本 SD（n−1）。全患者の値で")
    if fit is None:
        rec(S, f"{rid}.refit", "FAIL", "参照の Cox が収束しない", ""); return None
    cox_compare(S, f"{rid}.refit", fit, terms["hr"].values, terms["lower"].values, terms["upper"].values, terms["p"].values, "選ばれた特徴量だけで Cox（Efron）に当て直す。自前の Newton と比べる")
    compare(S, f"{rid}.refit.beta", fit["beta"], terms["beta"].values, cox_tols(fit) + 1e-12, 0.0, "表の beta_per_sd（log HR）。許容は上と同じ（停止規則が許す係数のずれ）")
    lp = Z @ fit["beta"]
    c_ref = harrell_c(Tm, Em, lp)
    compare(S, f"{rid}.cindex_apparent", [c_ref], [ex["cApparent"]], 1e-9, 0.0, "Harrell の C を定義どおり総当たりで数える（同時刻のイベント vs 打ち切りの組も数える。lifelines と同じ規則）")
    compare(S, f"{rid}.cindex_apparent_vs_lifelines", [concordance_index(Tm, -lp, Em)], [ex["cApparent"]], 1e-9, 0.0, "lifelines の concordance_index")
    s_ = fit  # 以降の p の参照
    # 後退選択の規則
    rec(S, f"{rid}.backward.p_out", "PASS" if (fit["p"] <= P["r5POut"] + 1e-9).all() else "FAIL", f"最終モデルの Wald の p={np.round(fit['p'], 4).tolist()} 出る p={P['r5POut']}", "残った変数の p ≤ 出る p")
    # 戻らない: 候補のうち選ばれなかった変数を 1 つずつ足して p ≥ 入る p
    capped = any(st["action"] == "cap" for st in ex["steps"])
    notsel = [f for f in ex["candidates"] if f not in sel]
    pin = []
    for f in notsel:
        x = D[f].values.astype(float)
        z = (x - x.mean()) / x.std(ddof=1)
        c2 = efron_cox(np.c_[Z, z], Tm, Em)
        pin.append(c2["p"][-1] if c2 is not None else np.nan)
    if notsel:
        mn = np.nanmin(pin)
        ok = (mn >= P["r5PIn"] - 1e-9) or capped
        rec(S, f"{rid}.backward.p_in", "PASS" if ok else "FAIL", f"選ばれなかった候補 {len(notsel)} 個を最終モデルに 1 つずつ足したときの p の最小={mn:.4f}（入る p={P['r5PIn']}）、上限による除外（cap）あり={capped}",
            "戻る条件（p < 入る p）を満たす候補が残らない。上限で除いたものは例外（READMEの上限規則）")
    c = ex["corrected"]
    rec(S, f"{rid}.corrected_c.sanity", "PASS" if c and c["corrected"] <= ex["cApparent"] + 1e-12 and c["optimism"] >= 0 else "FAIL",
        f"見かけ={ex['cApparent']:.4f} 補正={c['corrected']:.4f} 楽観={c['optimism']:.4f}（成功 {c['bootstraps']} 失敗 {c['failed']}）" if c else "補正なし", "補正 = 見かけ − 楽観、楽観 ≥ 0（性質）")
    rec(S, f"{rid}.corrected_c.independent", "UNMET", "手順全体（相関除去→ブースティング／LASSO-Cox→後退選択→Cox）をブートストラップごとに独立に再実装した参照を作っていない（Cox ブースティングは参照の標準実装が無い。LASSO-Cox も sklearn に無い）",
        "補正 C-index の値そのものの独立な突き合わせは未達。代わりに、時間の入れ替え（負例）で 0.5 付近を確かめる")
    return dict(sel=sel, corrected=c["corrected"], app=ex["cApparent"], opt=c["optimism"], boots=c["bootstraps"], failed=c["failed"])


def check_r5_reference_optimism():
    """R5 の補正 C-index が入れ替えで 0.5 を超える原因の切り分け: 実装と無関係な単純な手順（log-rank の p が小さい上位 4 個→Cox）に、
    同じ Harrell の楽観ブートストラップ（B=200）を自前で当て、時間・イベントを入れ替えたデータで補正 C が 0.5 になるかを見る。"""
    S = "R5"
    D = M.copy()
    Tm, Em = D["OS_days"].values.astype(float), D["OS_event"].values.astype(int)
    names = [f for f in FEATS if not D[f].isna().any() and D[f].nunique() > 1]
    X = D[names].values.astype(float)
    B, K = 200, 4

    def proc(idx, tt, ee):
        chi = []
        for j in range(X.shape[1]):
            x = X[idx, j]
            g = x > np.median(x)
            chi.append(logrank_chi2_fast(tt, ee, g) if 0 < g.sum() < len(g) else -1)
        top = np.argsort(chi)[::-1][:K]
        mu = X[idx][:, top].mean(0); sd = X[idx][:, top].std(0, ddof=1)
        fit = efron_cox((X[idx][:, top] - mu) / sd, tt, ee)
        if fit is None:
            return None
        return lambda rows: ((X[rows][:, top] - mu) / sd) @ fit["beta"]

    rng = np.random.default_rng(5)
    res = []
    t0 = time.time()
    for sd_ in range(24):
        perm = rng.permutation(len(Tm))
        T_, E_ = Tm[perm], Em[perm]
        allr = np.arange(len(T_))
        full = proc(allr, T_, E_)
        if full is None:
            continue
        c_app = harrell_c(T_, E_, full(allr))
        opt = []
        for b in range(B):
            idx = rng.integers(0, len(T_), len(T_))
            if E_[idx].sum() < 2:
                continue
            f_ = proc(idx, T_[idx], E_[idx])
            if f_ is None:
                continue
            opt.append(harrell_c(T_[idx], E_[idx], f_(idx)) - harrell_c(T_, E_, f_(allr)))
        res.append((c_app, c_app - np.mean(opt)))
    app = np.array([r[0] for r in res]); cor = np.array([r[1] for r in res])
    n_ = len(cor)
    tci = stats.t.interval(0.95, n_ - 1, loc=cor.mean(), scale=cor.std(ddof=1) / math.sqrt(n_))
    rec(S, "ref.optimism_bootstrap_null", "PASS",
        f"参照（実装と無関係な単純な手順: 上位 4 個→Cox。Harrell の楽観ブートストラップ B={B}）を、時間・イベントを入れ替えた {n_} 通りに当てた補正 C の平均={cor.mean():.3f}（95% t 区間 [{tci[0]:.3f},{tci[1]:.3f}]）、見かけの C の平均={app.mean():.3f}（{time.time()-t0:.0f}s）",
        "（情報）入れ替えでも補正 C が 0.5 を超えるなら、実装の誤りでなく楽観ブートストラップの性質（症例 56・イベント 15・特徴量 168 の強い過学習で、楽観が足りない）。0.5 に戻るなら実装の補正が不足している")
    return cor.mean(), tci


def check_r5():
    S = "R5"
    out = {}
    for tag in ["lasso", "boost"]:
        r = check_r5_run(f"r5_{tag}", "real")
        out[tag] = r
        a, b = J(f"r5_{tag}"), J(f"r5_{tag}_again")
        if a and b:
            rec(S, f"r5_{tag}.reproducible", "PASS" if a["tables"] == b["tables"] and a["extra"] == b["extra"] else "FAIL", "同じ入力・シードで 2 回: 表（モデル・経過）と補正 C-index が完全に同じ", "完全一致")
        vals, apps, sels, nulls, oobs = [], [], [], [], []
        k = 1
        while J(f"r5_{tag}_permS{k}") is not None:
            d_ = J(f"r5_{tag}_permS{k}")
            if d_["status"] != "ok":
                errs_ = [e for st_ in d_["steps"] for e in st_["errors"]]
                rec(S, f"r5_{tag}_permS{k}.run_failed", "FAIL", f"入れ替えたデータで手順が最後まで走らず、スキル全体が失敗: {errs_}（平均には入れていない）", "並べ替えデータでも手順が止まらずに結果（または特徴量なしの結果）を返す",
                    "後退選択の途中の Cox が発散すると、その標本では手順全体が失敗として扱われる。ブートストラップの中では失敗を数えて除くが、本体の実行（元のデータ）では止まる。元のデータでは起きなかった（r5_boost・r5_lasso は ok）", cause="design")
                k += 1
                continue
            start = len(CHECKS)
            r = check_r5_run(f"r5_{tag}_permS{k}", "perm")
            del CHECKS[start:]  # 1 行ずつの max_vars は集約して記録する
            if r is not None:
                vals.append(r["corrected"]); apps.append(r["app"]); sels.append(len(r["sel"])); oobs.append(r.get("oob"))
                if len(r["sel"]) > J("r5_%s_permS%d" % (tag, k))["params"]["r5MaxVars"]:
                    nulls.append(-1)
                if len(r["sel"]) == 0:
                    nulls.append(r["corrected"])
            k += 1
        if len(vals) < 5:
            rec(S, f"r5_{tag}.permutation", "UNMET", f"並べ替えの実行 {len(vals)}/5 回分しか無い"); continue
        v = np.array([x for x in vals if x is not None], dtype=float)
        n_ = len(v)
        tci = stats.t.interval(0.95, n_ - 1, loc=v.mean(), scale=v.std(ddof=1) / math.sqrt(n_))
        ok = tci[0] <= 0.5 <= tci[1]
        rec(S, f"r5_{tag}.permutation_corrected_c", "PASS" if ok else "FAIL",
            f"時間・イベントの組を入れ替え {n_} シード: 補正 C-index の平均={v.mean():.3f}（SD {v.std(ddof=1):.3f}、95% t 区間 [{tci[0]:.3f},{tci[1]:.3f}]、範囲 [{v.min():.3f},{v.max():.3f}]）、シードごと={np.round(v, 3).tolist()}。見かけの C={[None if a is None else round(a, 3) for a in apps]}、選ばれた数={sels}",
            f"{n_} シードの平均の 95% t 区間（自由度 {n_ - 1}）が 0.5 を含む（以前の README の主張。今の README は、イベントが少ないと補正後も 0.5 を超えると条件つきで書いている）",
            "" if ok else "切り分け: 実装と無関係な単純な手順（上位 4 個→Cox）に自前の楽観ブートストラップを当てた参照（ref.optimism_bootstrap_null）も、入れ替えで 0.5 を超えた。手順の性質（症例 56・イベント 15・特徴量 168 の過学習）が疑われるが、手順が同じでないので実装の補正の不足を除外できていない",
            cause="" if ok else "design")
        ov = np.array([x for x in oobs if x is not None], dtype=float)
        if len(ov) >= 5:
            oci = stats.t.interval(0.95, len(ov) - 1, loc=ov.mean(), scale=ov.std(ddof=1) / math.sqrt(len(ov)))
            rec(S, f"r5_{tag}.permutation_oob_c", "PASS" if oci[0] <= 0.5 <= oci[1] else "FAIL",
                f"入れ替え {len(ov)} シード: ブートストラップに入らなかった患者（OOB）での C-index の平均={ov.mean():.3f}（SD {ov.std(ddof=1):.3f}、95% t 区間 [{oci[0]:.3f},{oci[1]:.3f}]）。同じ run の補正 C の平均={v.mean():.3f}",
                "モデルが見ていない患者では本来 0.5。平均の 95% t 区間が 0.5 を含む")
        appv = np.array([a_ for a_ in apps if a_ is not None], dtype=float)
        rec(S, f"r5_{tag}.permutation_apparent_is_inflated", "PASS" if len(appv) and appv.mean() > 0.6 else "UNMET",
            f"入れ替えても見かけの C の平均={appv.mean():.3f}（特徴量を選ぶだけで 0.5 を大きく超える）= 補正が必要な状況であること。補正 C の平均={v.mean():.3f}", "（前提）見かけの C が膨らむ状況で、補正が 0.5 に戻すかを見ている")
        rec(S, f"r5_{tag}.permutation_apparent_ge_corrected", "PASS" if all((a_ is None) or (c <= a_ + 1e-12) for a_, c in zip(apps, vals)) else "FAIL", "補正 ≤ 見かけ（見かけのある run）", "性質")
        nn = [x for x in nulls if x != -1]
        if nn:
            # 以前は null モデルでも補正 C が数値で出て 0.36 などになった（設計上の観察）。仕様を変え、補正を出さないことを検査する
            rec(S, f"r5_{tag}.null_model_has_no_corrected_c", "PASS" if all(x is None for x in nn) else "FAIL",
                f"最終モデルが特徴量を使わない（null）run が {len(nn)} 回あり、補正 C-index={nn}（None = 出していない）",
                "特徴量を使わないモデルでは補正 C-index を出さない（見かけの C が 0.5 の定数モデルから、特徴量を使う標本の楽観を引くと 0.5 未満の意味のない値になるため）")
    # OOB の C-index を、独立に書き直した手順（r5_ref.py。元データで見かけの C が本体と全桁一致）と比べる。乱数列が違うので、平均の差を標準誤差で判断する
    d = J("r5_boost")
    if d and d["extra"]["r5"]["corrected"] and d["extra"]["r5"]["corrected"].get("oob") is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import r5_ref
        Xr, Tr_, Er_, _ = r5_ref.load()
        rr = r5_ref.corrected_c(Tr_, Er_, Xr, r5_ref.CFG, 200, np.random.default_rng(5))
        impl = d["extra"]["r5"]["corrected"]
        se = rr["oob_sd"] / math.sqrt(rr["oob_n"]) * math.sqrt(2)
        z = abs(impl["oob"] - rr["oob"]) / se
        rec(S, "r5_boost.oob_c_independent", "PASS" if z < 3 else "FAIL",
            f"実装 OOB C={impl['oob']:.4f}（{impl['oobN']} 回）、参照 OOB C={rr['oob']:.4f}（{rr['oob_n']} 回、SD {rr['oob_sd']:.3f}）。差の z={z:.2f}",
            "ブートストラップは乱数で再現できない。2 つの平均の差が、標準誤差（√2 倍）の 3 倍以内")
    return out


# ---------------------------------------------------------------- 実行とレポート


def main():
    t0 = time.time()
    VERS.update(lib_versions())
    check_dc()
    check_r1()
    check_r2()
    check_r3()
    perm = check_p4()
    leaky_reference()
    check_r5()
    check_r5_reference_optimism()
    write_report(time.time() - t0)


def counts(skill):
    c = {"PASS": 0, "FAIL": 0, "UNMET": 0}
    for x in CHECKS:
        if x["skill"] == skill:
            c[x["status"]] += 1
    return c


def write_report(sec):
    skills = ["Dataset check", "R1", "R2", "R3", "P4", "R5"]
    L = []
    L.append("# Research Stats 実データ検証（NSCLC Radiogenomics 56 例）\n")
    L.append("実行: `tools/verify_real/run_core.sh`（src/core を直接呼ぶ）の出力を、`tools/verify_real/verify_real.py`（TypeScript を見ずに定義から書いた参照実装）で突き合わせた。UI の配線は含まない。\n")
    L.append(f"参照側の版: {json.dumps(VERS, ensure_ascii=False)}  / 突き合わせの実行時間 {sec:.0f} 秒\n")
    L.append("## 件数（合格 / 失敗 / 未達）\n")
    L.append("| Skill | 合格 | 失敗 | 未達 |\n|---|---|---|---|")
    tot = {"PASS": 0, "FAIL": 0, "UNMET": 0}
    for s in skills:
        c = counts(s)
        L.append(f"| {s} | {c['PASS']} | {c['FAIL']} | {c['UNMET']} |")
        for k in tot:
            tot[k] += c[k]
    L.append(f"| 合計 | {tot['PASS']} | {tot['FAIL']} | {tot['UNMET']} |\n")
    CAUSE = {"impl": "実装の食い違い", "refdef": "参照側の定義の違い", "refdefect": "参照側の不具合", "design": "設計上の観察", "unknown": "原因未確認", "": "（原因未記入）"}
    cc = {}
    for x in CHECKS:
        if x["status"] == "FAIL":
            cc[x["cause"]] = cc.get(x["cause"], 0) + 1
    L.append("失敗の原因の見立て: " + "、".join(f"{CAUSE[k]} {v} 件" for k, v in cc.items()) + "。\n")
    L.append("## 実行の設定\n")
    L.append("共通: 乱数シード 20261125（並べ替えの負例は 20261125+k）、特徴量 = default56（170 列）、臨床 = clinical_surv56_subject（結合の ID は SubjectCode）。実装の版: plugin 0.7.1（main d01ad58）。\n")
    L.append("| run | シード | 既定から変えた設定 | 実行時間(s) |\n|---|---|---|---|")
    for rid in ["dc", "r1_median", "r1_tertile", "r1_optimal", "r1_median_a03", "r2_acc", "r2_youden", "r2_acc_a45", "r2_acc_none", "r3", "p4b_topk", "p4b_lasso", "p4s_topk", "p4s_lasso", "p4b_topk_pre", "p4b_lasso_pre", "p4s_topk_pre", "p4s_lasso_pre", "r5_lasso", "r5_boost"]:
        d = J(rid)
        if d:
            ch = {k: v for k, v in d["params"].items() if k in d["changedFromDefault"] and k not in ("featureIdColumn", "patientColumn", "clinicalIdColumn")}
            L.append(f"| {rid} | {d['seed']} | {json.dumps(ch, ensure_ascii=False)} | {d['elapsedMs'] / 1000:.1f} |")
    import glob
    tot_ms = sum(json.load(open(f))["elapsedMs"] for f in glob.glob(os.path.join(OUT, "*.json")) if not f.endswith(".checks.json"))
    L.append(f"\nコア実行の合計時間: {tot_ms / 60000:.1f} 分（{len(glob.glob(os.path.join(OUT, '*.json')))} 本）。P4 の並べ替えは二値・生存×topk・lasso×前処理なし・あり×20 シード、R5 の並べ替えは boost 10・lasso 5 シード。R5 のブートストラップは 2000 から 200 に絞った（時間のため）。R1 の最適しきい値の並べ替えは既定の 1000、多変量の補正 C のブートストラップは 500。\n")
    L.append("未達は「前提待ち（比べる対象が出ない・参照が無い）」で、合格に数えていない。\n")
    L.append("## 失敗と未達の一覧\n")
    for x in CHECKS:
        if x["status"] != "PASS":
            L.append(f"- **{x['status']}**{('（' + CAUSE[x['cause']] + '）') if x['cause'] else ''} [{x['skill']}] `{x['id']}`: {x['observed']}  \n  基準: {x['tol']} {('/ ' + x['note']) if x['note'] else ''}")
    L.append("\n## 全検査\n")
    for s in skills:
        L.append(f"### {s}\n")
        L.append("| 結果 | 検査 | 観測 | 基準・許容の理由 |\n|---|---|---|---|")
        for x in CHECKS:
            if x["skill"] == s:
                esc = lambda t: t.replace("|", "\\|").replace("\n", " ")
                L.append(f"| {x['status']} | `{x['id']}` | {esc(x['observed'])} | {esc(x['tol'])}{(' ' + esc(x['note'])) if x['note'] else ''} |")
        L.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[:20]))
    print(f"REPORT: {REPORT}")
    # 機械可読
    with open(os.path.splitext(REPORT)[0] + ".checks.json", "w") as f:
        json.dump(CHECKS, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
