"""再現性（R3）の参照値を、別実装（pingouin の intraclass_corr、numpy・scipy）で出す。
TypeScript 側とは独立に書いてあり、test/fixtures/reliability.json を作る。
pingouin はシステムの環境に入れず、使い捨ての venv に入れて実行する（README の開発の節を参照）:
  python3 -m venv /tmp/refvenv && /tmp/refvenv/bin/pip install pingouin==0.5.5 && /tmp/refvenv/bin/python tools/gen_reliability_fixture.py
"""
import json
import numpy as np
import pandas as pd
import pingouin as pg
import scipy
import scipy.stats as st

# pingouin は出力の CI95% を小数 2 桁に丸める。丸め前の値を使うため、後処理だけを外す（計算は pingouin のまま）
import pingouin.reliability as _rel
_rel._postprocess_dataframe = lambda df: df
out = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pingouin": pg.__version__}}

def icc_ref(M):
    n, k = M.shape
    df = pd.DataFrame({"s": np.repeat(np.arange(n), k), "c": np.tile(np.arange(k), n), "v": M.reshape(-1)})
    t = pg.intraclass_corr(data=df, targets="s", raters="c", ratings="v").set_index("Type")
    g = lambda ty: {"icc": float(t.loc[ty, "ICC"]), "lower": float(t.loc[ty, "CI95%"][0]), "upper": float(t.loc[ty, "CI95%"][1])}
    return {"single": g("ICC2"), "average": g("ICC2k"), "single_random_consistency": g("ICC3")}

cases = []
r = np.random.default_rng(11)
for n, k, noise, offset in [(20, 2, 0.5, 0.0), (15, 3, 1.0, 0.3), (30, 4, 0.3, 0.0), (25, 2, 0.2, 2.0), (12, 3, 2.0, 0.0)]:
    subj = r.normal(0, 1, size=(n, 1))
    M = subj + r.normal(0, noise, size=(n, k)) + offset * np.arange(k)[None, :]
    cases.append({"M": M.tolist(), "ref": icc_ref(M)})
out["icc"] = cases

ba = []
for n in (12, 30, 80):
    a = r.normal(10, 3, n); b = a * 1.02 + r.normal(0.4, 1.1, n)
    d = b - a; bias = d.mean(); sd = d.std(ddof=1); t = st.t.ppf(0.975, n - 1)
    se_bias = sd / np.sqrt(n); se_loa = np.sqrt(sd**2 * (1 / n + 1.96**2 / (2 * (n - 1))))
    ba.append({"a": a.tolist(), "b": b.tolist(), "bias": float(bias), "sd": float(sd),
               "bias_ci": [float(bias - t * se_bias), float(bias + t * se_bias)],
               "lo": float(bias - 1.96 * sd), "hi": float(bias + 1.96 * sd),
               "lo_ci": [float(bias - 1.96 * sd - t * se_loa), float(bias - 1.96 * sd + t * se_loa)],
               "hi_ci": [float(bias + 1.96 * sd - t * se_loa), float(bias + 1.96 * sd + t * se_loa)]})
out["ba"] = ba
out["dist"] = {
    "f": [{"p": p, "d1": d1, "d2": d2, "q": float(st.f.ppf(p, d1, d2))} for p, d1, d2 in [(0.975, 5, 7), (0.975, 19, 19.5), (0.025, 14, 28.3), (0.9, 2, 40), (0.975, 1, 2)]],
    "t": [{"p": p, "df": df, "q": float(st.t.ppf(p, df))} for p, df in [(0.975, 2), (0.975, 11), (0.975, 79), (0.9, 5), (0.025, 8), (0.5, 4)]],
}
json.dump(out, open("test/fixtures/reliability.json", "w"), allow_nan=False)
print("wrote", len(cases), "icc cases;", out["versions"])
