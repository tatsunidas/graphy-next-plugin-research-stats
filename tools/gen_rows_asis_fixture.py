"""行のまま解析（P4）の性能を、TypeScript 側とは別に再計算して出す。
入力は test/fixtures/rows_asis_input.json（行ごとの予測・患者・外側の分割。test/rowsAsIs.test.ts を ROWS_ASIS_DUMP=1 で走らせると書き出す）。
行ごとの予測を患者ごとに平均して 1 患者 = 1 点にし、scikit-learn の roc_auc_score（二値）と、Harrell の C（生存。定義どおりに数え直す）で、
外側の分割ごと・全体の値を出す。
使い方: ROWS_ASIS_DUMP=1 npx vitest run test/rowsAsIs.test.ts && /home/tatsunidas/anaconda3/bin/python3 tools/gen_rows_asis_fixture.py  → test/fixtures/rows_asis.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

d = json.load(open("test/fixtures/rows_asis_input.json"))


def patient_means(sec, rows, value="pred"):
    df = pd.DataFrame({"g": [sec["groups"][i] for i in rows], "v": [sec[value][i] for i in rows]})
    for k in ("y", "time", "event"):
        if sec[k] is not None:
            df[k] = [sec[k][i] for i in rows]
    return df.groupby("g", sort=False).agg({**{"v": "mean"}, **{k: "first" for k in ("y", "time", "event") if sec[k] is not None}})


def harrell(time, event, risk):
    conc = comp = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] <= time[i]:
                continue
            comp += 1
            if risk[i] > risk[j]:
                conc += 1
            elif risk[i] == risk[j]:
                conc += 0.5
    return conc / comp


out = {}
b = d["binary"]
rows_all = list(range(len(b["groups"])))
folds = []
for f in range(b["outerK"]):
    rows = [i for i in rows_all if b["assign"][i] == f]
    m = patient_means(b, rows)
    folds.append(float(roc_auc_score(m["y"], m["v"])))
mp = patient_means(b, rows_all)
out["binary"] = {"folds": folds, "pooled": float(roc_auc_score(mp["y"], mp["v"])), "rows": len(rows_all), "patients": int(len(mp))}

s = d["survival"]
folds = []
for f in range(s["outerK"]):
    rows = [i for i in range(len(s["groups"])) if s["assign"][i] == f]
    m = patient_means(s, rows)
    folds.append(harrell(m["time"].tolist(), m["event"].tolist(), m["v"].tolist()))
out["survival"] = {"folds": folds, "mean": float(np.mean(folds)), "rows": len(s["groups"])}
json.dump(out, open("test/fixtures/rows_asis.json", "w"), indent=1)
print(out)
