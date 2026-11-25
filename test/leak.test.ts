import { describe, expect, it } from "vitest";
import { parseCsv } from "../src/core/csv";
import { applyRowFilter, buildDataset, defaultFeatureColumns, type Dataset, type DatasetSpec } from "../src/core/dataset";
import { sourceOf } from "../src/core/delta";
import { runSkill } from "../src/core/engine";
import { suggestJoin } from "../src/core/joinQc";
import { checkPatientSplit, checkRowUse, idLikeColumns, rankIdentical, separatesCompletely, targetLeaks } from "../src/core/leak";
import { methodText } from "../src/core/methodText";
import { RowTrace } from "../src/core/pipeline/rowTrace";
import { finalModel, nestedCv, type NestedCvSpec } from "../src/core/pipeline/nestedCv";
import { qcFeatures } from "../src/core/pipeline/qc";
import { Rng } from "../src/core/rng";
import { suggestColumns } from "../src/core/suggest";
import { DATASET_CHECK } from "../src/core/skills";

const now = () => "2026-01-01T00:00:00Z";

// ---------------------------------------------------------------------------------------------
// 1. ID・由来・記録の列
// ---------------------------------------------------------------------------------------------
describe("idLikeColumns", () => {
  const N = 12;
  const rows = Array.from({ length: N }, (_, i) => [`C${i}`, i + 1, 100 + ((i * 7) % 5), (i * 0.37 + 1).toFixed(3), 20240100 + i, 5, `1.2.${i}`, 3 + (i % 4)].join(","));
  const csv = ["CaseLabel,RowNo,Count,f1,AcqDate,Provenance_N,SeriesUID,g2", ...rows].join("\n");
  const t = parseCsv(csv);
  const used = defaultFeatureColumns(t, ["CaseLabel", null]);

  it("flags a serial column, a date column and a UID/provenance column by name or value; ignores genuine features", () => {
    const r = idLikeColumns([t], ["CaseLabel", null], used);
    const byName = Object.fromEntries(r.map((x) => [x.column, x]));
    expect(byName.RowNo.reasons).toContain("name"); // Row で始まる
    expect(byName.RowNo.reasons).toContain("sequential");
    expect(byName.AcqDate.reasons).toContain("dateTime");
    expect(byName.AcqDate.reasons).toContain("sequential"); // 20240100+i は公差 1
    expect(byName.SeriesUID.reasons).toEqual(["uid"]);
    expect(byName.Provenance_N.reasons).toEqual(["name"]);
    for (const ok of ["f1", "g2", "Count"]) expect(byName[ok]).toBeUndefined(); // 小数・重複のある整数
  });
  it("separates warnings (columns actually used) from the already-excluded ones", () => {
    const r = idLikeColumns([t], ["CaseLabel", null], used);
    const byName = Object.fromEntries(r.map((x) => [x.column, x]));
    expect(used).not.toContain("Provenance_N"); // 既定の選択が既に除く
    expect(byName.Provenance_N.used).toBe(false);
    expect(used).not.toContain("SeriesUID"); // 文字列の列
    expect(byName.SeriesUID.used).toBe(false);
    expect(byName.RowNo.used).toBe(true);
    expect(byName.AcqDate.used).toBe(true);
  });
  it("negative: a table of genuine features only gives nothing", () => {
    const g = parseCsv(["CaseLabel,f1,f2,f3", ...Array.from({ length: 10 }, (_, i) => `C${i},${i * 0.31 + 0.2},${(i % 3) + 1},${Math.sin(i)}`)].join("\n"));
    expect(idLikeColumns([g], ["CaseLabel", null], ["f1", "f2", "f3"])).toEqual([]);
  });
  it("a column with a missing value or a fraction is not a sequence", () => {
    const g = parseCsv("a,b,c\n1,1.5,1\n2,2,\n3,3,3\n");
    expect(idLikeColumns([g], [null], ["a", "b", "c"]).map((x) => x.column)).toEqual(["a"]);
  });
  const seq = (name: string, v: (number | string)[]) => idLikeColumns([parseCsv(`${name}\n${v.join("\n")}\n`)], [null], [name]).some((x) => x.reasons.includes("sequential"));
  it("sequences in file order with a constant step are detected: 1..n, step 10, descending", () => {
    expect(seq("a", [1, 2, 3, 4, 5, 6])).toBe(true);
    expect(seq("a", [10, 20, 30, 40])).toBe(true);
    expect(seq("a", [9, 7, 5, 3])).toBe(true);
    expect(seq("a", [5, 4, 3])).toBe(true);
    expect(seq("a", [100, 200, 300])).toBe(true);
  });
  it("negative: distinct integers in random order, sorted with an uneven step, a fractional look-alike, fewer than 3 rows, a constant column", () => {
    // IntensityBasedStatistical_Range のように、整数で全部異なるが連番でない列（CT の値域）
    const range = [1240, 987, 1530, 1102, 1763, 845, 1391, 1218, 1650, 902, 1477, 1066];
    expect(new Set(range).size).toBe(range.length);
    expect(seq("IntensityBasedStatistical_Range", range)).toBe(false);
    expect(seq("IntensityBasedStatistical_Range", [...range].sort((x, y) => x - y))).toBe(false); // 整列しても公差が不揃い
    expect(seq("a", [3, 1, 2, 5, 4])).toBe(false); // 値の集合は連続だが順序がばらばら
    expect(seq("a", [1, 2, 4, 8, 16])).toBe(false);
    expect(seq("a", [1, 2, 3.5, 4])).toBe(false);
    expect(seq("a", [1, 1.5, 2, 2.5])).toBe(false); // 小数の連番に見える列
    expect(seq("a", [1, 2])).toBe(false);
    expect(seq("a", [4, 4, 4])).toBe(false);
    expect(seq("a", [1, 2, 3, "", 5])).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
// 2. 目的変数の混入
// ---------------------------------------------------------------------------------------------
function dataset(features: Record<string, (number | null)[]>, clinical: Dataset["clinical"]): Dataset {
  const names = Object.keys(features);
  const n = features[names[0]].length;
  return { ids: Array.from({ length: n }, (_, i) => `P${i}`), featureNames: names, features: Array.from({ length: n }, (_, i) => names.map((k) => features[k][i])), lesionCounts: Array(n).fill(1), clinical };
}
const N = 24;
const label = Array.from({ length: N }, (_, i) => (i % 2 ? "resp" : "nonresp"));
const lab01 = label.map((l) => (l === "resp" ? 1 : 0));
const time = Array.from({ length: N }, (_, i) => 3 + ((i * 7) % 29) + i / 100); // 同じ値が無い
const event = Array.from({ length: N }, (_, i) => (i % 3 === 0 ? 0 : 1));
// 結果と弱く関連するだけ: 群の分布は重なる
const noisy = lab01.map((y, i) => y * 0.8 + ((i * 37) % 11) / 10);

describe("targetLeaks", () => {
  const clin = (): Dataset["clinical"] => [
    { name: "OS_months", role: "time", values: time },
    { name: "Event", role: "event", values: event },
    { name: "Outcome01", role: "outcome", values: lab01 },
  ];
  it("positive: same name, identical values, complete separation, identical rank order (and reversed)", () => {
    const d = dataset({
      Outcome01: lab01.map((v) => v + 0.0), // 同名・同値
      sep: lab01.map((y, i) => (y === 1 ? 10 + i / 100 : 1 + i / 100)), // 結果を完全に分ける
      timeSq: time.map((v) => v ** 2), // 時間と順位が一致
      timeNeg: time.map((v) => -v), // 逆順（|ρ| = 1）
      evSep: event.map((e, i) => (e === 1 ? 5 + i / 100 : i / 100)),
      weak: noisy,
    }, clin());
    const r = targetLeaks(d);
    const get = (f: string, role: string) => r.find((x) => x.feature === f && x.role === role)?.kinds;
    expect(get("Outcome01", "outcome")).toEqual(["sameName", "identical", "separates"]);
    expect(get("sep", "outcome")).toEqual(["separates"]);
    expect(get("timeSq", "time")).toEqual(["rankIdentical"]);
    expect(get("timeNeg", "time")).toEqual(["rankIdentical"]);
    expect(get("evSep", "event")).toEqual(["separates"]);
    expect(r.filter((x) => x.feature === "weak")).toEqual([]);
  });
  it("detects a numeric-coded outcome label and the exact copy of time/event", () => {
    const d = dataset({ copyT: [...time], copyE: event.map((e) => e) }, clin());
    const r = targetLeaks(d);
    expect(r.find((x) => x.feature === "copyT" && x.role === "time")?.kinds).toEqual(["identical", "rankIdentical"]);
    expect(r.find((x) => x.feature === "copyE" && x.role === "event")?.kinds).toContain("identical");
  });
  it("negative: features that are only weakly related are not flagged (AUC high but not exactly 1, rho high but not exactly 1)", () => {
    const rng = new Rng(3);
    const noisyTime = time.map((v) => v + (rng.next() - 0.5) * 6); // 強く相関するが順位は完全には一致しない
    const nearSep = lab01.map((y, i) => (y === 1 ? 5 + i / 100 : 1 + i / 100) + (i === 4 ? 6 : 0)); // 1 人だけ重なる
    const tied = lab01.map((y, i) => (y === 1 ? 3 + (i % 4 === 1 ? 2 : 0) : 3)); // 最小の陽性 = 最大の陰性（同値があるので完全には分けない）
    const d = dataset({ noisyTime, nearSep, tied, noisy }, clin());
    expect(Math.abs(spearmanOf(noisyTime, time))).toBeGreaterThan(0.9);
    const r = targetLeaks(d);
    expect(r).toEqual([]);
  });
  it("one observation per class is not evidence (the separation would be a tautology)", () => {
    expect(separatesCompletely([1, 5], ["a", "b"])).toBe(false);
    expect(separatesCompletely([1, 2, 5, 6], ["a", "a", "b", "b"])).toBe(true);
    expect(separatesCompletely([1, 2, 2, 6], ["a", "a", "b", "b"])).toBe(false); // 2 と 2 が同値
    expect(separatesCompletely([1, 2, 5, 6, 7], ["a", "a", "b", "b", "c"])).toBe(false); // 3 値の結果は対象外
  });
  it("rank identity ignores rows with a missing value and needs 3 rows with variation", () => {
    expect(rankIdentical([{ x: 1, y: 10 }, { x: 2, y: 20 }])).toBe(false);
    expect(rankIdentical([{ x: 1, y: 10 }, { x: 2, y: 20 }, { x: 3, y: 30 }])).toBe(true);
    expect(rankIdentical([{ x: 1, y: 3 }, { x: 1, y: 2 }, { x: 1, y: 1 }])).toBe(false);
  });
  it("covariates are not targets", () => {
    const d = dataset({ age: time }, [{ name: "age", role: "covariate", values: time }]);
    expect(targetLeaks(d)).toEqual([]);
  });
});

function spearmanOf(a: number[], b: number[]) {
  const rk = (x: number[]) => x.map((v) => x.filter((w) => w < v).length + 1);
  const ra = rk(a), rb = rk(b), n = a.length, ma = (n + 1) / 2;
  let s = 0, sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { s += (ra[i] - ma) * (rb[i] - ma); sa += (ra[i] - ma) ** 2; sb += (rb[i] - ma) ** 2; }
  return s / Math.sqrt(sa * sb);
}

// ---------------------------------------------------------------------------------------------
// 3. 同じ患者が訓練と検証に分かれていないか
// ---------------------------------------------------------------------------------------------
function noise(seed: number, n: number, m: number) {
  const rng = new Rng(seed);
  const g = () => { const u = 1 - rng.next(), v = rng.next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  return { X: Array.from({ length: n }, () => Array.from({ length: m }, g)), y: Array.from({ length: n }, (_, i) => (i % 2)) };
}
const cvSpec = (y: number[], over: Partial<NestedCvSpec> = {}): NestedCvSpec => ({ outerK: 4, innerK: 3, repeats: 2, kCandidates: [1, 2], strata: y, impute: false, ...over });

describe("checkPatientSplit", () => {
  it("OK when each patient appears once (assignment from a real nested CV)", async () => {
    const { X, y } = noise(1, 40, 6);
    const dev = X.map((_, i) => i);
    const cv = await nestedCv(X, { kind: "binary", y }, dev, cvSpec(y), new Rng(1));
    const r = checkPatientSplit({ outerK: 4, innerK: 3, outerAssign: cv.outerAssign, innerAssign: cv.innerAssign }, dev.map((i) => `P${i}`));
    expect(r.ok).toBe(true);
    expect(r.checked).toBe(2 * 4 + 2 * 4 * 3); // 外側 4 分割 × 2 反復 + 内側 3 分割 × 4 × 2
    expect(r.violations).toEqual([]);
  });
  it("NG when the same patient has rows on both sides: the offending IDs are returned", async () => {
    const { X, y } = noise(1, 40, 6);
    const dev = X.map((_, i) => i);
    const cv = await nestedCv(X, { kind: "binary", y }, dev, cvSpec(y), new Rng(1));
    // 2 行ずつ同じ患者（行 i と i+20）。割り当ては行ごとに独立なので、分かれる患者が出る
    const patient = dev.map((i) => `P${i % 20}`);
    const r = checkPatientSplit({ outerK: 4, innerK: 3, outerAssign: cv.outerAssign, innerAssign: cv.innerAssign }, patient);
    const split = cv.outerAssign[0].filter((a, i) => i < 20 && a !== cv.outerAssign[0][i + 20]).length;
    expect(split).toBeGreaterThan(0);
    expect(r.ok).toBe(false);
    expect(r.patients.length).toBeGreaterThan(0);
    expect(r.violations.some((v) => v.level === "outer")).toBe(true);
    // 実際に分かれている患者だけを挙げる（独立に数え直す）
    const expected = new Set<string>();
    cv.outerAssign.forEach((assign) => { for (let i = 0; i < 20; i++) if (assign[i] !== assign[i + 20]) expected.add(`P${i}`); });
    for (const p of expected) expect(r.patients).toContain(p);
  });
  it("hand-made assignments: OK when whole patients stay together, NG when one is split; test-set overlap is NG", () => {
    const patient = ["a", "a", "b", "b", "c", "c", "d", "d"];
    const mk = (assign: number[]) => ({ outerK: 2, innerK: 2, outerAssign: [assign], innerAssign: [] as number[][][] });
    expect(checkPatientSplit(mk([0, 0, 1, 1, 0, 0, 1, 1]), patient).ok).toBe(true);
    const bad = checkPatientSplit(mk([0, 1, 1, 1, 0, 0, 1, 1]), patient);
    expect(bad.ok).toBe(false);
    expect(bad.patients).toEqual(["a"]);
    expect(checkPatientSplit(mk([0, 0, 1, 1, 0, 0, 1, 1]), patient, ["e", "f"]).ok).toBe(true);
    const t = checkPatientSplit(mk([0, 0, 1, 1, 0, 0, 1, 1]), patient, ["e", "b"]);
    expect(t.ok).toBe(false);
    expect(t.violations).toEqual([{ level: "test", repeat: null, fold: null, patient: "b" }]);
  });
  it("inner folds are checked against the patients of the outer training rows", () => {
    const patient = ["a", "a", "b", "b", "c", "c"];
    const outer = [0, 0, 1, 1, 1, 1]; // 外側 0 の検証 = a,a、訓練 = b,b,c,c
    const cv = { outerK: 2, innerK: 2, outerAssign: [outer], innerAssign: [[[0, 0, 1, 1], [0, 1]]] };
    // 外側 0 の訓練（b,b,c,c）の内側は b と c を分けて OK。外側 1 の訓練（a,a）の内側で a を分けると NG
    const r = checkPatientSplit(cv, patient);
    expect(r.ok).toBe(false);
    expect(r.violations.every((v) => v.level === "inner" && v.repeat === 0 && v.fold === 1 && v.patient === "a")).toBe(true);
    expect(r.violations).toHaveLength(2); // 内側の 2 つの分割のどちらでも、a は検証と訓練の両方にいる
    expect(r.patients).toEqual(["a"]);
  });
});

// ---------------------------------------------------------------------------------------------
// 4. 前処理・特徴量選択が訓練側の行だけか
// ---------------------------------------------------------------------------------------------
describe("checkRowUse", () => {
  const dev = Array.from({ length: 48 }, (_, i) => i);
  const data = noise(2, 48, 8);
  const task = { kind: "binary" as const, y: data.y };

  it("honest nested CV (top-k): every estimate used training rows only, and the trace is not empty", async () => {
    const trace = new RowTrace();
    const spec = cvSpec(data.y, { trace });
    await nestedCv(data.X, task, dev, spec, new Rng(2));
    finalModel(data.X, task, dev, spec, new Rng(2));
    const r = checkRowUse(trace);
    expect(r.level).toBe("ok");
    const stages = r.findings.map((f) => `${f.where}/${f.stage}`).sort();
    expect(stages).toEqual(["final", "finalInner", "fold", "inner"].flatMap((w) => ["fit", "impute", "rank"].map((st) => `${w}/${st}`)).sort());
    expect(r.findings.every((f) => f.entries > 0 && f.violating === 0)).toBe(true);
  });
  it("honest LASSO: impute and fit rows are training rows", async () => {
    const trace = new RowTrace();
    const spec = cvSpec(data.y, { trace, model: { kind: "lasso", nLambda: 20, rule: "min" }, repeats: 1 });
    await nestedCv(data.X, task, dev, spec, new Rng(2));
    const r = checkRowUse(trace);
    expect(r.level).toBe("ok");
    expect(r.findings.map((f) => f.stage).sort()).toEqual(["fit", "impute"]);
  });
  it("negative control: ranking features on the whole development set first (a real leak) is caught", async () => {
    const trace = new RowTrace();
    await nestedCv(data.X, task, dev, cvSpec(data.y, { trace }), new Rng(2), undefined, { rankOnAll: true });
    const r = checkRowUse(trace);
    expect(r.level).toBe("ng"); // 順位付け（目的変数を使う）が訓練側でない行を使った
    const rank = r.findings.filter((f) => f.stage === "rank" && f.violating > 0);
    expect(rank.length).toBeGreaterThan(0);
    expect(rank.every((f) => f.maxOutside > 0)).toBe(true);
    // 推定に使った行の数で独立に確かめる: 外側の訓練は 36 行、dev は 48 行なので、外側の検証の 12 行が混ざる
    expect(rank.find((f) => f.where === "fold")!.maxOutside).toBe(12);
  });
  it("a hand-made trace that reads a validation row is NG; reading only training rows is OK", () => {
    const mk = (stage: "fit" | "impute", rows: number[]) => {
      const t = new RowTrace();
      t.contexts.set("r0f0", { train: [0, 1, 2, 3] });
      t.scope = "r0f0";
      t.note(stage, rows);
      return checkRowUse(t);
    };
    expect(mk("fit", [0, 1, 2, 3]).level).toBe("ok");
    expect(mk("fit", [0, 1, 2]).level).toBe("ok");
    const bad = mk("fit", [0, 1, 2, 3, 4]);
    expect(bad.level).toBe("ng"); // 当てはめ（目的変数を使う）が検証側の行を読んだ
    expect(bad.findings[0]).toMatchObject({ stage: "fit", where: "fold", usesOutcome: true, entries: 1, violating: 1, maxOutside: 1 });
    const free = mk("impute", [0, 1, 2, 3, 4]);
    expect(free.level).toBe("notice"); // 補完（特徴量の値だけ）が検証側の行を読んだ: 注意
    expect(free.findings[0]).toMatchObject({ stage: "impute", usesOutcome: false, violating: 1 });
  });
  it("quality control reads all development rows (reported as a finding, not fixed); with nothing enabled it reads no row", () => {
    const X = data.X;
    const on = qcFeatures(X, dev, X[0].map((_, j) => `f${j}`), { dropConstant: true, correlationThreshold: null, exclude: null });
    const off = qcFeatures(X, dev, X[0].map((_, j) => `f${j}`), { dropConstant: false, correlationThreshold: null, exclude: ["f1"] });
    expect(on.usedRows).toEqual(dev);
    expect(off.usedRows).toEqual([]);
    const trace = new RowTrace();
    trace.note("qc", on.usedRows);
    trace.contexts.set("r0f0", { train: dev.filter((i) => i % 4 !== 0) });
    trace.contexts.set("r0f1", { train: dev.filter((i) => i % 4 !== 1) });
    const r = checkRowUse(trace);
    expect(r.level).toBe("notice"); // 品質管理は結果ラベルを使わない: NG ではなく注意
    expect(r.findings).toEqual([{ stage: "qc", where: "pre", usesOutcome: false, entries: 2, violating: 2, maxOutside: 12 }]);
    // 同じ記録に、目的変数を使う段階の違反が加わると NG
    trace.note("rank", dev);
    expect(checkRowUse(trace).level).toBe("ng");
  });
});

// ---------------------------------------------------------------------------------------------
// 5. 2D の行の絞り込みと集約
// ---------------------------------------------------------------------------------------------
const PATIENTS = Array.from({ length: 8 }, (_, i) => `P${i + 1}`);
const UNITS: [string, string, string][] = [["WHOLE_3D", "", ""], ["MAX_SECTION_2D", "ACQUISITION", "AXIAL"], ["SLICE_2D", "RESLICED", "SAGITTAL"]];
const feat2d = ["CaseLabel,PatientID,SubjectCode,Unit,PlaneSource,Plane,RowKey,f1,f2,Roi_Voxels",
  ...PATIENTS.flatMap((p, i) => UNITS.map(([u, ps, pl], k) => `C${String(i + 1).padStart(3, "0")},${p},S${i},${u},${ps},${pl},C${i + 1}|${u}|${ps}|${pl}|${k},${1 + i + k * 10},${i * 2 - k},${100 * (k + 1)}`))].join("\n") + "\n";
const clin2d = "PatientID,time,event,outcome\n" + PATIENTS.map((p, i) => `${p},${5 + i},${i % 2},${i % 2 ? "resp" : "no"}`).join("\n") + "\n";
const spec2d = (over: Partial<DatasetSpec> = {}): DatasetSpec => ({
  features: [sourceOf("roi-features-batch.csv", feat2d)], featureIdColumn: "RowKey", patientColumn: "PatientID", featureColumns: null,
  clinical: sourceOf("clin.csv", clin2d), clinicalIdColumn: "PatientID", roles: { time: null, event: null, outcome: null, covariates: [] }, ...over,
});

describe("2D rows: row filter, duplicate ID guidance, per-patient mean", () => {
  it("a 2D table with CaseLabel as the row ID stops with the duplicate error and the guidance", () => {
    const r = buildDataset(spec2d({ featureIdColumn: "CaseLabel", patientColumn: "PatientID" }));
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.errors[0]).toContain("行の ID「C001」が重複しています（roi-features-batch.csv）");
    const hint = r.errors[r.errors.length - 1];
    expect(hint).toContain("「RowKey」を行の ID にする");
    expect(hint).toContain("Unit・PlaneSource・Plane で絞り込む");
  });
  it("no guidance is added when the table has none of those columns", () => {
    const r = buildDataset({ ...spec2d(), features: [sourceOf("a.csv", "CaseLabel,f1\nA,1\nA,2\n")], featureIdColumn: "CaseLabel", patientColumn: null });
    if (r.ok) throw new Error("should fail");
    expect(r.errors.join("")).not.toContain("RowKey");
  });
  it("RowKey as the row ID: no duplicate stop; all units are averaged per patient (3 rows each) and recorded", () => {
    const r = buildDataset(spec2d());
    if (!r.ok) throw new Error(r.errors.join("\n"));
    expect(r.report.joined).toBe(8);
    expect(r.report.featureRows).toBe(24);
    expect(r.report.filteredRows).toBe(0);
    expect(r.report.averagedPatients).toBe(8);
    expect(r.report.averagedRows).toBe(24);
    expect(r.report.lesionsPerPatient).toEqual({ min: 3, max: 3 });
    expect(r.dataset.featureNames).toEqual(["f1", "f2"]); // Roi_ は特徴量でない
    expect(r.dataset.features[0]).toEqual([(1 + 11 + 21) / 3, (0 - 1 - 2) / 3]); // P1 の 3 行の平均
  });
  it("filter WHOLE_3D only: one row per patient, nothing averaged, the removed rows are counted", () => {
    const r = buildDataset(spec2d({ rowFilter: [{ column: "Unit", values: ["WHOLE_3D"] }] }));
    if (!r.ok) throw new Error(r.errors.join("\n"));
    expect(r.report.featureRows).toBe(8);
    expect(r.report.filteredRows).toBe(16);
    expect(r.report.averagedPatients).toBe(0);
    expect(r.dataset.features[0]).toEqual([1, 0]);
    expect(r.dataset.features[2]).toEqual([3, 4]);
  });
  it("filter on two units averages exactly those rows; filters on several columns are combined with AND", () => {
    const r = buildDataset(spec2d({ rowFilter: [{ column: "Unit", values: ["MAX_SECTION_2D", "SLICE_2D"] }] }));
    if (!r.ok) throw new Error(r.errors.join("\n"));
    expect(r.report.averagedRows).toBe(16);
    expect(r.dataset.features[0][0]).toBe((11 + 21) / 2);
    const both = buildDataset(spec2d({ rowFilter: [{ column: "Unit", values: ["MAX_SECTION_2D", "SLICE_2D"] }, { column: "Plane", values: ["AXIAL"] }] }));
    if (!both.ok) throw new Error("not ok");
    expect(both.report.featureRows).toBe(8);
    expect(both.dataset.features[0][0]).toBe(11);
  });
  it("a filter that leaves no row stops with an error; an unknown column too", () => {
    const none = buildDataset(spec2d({ rowFilter: [{ column: "Unit", values: [] }] }));
    expect(none.ok).toBe(false);
    if (!none.ok) expect(none.errors[0]).toContain("1 行も残りません");
    const none2 = buildDataset(spec2d({ rowFilter: [{ column: "Unit", values: ["NOPE"] }] }));
    expect(none2.ok).toBe(false);
    const e: string[] = [];
    applyRowFilter(spec2d().features, [{ column: "Nope", values: ["x"] }], e);
    expect(e[0]).toContain("絞り込みの列「Nope」がありません");
  });
  it("excludeColumns removes a column from the default features", () => {
    const r = buildDataset(spec2d({ excludeColumns: ["f2"] }));
    if (!r.ok) throw new Error("not ok");
    expect(r.dataset.featureNames).toEqual(["f1"]);
  });
  it("suggestJoin picks RowKey as the row ID and CaseLabel/PatientID as the patient for a 2D table", () => {
    const f = parseCsv(feat2d), c = parseCsv(clin2d);
    const s = suggestJoin([f], c, suggestColumns(f, c));
    expect(s.featureId).toBe("RowKey");
    expect(s.patient).toBe("PatientID");
    expect(s.clinicalId).toBe("PatientID");
  });
  it("RowKey is preferred as the row ID even when another unique column comes first", () => {
    const f = parseCsv(feat2d.split("\n").filter(Boolean).map((l, r) => (r === 0 ? `Idx,${l}` : `${r},${l}`)).join("\n"));
    const c = parseCsv(clin2d);
    const s = suggestJoin([f], c, suggestColumns(f, c));
    expect(s.featureId).toBe("RowKey");
    expect(s.patient).toBe("PatientID");
  });
  it("the filter and the number of averaged rows are in the run record and the method text", async () => {
    const params = { ...DATASET_CHECK.defaults, featureIdColumn: "RowKey", patientColumn: "PatientID", clinicalIdColumn: "PatientID", rowFilter: [{ column: "Unit", values: ["MAX_SECTION_2D", "SLICE_2D"] }], excludeFeatureColumns: ["f2"] };
    const { record } = await runSkill(DATASET_CHECK, { features: [sourceOf("roi.csv", feat2d)], clinical: sourceOf("clin.csv", clin2d) }, params, { seed: 1, pluginVersion: "t", now });
    expect(record.status).toBe("ok");
    const join = record.steps.find((s) => s.id === "join")!;
    expect(join.summary).toMatchObject({ rowFilter: "Unit=MAX_SECTION_2D|SLICE_2D", filteredRows: 8, averagedPatients: 8, averagedRows: 16, lesionsMin: 2, lesionsMax: 2, features: 1, excludedFeatureColumns: "f2" });
    expect(record.params.rowFilter).toEqual([{ column: "Unit", values: ["MAX_SECTION_2D", "SLICE_2D"] }]);
    const ja = methodText(record, DATASET_CHECK, "ja"), en = methodText(record, DATASET_CHECK, "en");
    expect(ja).toContain("Unit=MAX_SECTION_2D|SLICE_2D に絞り込んだ（外した行 8 行）");
    expect(ja).toContain("8 人（計 16 行、患者あたり 2〜2 行）は、欠損を除く平均で患者ごとに 1 行");
    expect(en).toContain("restricted to Unit=MAX_SECTION_2D|SLICE_2D (8 rows removed)");
    expect(en).toContain("For 8 patients with several rows (16 rows in total; 2 to 2 rows per patient)");
    expect(ja).toContain("rowFilter=[Unit=MAX_SECTION_2D|SLICE_2D]"); // 既定値から変更した設定
    expect(ja).not.toContain("[object Object]");
  });
  it("the leak step of the check records ID-like columns in use and target leaks", async () => {
    const feat = "CaseLabel,Patient,RowNo,f1,y\n" + PATIENTS.map((p, i) => `${p}_L,${p},${i + 1},${(i * 0.37) % 1},${(i % 2 ? 20 : 0) + (i % 3)}`).join("\n") + "\n";
    const clin = "Patient,time,event,outcome\n" + PATIENTS.map((p, i) => `${p},${5 + i},${i % 2},${i % 2 ? "resp" : "no"}`).join("\n") + "\n";
    const params = { ...DATASET_CHECK.defaults, featureIdColumn: "CaseLabel", patientColumn: "Patient", clinicalIdColumn: "Patient", outcomeColumn: "outcome" };
    const { record } = await runSkill(DATASET_CHECK, { features: [sourceOf("f.csv", feat)], clinical: sourceOf("c.csv", clin) }, params, { seed: 1, pluginVersion: "t", now });
    const leak = record.steps.find((s) => s.id === "leak")!;
    expect(leak.summary.idLikeUsed).toBe(1);
    expect(leak.summary.idLikeColumns).toBe("RowNo");
    expect(leak.summary.targetLeaks).toBe(1);
    expect(leak.summary.targetLeakDetail).toBe("y:outcome:separates");
    expect(leak.warnings).toHaveLength(2);
  });
});
