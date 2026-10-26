import { describe, expect, it } from "vitest";
import { buildDataset, defaultFeatureColumns, meanOrNull, type DatasetSpec } from "../src/core/dataset";
import { parseCsv } from "../src/core/csv";
import { sourceOf } from "../src/core/delta";

const FEAT = "CaseLabel,Patient,f1,f2,Provenance_X,Roi_Voxels\nL1,P1,1,10,a,100\nL2,P1,3,NA,a,200\nL3,P2,5,20,a,300\nL4,P3,7,30,a,400\n";
const CLIN = "Patient,time,event,age,outcome\nP1,12,1,60,resp\nP2,8,0,70,no\nP4,5,1,50,resp\n";

const spec = (over: Partial<DatasetSpec> = {}): DatasetSpec => ({
  features: [sourceOf("feat.csv", FEAT)],
  featureIdColumn: "CaseLabel",
  patientColumn: "Patient",
  featureColumns: null,
  clinical: sourceOf("clin.csv", CLIN),
  clinicalIdColumn: "Patient",
  roles: { time: "time", event: "event", outcome: "outcome", covariates: ["age"] },
  ...over,
});

describe("buildDataset", () => {
  it("joins by patient and reports the one-sided patients", () => {
    const r = buildDataset(spec());
    if (!r.ok) throw new Error(r.errors.join("\n"));
    expect(r.dataset.ids).toEqual(["P1", "P2"]);
    expect(r.report.joined).toBe(2);
    expect(r.report.patientsInFeatures).toBe(3);
    expect(r.report.patientsInClinical).toBe(3);
    expect(r.report.featureOnly).toEqual(["P3"]);
    expect(r.report.clinicalOnly).toEqual(["P4"]);
    expect(r.report.exclusions.map((e) => `${e.id}:${e.reason}`)).toEqual(["P3:臨床の表にいない", "P4:特徴量の表にいない"]);
    expect(r.report.lesionsPerPatient).toEqual({ min: 1, max: 2 });
    expect(r.report.featureRows).toBe(4);
  });
  it("averages the lesions of a patient, ignoring missing values and never filling 0", () => {
    const r = buildDataset(spec());
    if (!r.ok) throw new Error("not ok");
    expect(r.dataset.featureNames).toEqual(["f1", "f2"]); // Provenance_・Roi_ は特徴量にしない
    expect(r.dataset.features[0]).toEqual([2, 10]); // P1: f1=(1+3)/2, f2=10（欠損を 0 とみて 5 にしない）
    expect(r.dataset.features[1]).toEqual([5, 20]);
    expect(r.dataset.lesionCounts).toEqual([2, 1]);
  });
  it("keeps a patient whose lesions are all missing as missing, and counts it", () => {
    const feat = "CaseLabel,Patient,f1\nL1,P1,NA\nL2,P1,\nL3,P2,4\n";
    const r = buildDataset(spec({ features: [sourceOf("f.csv", feat)] }));
    if (!r.ok) throw new Error("not ok");
    expect(r.dataset.features).toEqual([[null], [4]]);
    expect(r.report.missing).toEqual([{ name: "f1", count: 1, of: 2 }]);
  });
  it("reads the clinical roles with their types", () => {
    const r = buildDataset(spec());
    if (!r.ok) throw new Error("not ok");
    const by = Object.fromEntries(r.dataset.clinical.map((c) => [c.name, c.values]));
    expect(by.time).toEqual([12, 8]);
    expect(by.event).toEqual([1, 0]);
    expect(by.age).toEqual([60, 70]);
    expect(by.outcome).toEqual(["resp", "no"]);
    expect(r.report.survivalUsable).toBe(2);
  });
  it("excludes a patient with a missing time or event from the survival count and says so", () => {
    const clin = "Patient,time,event\nP1,12,1\nP2,NA,0\n";
    const r = buildDataset(spec({ clinical: sourceOf("c.csv", clin), roles: { time: "time", event: "event", outcome: null, covariates: [] } }));
    if (!r.ok) throw new Error("not ok");
    expect(r.report.survivalUsable).toBe(1);
    expect(r.report.exclusions.some((e) => e.id === "P2" && /生存解析から除く/.test(e.reason))).toBe(true);
    expect(r.report.missing.find((m) => m.name === "time")).toEqual({ name: "time", count: 1, of: 2 });
  });
  it("stops on a duplicated patient in the clinical table (negative)", () => {
    const clin = CLIN + "P1,3,0,40,no\n";
    const r = buildDataset(spec({ clinical: sourceOf("c.csv", clin) }));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors.join()).toMatch(/患者 ID「P1」が重複/);
  });
  it("stops on a duplicated row id, and on a duplicated patient when one row means one patient (negative)", () => {
    const dupRow = "CaseLabel,Patient,f1\nL1,P1,1\nL1,P2,2\n";
    const a = buildDataset(spec({ features: [sourceOf("f.csv", dupRow)] }));
    expect(!a.ok && a.errors.join()).toMatch(/行の ID「L1」が重複/);
    const b = buildDataset(spec({ patientColumn: null, features: [sourceOf("f.csv", "CaseLabel,f1\nP1,1\nP1,2\n")], clinicalIdColumn: "Patient" }));
    expect(!b.ok && b.errors.join()).toMatch(/重複/);
  });
  it("stops when the event is not 0 or 1 or the time is not positive, naming the value (negative)", () => {
    const bad = buildDataset(spec({ clinical: sourceOf("c.csv", "Patient,time,event\nP1,12,2\nP2,0,1\n"), roles: { time: "time", event: "event", outcome: null, covariates: [] } }));
    expect(bad.ok).toBe(false);
    if (!bad.ok) {
      expect(bad.errors.join("\n")).toMatch(/イベントは 0 か 1.*P1=2/);
      expect(bad.errors.join("\n")).toMatch(/時間は正の数.*P2=0/);
    }
  });
  it("stops on a non-numeric value in a feature column that was chosen, naming the cell (negative)", () => {
    const feat = "CaseLabel,Patient,f1\nL1,P1,1\nL2,P2,high\n";
    const r = buildDataset(spec({ features: [sourceOf("f.csv", feat)], featureColumns: ["f1"] }));
    expect(!r.ok && r.errors.join()).toMatch(/f.csv の 3 行目「f1」= high/);
  });
  it("concatenates feature files with the same header and refuses different headers (negative)", () => {
    const a = sourceOf("a.csv", "CaseLabel,Patient,f1\nL1,P1,1\n");
    const b = sourceOf("b.csv", "CaseLabel,Patient,f1\nL2,P2,3\n");
    const r = buildDataset(spec({ features: [a, b] }));
    if (!r.ok) throw new Error(r.errors.join());
    expect(r.dataset.features).toEqual([[1], [3]]);
    expect(r.report.featureFiles).toBe(2);
    const c = sourceOf("c.csv", "CaseLabel,Patient,g1\nL3,P3,3\n");
    const bad = buildDataset(spec({ features: [a, c] }));
    expect(!bad.ok && bad.errors.join()).toMatch(/c.csv: 見出しが a.csv と違います/);
  });
  it("warns about IDs that differ only in letter case, and still does not join them", () => {
    const clin = "Patient,time,event\np1,12,1\n";
    const r = buildDataset(spec({ clinical: sourceOf("c.csv", clin), roles: { time: "time", event: "event", outcome: null, covariates: [] } }));
    if (!r.ok) throw new Error("not ok");
    expect(r.report.joined).toBe(0);
    expect(r.report.warnings.join()).toMatch(/大文字小文字/);
    expect(r.report.warnings.join()).toMatch(/結合できた患者が 0 人/);
  });
  it("names the missing column when a role column does not exist (negative)", () => {
    const r = buildDataset(spec({ roles: { time: "months", event: "event", outcome: null, covariates: [] } }));
    expect(!r.ok && r.errors.join()).toMatch(/列「months」がありません/);
  });
  it("refuses one column in two roles (negative)", () => {
    const r = buildDataset(spec({ roles: { time: "time", event: "event", outcome: null, covariates: ["time"] } }));
    expect(!r.ok && r.errors.join()).toMatch(/同じ列を複数の役割/);
  });
});

describe("helpers", () => {
  it("defaultFeatureColumns keeps numeric columns and drops record columns and identifiers", () => {
    const t = parseCsv("CaseLabel,Patient,f1,note,Scale2mm_Mean,Scale2mm_Filter_Kind,Provenance_Sha,Roi_Voxels,allna\nL1,P1,1,x,5,LoG,a,3,NA\n");
    expect(defaultFeatureColumns(t, ["CaseLabel", "Patient"])).toEqual(["f1", "Scale2mm_Mean"]);
  });
  it("meanOrNull ignores missing and returns null for none", () => {
    expect(meanOrNull([1, null, 3])).toBe(2);
    expect(meanOrNull([null, null])).toBeNull();
    expect(meanOrNull([])).toBeNull();
  });
});
