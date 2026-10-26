import { describe, expect, it } from "vitest";
import { buildDelta, deltaToTable, percentChange, sourceOf } from "../src/core/delta";
import { buildDataset } from "../src/core/dataset";

const BASE = "CaseLabel,Patient,f1\nL1,P1,10\nL2,P1,20\nL3,P2,0\nL4,P2,4\nL5,P3,6\n";
const FOLLOW = "CaseLabel,Patient,f1\nL1,P1,15\nL2,P1,10\nL3,P2,5\nL4,P2,6\nL6,P3,1\n";

const spec = (order: "lesion-then-mean" | "mean-then-change") => ({
  baseline: [sourceOf("base.csv", BASE)], followup: [sourceOf("follow.csv", FOLLOW)],
  idColumn: "CaseLabel", patientColumn: "Patient", featureColumns: ["f1"], order,
});

describe("percentChange", () => {
  it("divides by the absolute baseline so an increase is positive for a negative baseline too", () => {
    expect(percentChange(10, 15)).toBeCloseTo(50, 12);
    expect(percentChange(-10, -5)).toBeCloseTo(50, 12);
    expect(percentChange(-10, -15)).toBeCloseTo(-50, 12);
  });
  it("does not make a number from a zero baseline or a missing value (negative)", () => {
    expect(percentChange(0, 5)).toBe("zero-baseline");
    expect(percentChange(null, 5)).toBe("missing");
    expect(percentChange(5, null)).toBe("missing");
  });
});

describe("buildDelta", () => {
  it("lesion-then-mean: changes per lesion, then the patient mean (hand-computed)", () => {
    const r = buildDelta(spec("lesion-then-mean"));
    if (!r.ok) throw new Error(r.errors.join());
    expect(r.result.patients).toEqual(["P1", "P2"]); // P3 は後の時点に L5 が無い
    // P1: (15-10)/10=+50%, (10-20)/20=-50% → 平均 0
    expect(r.result.values[0][0]).toBeCloseTo(0, 12);
    // P2: L3 は基準値 0 → 欠損。L4: (6-4)/4=+50% → 平均は L4 だけで 50
    expect(r.result.values[1][0]).toBeCloseTo(50, 12);
    expect(r.result.problems).toEqual([{ id: "L3", feature: "f1", reason: "zero-baseline" }]);
  });
  it("mean-then-change: the patient means first (hand-computed) and the result differs", () => {
    const a = buildDelta(spec("lesion-then-mean"));
    const b = buildDelta(spec("mean-then-change"));
    if (!a.ok || !b.ok) throw new Error("not ok");
    // P1: 平均 前 15・後 12.5 → (12.5-15)/15 = -16.666…%
    expect(b.result.values[0][0]).toBeCloseTo(-100 / 6, 10);
    // P2: 平均 前 2・後 5.5 → (5.5-2)/2 = 175%
    expect(b.result.values[1][0]).toBeCloseTo(175, 10);
    expect(b.result.values[0][0]).not.toBeCloseTo(a.result.values[0][0]!, 3); // 順序を取り違えていない
  });
  it("lists lesions that exist at only one time point and does not use them", () => {
    const r = buildDelta(spec("lesion-then-mean"));
    if (!r.ok) throw new Error("not ok");
    expect(r.result.unmatched).toEqual([{ id: "L5", where: "baseline" }, { id: "L6", where: "followup" }]);
  });
  it("makes a missing patient value when every lesion's change is impossible, not 0", () => {
    const base = "CaseLabel,Patient,f1\nL1,P1,0\n";
    const fol = "CaseLabel,Patient,f1\nL1,P1,5\n";
    const r = buildDelta({ ...spec("lesion-then-mean"), baseline: [sourceOf("b.csv", base)], followup: [sourceOf("f.csv", fol)] });
    if (!r.ok) throw new Error("not ok");
    expect(r.result.values).toEqual([[null]]);
  });
  it("stops when a lesion belongs to different patients at the two time points (negative)", () => {
    const fol = "CaseLabel,Patient,f1\nL1,P9,15\n";
    const r = buildDelta({ ...spec("lesion-then-mean"), baseline: [sourceOf("b.csv", "CaseLabel,Patient,f1\nL1,P1,10\n")], followup: [sourceOf("f.csv", fol)] });
    expect(!r.ok && r.errors.join()).toMatch(/患者が 2 時点で違います/);
  });
  it("feeds the joined dataset through a table with one row per patient", () => {
    const r = buildDelta(spec("lesion-then-mean"));
    if (!r.ok) throw new Error("not ok");
    const t = deltaToTable(r.result, "Patient");
    expect(t.header).toEqual(["Patient", "Delta_f1"]);
    const d = buildDataset({
      features: [{ name: "delta", sha256: null, table: t }], featureIdColumn: "Patient", patientColumn: null, featureColumns: null,
      clinical: sourceOf("c.csv", "Patient,time,event\nP1,5,1\nP2,6,0\n"), clinicalIdColumn: "Patient",
      roles: { time: "time", event: "event", outcome: null, covariates: [] },
    });
    if (!d.ok) throw new Error(d.errors.join());
    expect(d.dataset.featureNames).toEqual(["Delta_f1"]);
    expect(d.dataset.features[1][0]).toBeCloseTo(50, 12);
  });
});
