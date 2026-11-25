import { describe, expect, it } from "vitest";
import { parseCsv } from "../src/core/csv";
import { assignPair, idVariants, joinNotes, normalizeId, overlapPairs, sharedColumns, suggestJoin } from "../src/core/joinQc";
import { suggestColumns } from "../src/core/suggest";

const tbl = parseCsv;
const hintOf = (f: string, c: string) => suggestColumns(tbl(f), tbl(c));

// 特徴量の表: 病変ごとの行。PatientID と SubjectCode は別の体系の ID。
const FEAT = [
  "CaseLabel,SubjectCode,PatientID,f1",
  "L1,S-01,P001,1.5", "L2,S-01,P001,2.5", "L3,S-02,P002,3.5", "L4,S-03,P003,4.5", "L5,S-04,P004,5.5",
].join("\n");

describe("normalizeId", () => {
  it("unifies width, case, spaces, zero-width and control characters", () => {
    expect(normalizeId("ＰＡＴ００１")).toBe("pat001");
    expect(normalizeId("  Pat001\t")).toBe("pat001");
    expect(normalizeId("Pa​t0﻿01\u0007")).toBe("pat001");
    expect(normalizeId("P-01")).not.toBe(normalizeId("P01"));
  });
});

describe("suggestJoin: picks the pair whose values actually overlap", () => {
  it("chooses PatientID when the clinical key is PatientID (the name hint points at SubjectCode)", () => {
    const clin = "PatientID,time,event\nP001,10,1\nP002,8,0\nP003,6,1\nP004,4,0\nP099,3,0\n";
    const hint = hintOf(FEAT, clin);
    expect(hint.patient).toBe("SubjectCode"); // 名前の推測は外れる
    const r = suggestJoin([tbl(FEAT)], tbl(clin), hint);
    expect(r.best).toMatchObject({ featureColumn: "PatientID", clinicalColumn: "PatientID", normalized: 4, exact: 4 });
    expect([r.featureId, r.patient, r.clinicalId]).toEqual(["CaseLabel", "PatientID", "PatientID"]);
  });
  it("chooses SubjectCode when the clinical key is SubjectCode", () => {
    const clin = "SubjectCode,time,event\nS-01,10,1\nS-02,8,0\nS-03,6,1\nS-09,4,0\n";
    const r = suggestJoin([tbl(FEAT)], tbl(clin), hintOf(FEAT, clin));
    expect(r.best).toMatchObject({ featureColumn: "SubjectCode", clinicalColumn: "SubjectCode", normalized: 3 });
    expect([r.patient, r.clinicalId]).toEqual(["SubjectCode", "SubjectCode"]);
  });
  it("uses the row column as the row id when it is the best key and unique", () => {
    const feat = "CaseLabel,f1\nA,1.5\nB,2.5\nC,3.5\n";
    const clin = "ID,time\nA,1.5\nB,2.5\nZ,3.5\n";
    const r = suggestJoin([tbl(feat)], tbl(clin), hintOf(feat, clin));
    expect([r.featureId, r.patient, r.clinicalId]).toEqual(["CaseLabel", null, "ID"]);
  });
  it("breaks a tie in favour of the name-based hint", () => {
    const feat = "CaseLabel,SubjectCode,PatientID,f1\nL1,P1,P1,1\nL2,P2,P2,2\n";
    const clin = "PatientID,time\nP1,1\nP2,2\n";
    const hint = hintOf(feat, clin);
    const r = suggestJoin([tbl(feat)], tbl(clin), hint);
    expect(r.pairs[0].normalized).toBe(r.pairs[1].normalized);
    expect(r.best?.featureColumn).toBe(hint.patient);
  });
  it("ignores fractional numeric columns, whose values may coincide with clinical numbers (negative)", () => {
    const feat = "CaseLabel,f1\nA,1.5\nB,2.5\nC,3.5\n";
    const clin = "ID,time\nA,1.5\nB,2.5\nZ,3.5\n";
    expect(overlapPairs([tbl(feat)], tbl(clin)).some((p) => p.featureColumn === "f1" || p.clinicalColumn === "time")).toBe(false);
  });
  it("returns no best pair when nothing overlaps (negative)", () => {
    const clin = "PatientID,time\nX1,1\nX2,2\n";
    const r = suggestJoin([tbl(FEAT)], tbl(clin), hintOf(FEAT, clin));
    expect(r.best).toBeNull();
    expect(r.pairs.every((p) => p.normalized === 0)).toBe(true);
  });
  it("counts overlaps through normalization but reports the exact count separately", () => {
    const feat = "CaseLabel,Patient,f1\nL1,ＰＡＴ１,1\nL2,pat2,2\nL3,Pat3,3\n";
    const clin = "Patient,time\nPAT１,1\nPAT2,2\nPat3,3\n";
    const p = overlapPairs([tbl(feat)], tbl(clin)).find((x) => x.featureColumn === "Patient" && x.clinicalColumn === "Patient")!;
    expect([p.normalized, p.exact]).toEqual([3, 1]);
  });
});

describe("assignPair", () => {
  it("keeps the hint row id when the key column repeats across lesions", () => {
    const pair = overlapPairs([tbl(FEAT)], tbl("PatientID\nP001\n")).find((x) => x.featureColumn === "PatientID")!;
    expect(assignPair([tbl(FEAT)], pair, "CaseLabel")).toEqual({ featureId: "CaseLabel", patient: "PatientID" });
  });
});

describe("idVariants / joinNotes", () => {
  it("finds IDs that differ only by case, width, spaces or zero-width characters", () => {
    expect(idVariants(["ＰＡＴ１", "pat2​", "other"], ["PAT1", "Pat2", "zzz"])).toEqual([{ feature: "ＰＡＴ１", clinical: "PAT1" }, { feature: "pat2​", clinical: "Pat2" }]);
    expect(idVariants(["A1"], ["A2"])).toEqual([]);
  });
  const feat = [tbl("CaseLabel,Patient,age,f1\nL1,P1,50,1\nL2,P2,60,2\n")];
  const clin = tbl("Patient,age,time\nP1,50,3\nP2,60,4\n");
  it("warns about same-named columns other than the ID columns", () => {
    expect(sharedColumns(feat, clin, ["Patient", "CaseLabel"])).toEqual(["age"]);
    const pairs = overlapPairs(feat, clin).sort((a, b) => b.normalized - a.normalized);
    const notes = joinNotes({ features: feat, clinical: clin, featureKey: "Patient", clinicalKey: "Patient", featureId: "CaseLabel", pairs, featureOnly: [], clinicalOnly: [] });
    expect(notes).toContainEqual({ kind: "sharedColumns", columns: ["age"] });
  });
  it("warns when no pair overlaps, and names the better pair when a worse one is in use", () => {
    const none = joinNotes({ features: feat, clinical: tbl("Patient\nX\n"), featureKey: "Patient", clinicalKey: "Patient", featureId: "CaseLabel", pairs: overlapPairs(feat, tbl("Patient\nX\n")), featureOnly: [], clinicalOnly: [] });
    expect(none.map((n) => n.kind)).toContain("noOverlap");
    const c2 = tbl("SubjectCode,PatientID\nS1,P001\n");
    const f2 = [tbl(FEAT)];
    const pairs = overlapPairs(f2, c2).sort((a, b) => b.normalized - a.normalized);
    const notes = joinNotes({ features: f2, clinical: c2, featureKey: "SubjectCode", clinicalKey: "SubjectCode", featureId: "CaseLabel", pairs, featureOnly: [], clinicalOnly: [] });
    const better = notes.find((n) => n.kind === "betterPair");
    expect(better).toBeTruthy();
  });
  it("does not warn when the used pair is the best (negative)", () => {
    const pairs = overlapPairs(feat, clin).sort((a, b) => b.normalized - a.normalized);
    const kinds = joinNotes({ features: feat, clinical: clin, featureKey: "Patient", clinicalKey: "Patient", featureId: "CaseLabel", pairs, featureOnly: [], clinicalOnly: [] }).map((n) => n.kind);
    expect(kinds).not.toContain("noOverlap");
    expect(kinds).not.toContain("betterPair");
    expect(kinds).not.toContain("variants");
  });
});
