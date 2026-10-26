import { describe, expect, it } from "vitest";
import { parseCsv } from "../src/core/csv";
import { suggestColumns } from "../src/core/suggest";
import { datasetCsv } from "../src/core/exportDataset";
import { buildDataset } from "../src/core/dataset";
import { sourceOf } from "../src/core/delta";

describe("suggestColumns", () => {
  it("picks the usual names", () => {
    const f = parseCsv("CaseLabel,Patient,f1\nL1,P1,1\n");
    const c = parseCsv("Patient,OS_months,Event,Outcome\nP1,3,1,a\n");
    expect(suggestColumns(f, c)).toEqual({ featureId: "CaseLabel", patient: "Patient", clinicalId: "Patient", time: "OS_months", event: "Event", outcome: "Outcome" });
  });
  it("falls back to the first column and suggests nothing it cannot find", () => {
    const f = parseCsv("a,b\n1,2\n");
    const c = parseCsv("x,y\n1,2\n");
    expect(suggestColumns(f, c)).toEqual({ featureId: "a", patient: null, clinicalId: "x", time: null, event: null, outcome: null });
    expect(suggestColumns(null, null).featureId).toBe("");
  });
});

describe("datasetCsv", () => {
  it("writes one row per patient with empty cells for missing values", () => {
    const r = buildDataset({
      features: [sourceOf("f.csv", "CaseLabel,f1,f2\nP1,1,\nP2,3,4\n")], featureIdColumn: "CaseLabel", patientColumn: null, featureColumns: null,
      clinical: sourceOf("c.csv", "CaseLabel,time,event\nP1,5,1\nP2,6,0\n"), clinicalIdColumn: "CaseLabel",
      roles: { time: "time", event: "event", outcome: null, covariates: [] },
    });
    if (!r.ok) throw new Error(r.errors.join());
    expect(datasetCsv(r.dataset, "CaseLabel")).toBe("CaseLabel,time,event,f1,f2\r\nP1,5,1,1,\r\nP2,6,0,3,4\r\n");
  });
});
