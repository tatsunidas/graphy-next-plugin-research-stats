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

describe("suggestColumns: patient column", () => {
  const sug = (f: string, c: string) => suggestColumns(parseCsv(f), parseCsv(c));
  it("picks SubjectCode of a batch-extraction table, and the same-named clinical column", () => {
    const r = sug("CaseLabel,SubjectCode,f1\nL1,S1,1\n", "SubjectCode,time,event\nS1,3,1\n");
    expect([r.featureId, r.patient, r.clinicalId]).toEqual(["CaseLabel", "SubjectCode", "SubjectCode"]);
  });
  it("keeps the former defaults", () => {
    for (const name of ["Patient", "PatientID", "subject_id", "Subject", "patient_label"]) {
      expect(sug(`CaseLabel,${name},f1\nL1,P1,1\n`, "x\n1\n").patient).toBe(name);
    }
  });
  it("accepts the code spellings", () => {
    for (const name of ["Subject_Code", "subject code", "Patient Label"]) {
      expect(sug(`CaseLabel,${name},f1\nL1,P1,1\n`, "x\n1\n").patient).toBe(name);
    }
  });
  it("does not match names that only contain the words (negative)", () => {
    expect(sug("CaseLabel,SubjectCodeX,xSubjectCode,f1\nL1,a,b,1\n", "x\n1\n").patient).toBeNull();
  });
  it("never uses the feature id column as the patient", () => {
    expect(sug("SubjectCode,f1\nS1,1\n", "SubjectCode\nS1\n")).toMatchObject({ featureId: "SubjectCode", patient: null });
  });
  it("prefers the clinical column named like the patient column, ignoring case", () => {
    const r = sug("CaseLabel,Patient,f1\nL1,P1,1\n", "CaseLabel,patient,time\nL1,P1,3\n");
    expect(r.clinicalId).toBe("patient");
  });
  it("otherwise keeps the former order", () => {
    expect(sug("CaseLabel,SubjectCode,f1\nL1,S1,1\n", "a,CaseLabel,Patient\n1,L1,P1\n").clinicalId).toBe("CaseLabel");
    expect(sug("CaseLabel,f1\nL1,1\n", "a,Patient\n1,P1\n").clinicalId).toBe("Patient");
    expect(sug("CaseLabel,f1\nL1,1\n", "a,b\n1,2\n").clinicalId).toBe("a");
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
