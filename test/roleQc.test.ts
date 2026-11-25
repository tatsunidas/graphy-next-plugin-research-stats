import { describe, expect, it } from "vitest";
import { parseCsv } from "../src/core/csv";
import { covariateQc, eventQc, outcomeQc, rolesForSkill, sanitizeRoles, timeQc } from "../src/core/roleQc";

const t = parseCsv("id,time,event,grp,age,sex\na,12,1,x,50,M\nb,0,0,y,NA,F\nc,abc,2,y,61,F\nd,-3,,z,7.5,F\ne,,1,x,60,M\nf,5.5,0,x,,M\n");

describe("role column QC", () => {
  it("time: counts positives, missing and invalid values", () => {
    const q = timeQc(t, "time")!;
    expect([q.rows, q.positive, q.missing, q.invalid]).toEqual([6, 2, 1, 3]);
    expect(q.invalidExamples).toEqual(["0", "abc", "-3"]);
  });
  it("a clean time column has no invalid values (negative control)", () => {
    expect(timeQc(parseCsv("t\n1\n2.5\n"), "t")).toMatchObject({ positive: 2, invalid: 0, missing: 0 });
  });
  it("event: 0 and 1 counted, anything else (2, text) invalid", () => {
    const q = eventQc(t, "event")!;
    expect([q.zero, q.one, q.missing, q.invalid]).toEqual([2, 2, 1, 1]);
    expect(q.invalidExamples).toEqual(["2"]);
    expect(eventQc(parseCsv("e\n0\n1\n1\n"), "e")).toMatchObject({ invalid: 0 });
  });
  it("outcome: distinct labels with counts, warns when not binary", () => {
    const q = outcomeQc(t, "grp")!;
    expect(q.values).toEqual([{ label: "x", count: 3 }, { label: "y", count: 2 }, { label: "z", count: 1 }]);
    expect(q.binary).toBe(false);
    expect(outcomeQc(t, "sex")).toMatchObject({ binary: true, missing: 0 });
  });
  it("covariate: numeric or not, missing counted (not zero-filled)", () => {
    expect(covariateQc(t, "age")).toMatchObject({ numeric: true, missing: 2 });
    expect(covariateQc(t, "sex")).toMatchObject({ numeric: false, textExamples: ["M", "F"] });
  });
  it("returns null for a column that is not in the table", () => {
    expect(timeQc(t, "nope")).toBeNull();
  });
});

describe("roles per skill", () => {
  it("lists only the roles each analysis needs", () => {
    expect(rolesForSkill("r1-prognosis", {})).toEqual({ time: true, event: true, outcome: false, positive: false, covariates: true });
    expect(rolesForSkill("r2-discrimination", {})).toMatchObject({ time: false, event: false, outcome: true, positive: true });
    expect(rolesForSkill("r3-reproducibility", {})).toEqual({ time: false, event: false, outcome: false, positive: false, covariates: false });
    expect(rolesForSkill("r5-model", {})).toMatchObject({ time: true, event: true, outcome: false });
    expect(rolesForSkill("p4-pipeline", { outcomeKind: "binary" })).toMatchObject({ outcome: true, positive: true, time: false });
    expect(rolesForSkill("p4-pipeline", { outcomeKind: "survival" })).toMatchObject({ outcome: false, time: true, event: true });
    expect(rolesForSkill("p4-pipeline", { outcomeKind: null })).toMatchObject({ outcome: false, time: false });
    expect(rolesForSkill(null, {})).toMatchObject({ time: false });
  });
  it("drops the columns an analysis does not use", () => {
    const p = sanitizeRoles(rolesForSkill("r2-discrimination", {}), { timeColumn: "time", eventColumn: "event", outcomeColumn: "grp", covariates: ["age"], seedLike: 1 });
    expect(p).toMatchObject({ timeColumn: null, eventColumn: null, outcomeColumn: "grp", covariates: [], seedLike: 1 });
  });
});
