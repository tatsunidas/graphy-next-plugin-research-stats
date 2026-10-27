import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { addEvaluation, canonicalJson, checkLock, fingerprint, parseLock, sealLock, type LockBody } from "../src/core/pipeline/lockFile";

const body = async (): Promise<LockBody> => ({
  format: "research-stats-test-lock", version: 1, createdAt: "2026-10-27T00:00:00.000Z", pluginVersion: "0.5.0",
  outcomeKind: "binary", seed: 42, testFraction: 0.25, stratifiedBy: "outcome",
  cohortFingerprint: await fingerprint(["p1", "p2", "p3", "p4"], ["yes", "no", "yes", "no"]), cohortSize: 4,
  testIds: ["p2"], evaluations: [],
});

describe("canonical JSON and fingerprints", () => {
  it("sorts keys at every level and is independent of key order", () => {
    expect(canonicalJson({ b: 1, a: [{ d: "x", c: null }] })).toBe('{"a":[{"c":null,"d":"x"}],"b":1}');
    expect(canonicalJson({ a: 1, b: 2 })).toBe(canonicalJson({ b: 2, a: 1 }));
    expect(() => canonicalJson({ a: NaN })).toThrow();
  });
  it("seal hash equals SHA-256 of the canonical body computed with node:crypto", async () => {
    const b = await body();
    const l = await sealLock(b);
    expect(l.sha256).toBe(createHash("sha256").update(canonicalJson(b), "utf8").digest("hex"));
  });
  it("cohort fingerprint ignores row order but sees a changed outcome or ID", async () => {
    const a = await fingerprint(["p1", "p2"], ["yes", "no"]);
    expect(await fingerprint(["p2", "p1"], ["no", "yes"])).toBe(a);
    expect(await fingerprint(["p1", "p2"], ["no", "no"])).not.toBe(a);
    expect(await fingerprint(["p1", "p3"], ["yes", "no"])).not.toBe(a);
  });
});

describe("lock file", () => {
  it("round-trips, and each evaluation adds 1 to the count", async () => {
    const l0 = await sealLock(await body());
    const p0 = await parseLock(JSON.stringify(l0));
    expect("error" in p0).toBe(false);
    const l1 = await addEvaluation(l0, { at: "t1", featuresFingerprint: "f", summary: { auc: 0.7 } });
    const l2 = await addEvaluation(l1, { at: "t2", featuresFingerprint: "g", summary: { auc: 0.71 } });
    expect(l2.evaluations.map((e) => e.n)).toEqual([1, 2]);
    expect(l2.sha256).not.toBe(l1.sha256);
    expect("error" in (await parseLock(JSON.stringify(l2, null, 2)))).toBe(false);
  });
  it("detects edits to the test IDs, the history or the seed", async () => {
    const l = await addEvaluation(await sealLock(await body()), { at: "t1", featuresFingerprint: "f", summary: { auc: 0.7 } });
    const edits: ((o: Record<string, unknown>) => void)[] = [
      (o) => { o.testIds = ["p3"]; },
      (o) => { o.evaluations = []; },
      (o) => { o.seed = 43; },
      (o) => { (o.evaluations as { summary: { auc: number } }[])[0].summary.auc = 0.9; },
    ];
    for (const ed of edits) {
      const o = JSON.parse(JSON.stringify(l));
      ed(o);
      const r = await parseLock(JSON.stringify(o));
      expect("error" in r && r.error).toMatch(/書き換え|壊れて/);
    }
  });
  it("rejects a sealed file whose history is not numbered 1, 2, …", async () => {
    const b = await body();
    const ev = { at: "t", featuresFingerprint: "f", summary: {} };
    const l = await sealLock({ ...b, evaluations: [{ ...ev, n: 1 }, { ...ev, n: 1 }] });
    const r = await parseLock(JSON.stringify(l));
    expect("error" in r && r.error).toMatch(/履歴/);
  });
  it("rejects files that are not lock files", async () => {
    expect("error" in (await parseLock("{"))).toBe(true);
    expect("error" in (await parseLock('{"format":"other"}'))).toBe(true);
    const l = await sealLock(await body());
    expect("error" in (await parseLock(JSON.stringify({ ...l, version: 2 })))).toBe(true);
  });
  it("can only be used with the same cohort and outcome kind", async () => {
    const l = await sealLock(await body());
    expect(checkLock(l, "binary", l.cohortFingerprint, ["p1", "p2", "p3", "p4"])).toBeNull();
    expect(checkLock(l, "survival", l.cohortFingerprint, ["p1", "p2", "p3", "p4"])).toMatch(/結果の型/);
    const other = await fingerprint(["p1", "p2", "p3", "p4"], ["yes", "yes", "yes", "no"]);
    expect(checkLock(l, "binary", other, ["p1", "p2", "p3", "p4"])).toMatch(/違います/);
    expect(checkLock(l, "binary", l.cohortFingerprint, ["p1", "p3"])).toMatch(/いません/);
  });
});
