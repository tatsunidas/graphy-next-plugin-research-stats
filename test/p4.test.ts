import { describe, expect, it } from "vitest";
import { runRecipe } from "../src/core/engine";
import { P4_PIPELINE, type P4State } from "../src/core/recipes";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { methodText } from "../src/core/methodText";
import { parseLock } from "../src/core/pipeline/lockFile";
import { p4FoldsCsv, p4PredictionsCsv } from "../src/core/exportP4";

/** f_a は強く、f_b は中くらいに結果と関係する。f_n* は無関係。f_gap は一部欠損。f_dup は f_a とほぼ同じ。 */
function synth(n: number, seed: number, o: { gap?: boolean; survival?: boolean; testShift?: (id: string) => number } = {}) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const head = ["CaseLabel", "f_a", "f_b", "f_n0", "f_n1", "f_n2", "f_dup", ...(o.gap ? ["f_gap"] : [])];
  const feat = [head.join(",")], clin = ["CaseLabel,outcome,time,event"];
  for (let i = 0; i < n; i++) {
    const g = r.next() < 0.45 ? 1 : 0;
    const a = norm() + 1.4 * g, b = norm() + 0.7 * g;
    const id = `P${i}`, sh = o.testShift?.(id) ?? 0;
    feat.push([id, (a + sh).toFixed(4), b.toFixed(4), norm().toFixed(4), norm().toFixed(4), norm().toFixed(4), (a * 1.01 + 0.001 * norm()).toFixed(4), ...(o.gap ? [i % 7 === 0 ? "" : norm().toFixed(4)] : [])].join(","));
    const time = (-Math.log(1 - r.next()) / Math.exp(0.9 * a)).toFixed(4), ev = r.next() < 0.8 ? 1 : 0;
    clin.push(`${id},${g ? "resp" : "no"},${Number(time) + 0.001},${ev}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}

const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", outcomeColumn: "outcome", timeColumn: "time", eventColumn: "event" };
const binary = { ...base, outcomeKind: "binary", positiveLabel: "resp", outerK: 5, innerK: 4, repeats: 1, kCandidates: "1, 2, 4" };
const opts = (seed = 1) => ({ seed, pluginVersion: "0.5.0", now: () => "2026-10-27T00:00:00Z" });
const run = (inputs: ReturnType<typeof synth> & { lock?: { name: string; text: string }; exclusion?: ReturnType<typeof sourceOf> }, params: Record<string, unknown>, seed = 1) =>
  runRecipe(P4_PIPELINE, inputs, params as never, opts(seed));
const lastError = (r: Awaited<ReturnType<typeof run>>) => r.record.steps.flatMap((s) => s.errors).join(" / ");

describe("P4 pipeline recipe", () => {
  it("does not run until the settings without defaults are chosen", async () => {
    const d = synth(60, 1);
    expect(lastError(await run(d, { ...base }))).toMatch(/結果の型/);
    expect(lastError(await run(d, { ...binary, lockMode: null }))).toMatch(/テスト集合の扱い/);
    expect(lastError(await run(d, { ...binary, lockMode: "none", outerK: null }))).toMatch(/外側の分割数/);
    expect(lastError(await run(d, { ...binary, lockMode: "none", kCandidates: null }))).toMatch(/候補/);
    expect(lastError(await run(d, { ...binary, lockMode: "create" }))).toMatch(/割合/);
    expect(lastError(await run(d, { ...binary, outcomeKind: "survival", lockMode: "none" }))).toMatch(/層化/);
  });

  it("create → save → use: the same seed gives the same split and cross-validation, evaluation counts 1, 2", async () => {
    const d = synth(100, 3);
    const c = await run(d, { ...binary, lockMode: "create", testFraction: 0.25 });
    expect(c.record.status).toBe("ok");
    const sc = c.state.p4 as P4State;
    expect(sc.split!.lockIsNew).toBe(true);
    expect(sc.split!.test.length).toBeGreaterThan(20);
    expect(sc.cv!.pooled[0]!).toBeGreaterThan(0.75);
    expect(c.record.steps.find((s) => s.id === "p4-test")!.status).toBe("skipped");
    const text = JSON.stringify(sc.split!.lock, null, 2);
    expect("error" in (await parseLock(text))).toBe(false);

    // 作ったばかりの固定ファイルでは評価しない（保存して読み込んでから）
    expect(lastError(await run(d, { ...binary, lockMode: "create", testFraction: 0.25, evaluateTest: true }))).toMatch(/読み込んで/);

    const u = await run({ ...d, lock: { name: "lock.json", text } }, { ...binary, lockMode: "use" });
    const su = u.state.p4 as P4State;
    expect(su.split!.test).toEqual(sc.split!.test);
    expect(su.cv!.folds).toEqual(sc.cv!.folds);
    expect(u.record.steps.find((s) => s.id === "p4-test")!.status).toBe("skipped");

    const e1 = await run({ ...d, lock: { name: "lock.json", text } }, { ...binary, lockMode: "use", evaluateTest: true });
    const t1 = (e1.state.p4 as P4State).test!;
    expect(e1.record.status).toBe("ok");
    expect(t1.evaluation).toBe(1);
    expect(t1.n).toBe(sc.split!.test.length);
    expect(t1.value!).toBeGreaterThan(0.7);
    expect(t1.lower!).toBeLessThan(t1.value!);
    expect(e1.record.steps.flatMap((s) => s.warnings).join()).not.toMatch(/回目の評価/);
    const e2 = await run({ ...d, lock: { name: "lock.json", text: JSON.stringify(t1.lock) } }, { ...binary, lockMode: "use", evaluateTest: true });
    expect((e2.state.p4 as P4State).test!.evaluation).toBe(2);
    expect(e2.record.steps.flatMap((s) => s.warnings).join()).toMatch(/2 回目の評価/);
    expect(e2.record.steps.find((s) => s.id === "p4-split")!.warnings.join()).toMatch(/既に 1 回/);
  });

  it("a hand-edited lock file or one made from other data is refused", async () => {
    const d = synth(80, 4);
    const lock = (await run(d, { ...binary, lockMode: "create", testFraction: 0.25 })).state.p4 as P4State;
    const o = JSON.parse(JSON.stringify(lock.split!.lock));
    o.testIds = o.testIds.slice(1);
    expect(lastError(await run({ ...d, lock: { name: "x", text: JSON.stringify(o) } }, { ...binary, lockMode: "use" }))).toMatch(/書き換え/);
    const other = synth(80, 5);
    expect(lastError(await run({ ...other, lock: { name: "x", text: JSON.stringify(lock.split!.lock) } }, { ...binary, lockMode: "use" }))).toMatch(/違います/);
    expect(lastError(await run(d, { ...binary, lockMode: "use" }))).toMatch(/読み込んで/);
  });

  it("test patients do not influence quality control or the cross-validation", async () => {
    const d0 = synth(90, 6);
    const c = (await run(d0, { ...binary, lockMode: "create", testFraction: 0.3 })).state.p4 as P4State;
    const testIds = new Set(c.split!.lock!.testIds);
    const text = JSON.stringify(c.split!.lock);
    const d1 = synth(90, 6, { testShift: (id) => (testIds.has(id) ? 50 : 0) });
    const a = (await run({ ...d0, lock: { name: "l", text } }, { ...binary, lockMode: "use", correlationThreshold: 0.95, dropConstant: true })).state.p4 as P4State;
    const b = (await run({ ...d1, lock: { name: "l", text } }, { ...binary, lockMode: "use", correlationThreshold: 0.95, dropConstant: true })).state.p4 as P4State;
    expect(b.qc!.removed).toEqual(a.qc!.removed);
    expect(b.cv!.folds).toEqual(a.cv!.folds);
    expect(b.cv!.pred).toEqual(a.cv!.pred);
    expect(a.qc!.removed.map((x) => x.feature)).toEqual(["f_dup"]); // f_a とほぼ同じ列は除かれる（欠損が同じなら列の順が先の f_a を残す）
  });

  it("missing values: must choose; complete drops cases with a list; impute keeps them", async () => {
    const d = synth(90, 7, { gap: true });
    expect(lastError(await run(d, { ...binary, lockMode: "none" }))).toMatch(/欠損が \d+ 個/);
    const cc = (await run(d, { ...binary, lockMode: "none", missing: "complete" })).state.p4 as P4State;
    expect(cc.qc!.droppedDev.length).toBe(13);
    expect(cc.qc!.dev.length).toBe(90 - 13);
    const im = await run(d, { ...binary, lockMode: "none", missing: "impute" });
    expect(im.record.status).toBe("ok");
    expect((im.state.p4 as P4State).qc!.dev.length).toBe(90);
  });

  it("exclusion list from the reproducibility recipe removes features", async () => {
    const d = synth(60, 8);
    const ex = sourceOf("ex.csv", "feature,category\nf_n0,poor\nf_zzz,poor\n", null);
    const r = await run({ ...d, exclusion: ex }, { ...binary, lockMode: "none" });
    const st = r.state.p4 as P4State;
    expect(st.names).not.toContain("f_n0");
    expect(r.record.steps.find((s) => s.id === "p4-qc")!.warnings.join()).toMatch(/f_zzz/);
    expect(lastError(await run({ ...d, exclusion: sourceOf("bad.csv", "name\nx\n", null) }, { ...binary, lockMode: "none" }))).toMatch(/feature/);
  });

  it("survival: stratified by event when chosen, C-index on the test set with a bootstrap interval", async () => {
    const d = synth(120, 9);
    const surv = { ...base, outcomeKind: "survival", stratifySurvival: true, outerK: 5, innerK: 3, repeats: 1, kCandidates: "1,2", testBootstrapB: 200 };
    const c = (await run(d, { ...surv, lockMode: "create", testFraction: 0.25 })).state.p4 as P4State;
    expect(c.split!.lock!.stratifiedBy).toBe("event");
    expect(c.cv!.summary.mean!).toBeGreaterThan(0.6);
    const e = await run({ ...d, lock: { name: "l", text: JSON.stringify(c.split!.lock) } }, { ...surv, lockMode: "use", evaluateTest: true });
    expect(e.record.status).toBe("ok");
    const t = (e.state.p4 as P4State).test!;
    expect(t.value!).toBeGreaterThan(0.6);
    expect(t.lower!).toBeLessThan(t.value!);
    expect(t.upper!).toBeGreaterThan(t.value!);
    // 二値の固定ファイルは生存に使えない
    const b = (await run(d, { ...binary, lockMode: "create", testFraction: 0.25 })).state.p4 as P4State;
    expect(lastError(await run({ ...d, lock: { name: "l", text: JSON.stringify(b.split!.lock) } }, { ...surv, lockMode: "use" }))).toMatch(/結果の型/);
  });

  it("method text, fold and prediction CSVs, and warnings about too many features", async () => {
    const d = synth(80, 10);
    const r = await run(d, { ...binary, lockMode: "create", testFraction: 0.25 });
    const st = r.state.p4 as P4State;
    const ja = methodText(r.record, P4_PIPELINE, "ja"), en = methodText(r.record, P4_PIPELINE, "en");
    expect(ja).toMatch(/入れ子の交差検証/);
    expect(ja).toMatch(/信頼区間は付けない/);
    expect(en).toMatch(/nested cross-validation/);
    expect(p4FoldsCsv(st).split(/\r?\n/)[0]).toBe("repeat,fold,n_train,n_test,k,features,score,error");
    expect(p4PredictionsCsv(st).split(/\r?\n/).length).toBe(st.qc!.dev.length + 2);
    // 特徴量の数（40）が症例数（30）より多いときは警告する
    const r2 = new Rng(12);
    const head = ["CaseLabel", ...Array.from({ length: 40 }, (_, j) => `w${j}`)];
    const rows = Array.from({ length: 30 }, (_, i) => [`P${i}`, ...head.slice(1).map(() => r2.next().toFixed(4))].join(","));
    const clin = ["CaseLabel,outcome,time,event", ...Array.from({ length: 30 }, (_, i) => `P${i},${i % 2 ? "resp" : "no"},1,1`)];
    const wide = { features: [sourceOf("w.csv", [head.join(","), ...rows].join("\n") + "\n", null)], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
    const w = await run(wide, { ...binary, lockMode: "none", outerK: 3, innerK: 2, kCandidates: "1" });
    expect(w.record.steps.find((s) => s.id === "p4-qc")!.warnings.join()).toMatch(/特徴量の数（40）が開発集合の症例数（30）より多く/);
  });
});
