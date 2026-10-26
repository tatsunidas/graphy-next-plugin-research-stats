import { describe, expect, it } from "vitest";
import { runRecipe } from "../src/core/engine";
import { R2_DISCRIMINATION, type R2State } from "../src/core/recipes";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { methodText } from "../src/core/methodText";
import { r2CurvesCsv, r2ResultsCsv, treeCsv } from "../src/core/exportR2";

/** f_a は強く、f_b は中くらいに結果ラベルと関係する。f_noise* は無関係。f_gap は一部欠損、f_sparse は 4 人にしか値が無い。 */
function synth(n: number, seed: number, labels: [string, string] = ["resp", "no"], extra?: (i: number) => string) {
  const r = new Rng(seed);
  const head = ["CaseLabel", "f_a", "f_b", "f_noise0", "f_noise1", "f_noise2", "f_gap", "f_sparse"];
  const feat = [head.join(",")], clin = ["CaseLabel,outcome,age"];
  for (let i = 0; i < n; i++) {
    const g = r.next() < 0.45 ? 1 : 0;
    const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
    const a = norm() + 1.6 * g, b = norm() + 0.8 * g;
    feat.push([`P${i}`, a.toFixed(4), b.toFixed(4), norm().toFixed(4), norm().toFixed(4), norm().toFixed(4), i % 9 === 0 ? "" : norm().toFixed(4), i < 4 ? "1.5" : "NA"].join(","));
    clin.push(`P${i},${extra?.(i) ?? labels[g === 1 ? 0 : 1]},${40 + (i % 30)}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}
const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", outcomeColumn: "outcome", positiveLabel: "resp" };
const opts = (seed = 1) => ({ seed, pluginVersion: "0.3.0", now: () => "2026-10-27T00:00:00Z" });

describe("R2 discrimination recipe", () => {
  it("finds the planted features, combines them and validates by LOO", async () => {
    const { record, state } = await runRecipe(R2_DISCRIMINATION, synth(120, 21), { ...base, tree: true, mcIterations: 200 }, opts());
    expect(record.status).toBe("ok");
    const st = state.r2 as R2State;
    const a = st.rows.find((r) => r.feature === "f_a")!, noise = st.rows.filter((r) => r.feature.startsWith("f_noise"));
    expect(a.auc).toBeGreaterThan(0.8);
    expect(a.direction).toBe("higher");
    expect(a.q).toBeLessThan(0.05);
    expect(noise.filter((r) => (r.q as number) < 0.05).length).toBe(0);
    expect(a.perf!.confusion.tp + a.perf!.confusion.fn).toBe(a.positives); // 混同行列の陽性数が一致
    expect(st.combo!.features).toEqual(["f_a", "f_b"]);
    expect(st.combo!.loo!.auc).toBeGreaterThan(0.7);
    expect(st.combo!.loo!.auc).toBeLessThan(st.combo!.apparent!.auc + 0.02);
    expect(st.combo!.selectionCounts[0].feature).toBe("f_a");
    expect(st.tree!.perf.accuracy.value).toBeGreaterThan(0.7);
    expect(st.tree!.mc.accuracy.lower).toBeLessThan(st.tree!.perf.accuracy.value!);
    expect(r2ResultsCsv(st.rows).split("\r\n")[0]).toContain("auc_lower");
    expect(r2CurvesCsv(st.combo!)).toContain("dca_model");
    expect(treeCsv(st.tree!.root, st.context.names)).toContain("f_a");
  });
  it("does not turn an unanalysable feature into an AUC (missing is not 0)", async () => {
    const { state } = await runRecipe(R2_DISCRIMINATION, synth(80, 5), base, opts());
    const st = state.r2 as R2State;
    const sp = st.rows.find((r) => r.feature === "f_sparse")!;
    expect(sp.error).toMatch(/解析できません/);
    expect(sp.auc).toBeNull();
    expect(sp.q).toBeNull();
    expect(st.rows.find((r) => r.feature === "f_gap")!.n).toBeLessThan(80);
    expect(st.comparisons).toBe(st.rows.length - 1);
  });
  it("is reproducible for the same seed (tree and Monte Carlo included)", async () => {
    const run = (seed: number) => runRecipe(R2_DISCRIMINATION, synth(90, 8), { ...base, tree: true, mcIterations: 300 }, opts(seed));
    const a = await run(3), b = await run(3), c = await run(4);
    expect(b.record).toEqual(a.record);
    expect((b.state.r2 as R2State).tree!.mc).toEqual((a.state.r2 as R2State).tree!.mc);
    expect((c.state.r2 as R2State).tree!.mc.means).not.toEqual((a.state.r2 as R2State).tree!.mc.means);
  });
  it("stops with a reason for no positive label, an unknown label, three labels, and a tiny group (negative)", async () => {
    const err = async (inputs: ReturnType<typeof synth>, params: Record<string, unknown>) => (await runRecipe(R2_DISCRIMINATION, inputs, params as never, opts())).record.steps.at(-1)!.errors.join();
    expect(await err(synth(60, 1), { ...base, positiveLabel: null })).toMatch(/陽性とするラベルを選んでください/);
    expect(await err(synth(60, 1), { ...base, positiveLabel: "zzz" })).toMatch(/にありません/);
    expect(await err(synth(60, 1, ["resp", "no"], (i) => (i % 7 === 0 ? "mid" : i % 2 ? "resp" : "no")), base)).toMatch(/2 種類にしてください/);
    expect(await err(synth(60, 1, ["resp", "no"], (i) => (i < 3 ? "resp" : "no")), base)).toMatch(/各群 5 人以上/);
  });
  it("makes no combination when fewer than two features meet the criterion, and says why", async () => {
    const { record, state } = await runRecipe(R2_DISCRIMINATION, synth(40, 2), { ...base, alpha: 1e-6 }, opts());
    expect(record.steps.find((s) => s.id === "combination")!.status).toBe("skipped");
    expect((state.r2 as R2State).combo!.notes.join()).toMatch(/組み合わせは作りません/);
  });
  it("writes the method text with the labels, the criterion and the LOO", async () => {
    const { record } = await runRecipe(R2_DISCRIMINATION, synth(100, 21), { ...base, tree: true, mcIterations: 100 }, opts());
    const ja = methodText(record, R2_DISCRIMINATION, "ja"), en = methodText(record, R2_DISCRIMINATION, "en");
    expect(ja).toContain("Clopper–Pearson");
    expect(ja).toContain("leave-one-out");
    expect(ja).toContain("検証はしていない");
    expect(en).toContain("Monte Carlo");
  });
});
