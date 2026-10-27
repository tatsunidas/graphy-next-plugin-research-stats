import { describe, expect, it } from "vitest";
import { runSkill, CANCELLED } from "../src/core/engine";
import { R1_PROGNOSIS, type R1State } from "../src/core/skills";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { methodText } from "../src/core/methodText";

/** 既知の効果を入れた合成データ: f_signal だけが生存に効く。f_noise* は無関係。f_gap は一部欠損、f_sparse は全欠損。 */
function synth(n: number, seed: number, opts: { textCov?: boolean; noiseCols?: number } = {}) {
  const r = new Rng(seed);
  const noiseCols = opts.noiseCols ?? 5;
  const head = ["CaseLabel", "f_signal", ...Array.from({ length: noiseCols }, (_, i) => `f_noise${i}`), "f_gap", "f_sparse"];
  const feat: string[] = [head.join(",")], clin: string[] = [`CaseLabel,time,event,age`];
  for (let i = 0; i < n; i++) {
    const sig = r.next() * 2 - 1;
    const t = -Math.log(1 - r.next()) / Math.exp(1.5 * sig) * 10;
    const c = -Math.log(1 - r.next()) * 14;
    const time = Math.max(0.1, Math.round(Math.min(t, c) * 10) / 10), ev = t <= c ? 1 : 0;
    const noise = Array.from({ length: noiseCols }, () => (r.next() * 4).toFixed(3));
    feat.push([`P${i}`, sig.toFixed(4), ...noise, i % 7 === 0 ? "" : r.next().toFixed(3), i < 4 ? r.next().toFixed(3) : "NA"].join(","));
    clin.push(`P${i},${time},${ev},${opts.textCov && i === 0 ? "old" : 40 + Math.floor(r.next() * 40)}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}
const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", timeColumn: "time", eventColumn: "event" };
const opts = (seed = 1) => ({ seed, pluginVersion: "0.2.0", now: () => "2026-10-27T00:00:00Z" });

describe("R1 prognosis skill", () => {
  it("selects the planted signal and not the noise (median split, BH)", async () => {
    const { record, state } = await runSkill(R1_PROGNOSIS, synth(120, 11), { ...base, covariates: ["age"], bootstrapB: 50 }, opts());
    expect(record.status).toBe("ok");
    const st = state.r1 as R1State;
    const sig = st.rows.find((r) => r.feature === "f_signal")!;
    expect(sig.q).toBeLessThan(0.05);
    expect(sig.hrGroup).toBeGreaterThan(1.5); // 値が大きいほどハザードが高い（係数 +1.5）
    expect(sig.hrPerSd).toBeGreaterThan(1);
    const noiseSelected = st.rows.filter((r) => r.feature.startsWith("f_noise") && (r.q as number) < 0.05);
    expect(noiseSelected.length).toBeLessThanOrEqual(1);
    expect(st.model!.selected).toContain("f_signal");
    expect(st.model!.terms.map((t) => t.name)).toContain("age");
    expect(st.model!.cIndexApparent).toBeGreaterThan(0.6);
    expect(st.model!.corrected!.corrected).toBeLessThanOrEqual(st.model!.cIndexApparent!);
    expect(st.model!.addedValue!.p).toBeLessThan(0.05);
  });
  it("does not turn an unanalysable feature into a p value (missing is not 0)", async () => {
    const { state } = await runSkill(R1_PROGNOSIS, synth(60, 3), { ...base, bootstrapB: 0 }, opts());
    const st = state.r1 as R1State;
    const all = st.rows.find((r) => r.feature === "f_sparse")!;
    expect(all.error).toMatch(/解析できません/);
    expect(all.n).toBe(4);
    expect(all.pUsed).toBeNull();
    expect(all.q).toBeNull();
    const gap = st.rows.find((r) => r.feature === "f_gap")!;
    expect(gap.n).toBeLessThan(60); // 欠損の患者は除かれる
    expect(st.comparisons).toBe(st.rows.length - 1); // 比較数に解析できなかった特徴量を数えない
  });
  it("reports the same results for the same seed; the corrected p of the optimal cut-off depends on the seed", async () => {
    const mk = (seed: number) => runSkill(R1_PROGNOSIS, synth(70, 5, { noiseCols: 2 }), { ...base, groupMode: "optimal", permutations: 99, bootstrapB: 0 }, opts(seed));
    const a = await mk(1), b = await mk(1), c = await mk(2);
    expect(b.record).toEqual(a.record);
    const pa = (a.state.r1 as R1State).rows.map((r) => r.correctedP), pc = (c.state.r1 as R1State).rows.map((r) => r.correctedP);
    expect(pc).not.toEqual(pa);
    const sig = (a.state.r1 as R1State).rows[0];
    expect(sig.rawOptimalP).toBeLessThanOrEqual(sig.correctedP!); // 素の最小 p は、補正 p より小さいか等しい
  });
  it("stops with a reason for a text covariate, for no event, and for bad settings", async () => {
    const t = await runSkill(R1_PROGNOSIS, synth(40, 2, { textCov: true }), { ...base, covariates: ["age"] }, opts());
    expect(t.record.status).toBe("failed");
    expect(t.record.steps.at(-1)!.errors.join()).toMatch(/数でない値/);
    const s = synth(40, 2);
    const noEvent = { ...s, clinical: sourceOf("c.csv", s.clinical.table.rows.reduce((acc, r) => acc + `${r[0]},${r[1]},0,${r[3]}\n`, "CaseLabel,time,event,age\n")) };
    const e = await runSkill(R1_PROGNOSIS, noEvent, base, opts());
    expect(e.record.steps.at(-1)!.errors.join()).toMatch(/イベントが 0 件/);
    const bad = await runSkill(R1_PROGNOSIS, synth(40, 2), { ...base, groupMode: "value" }, opts());
    expect(bad.record.steps.at(-1)!.errors.join()).toMatch(/しきい値/);
  });
  it("can be cancelled and says so", async () => {
    let calls = 0;
    const { record } = await runSkill(R1_PROGNOSIS, synth(60, 4), { ...base, groupMode: "optimal", permutations: 200 }, { ...opts(), isCancelled: () => ++calls > 3 });
    expect(record.status).toBe("failed");
    expect(record.steps.at(-1)!.errors.join()).toContain(CANCELLED);
  });
  it("writes the method text with the skill's settings and sources", async () => {
    const { record } = await runSkill(R1_PROGNOSIS, synth(60, 4), { ...base, multiplicity: "none", bootstrapB: 0 }, opts());
    const ja = methodText(record, R1_PROGNOSIS, "ja");
    expect(ja).toContain("multiplicity=none");
    expect(ja).toContain("多重比較の補正をしない");
  });
  it("timing of the heavy mode (informational)", async () => {
    const t0 = Date.now();
    await runSkill(R1_PROGNOSIS, synth(100, 8, { noiseCols: 20 }), { ...base, groupMode: "optimal", permutations: 1000, bootstrapB: 0 }, opts());
    // eslint-disable-next-line no-console
    console.log(`optimal, 23 features, n=100, B=1000: ${Date.now() - t0} ms`);
  }, 120000);
});
