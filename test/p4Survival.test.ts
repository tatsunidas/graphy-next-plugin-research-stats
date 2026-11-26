import { describe, expect, it } from "vitest";
import { runSkill } from "../src/core/engine";
import { P4_PIPELINE, type P4State } from "../src/core/skills";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { methodText } from "../src/core/methodText";

/** f_a・f_b が生存に効く（f_a が強い）。f_n* は無関係。effect = 0 なら、特徴量と生存は無関係（入れ替えと同じ）。 */
function synth(n: number, seed: number, effect = 0.9) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const feat = ["CaseLabel,f_a,f_b,f_n0,f_n1,f_n2,f_n3"], clin = ["CaseLabel,time,event"];
  for (let i = 0; i < n; i++) {
    const a = norm(), b = norm();
    feat.push([`P${i}`, a.toFixed(4), b.toFixed(4), norm().toFixed(4), norm().toFixed(4), norm().toFixed(4), norm().toFixed(4)].join(","));
    const t = -Math.log(1 - r.next()) / Math.exp(effect * a + 0.5 * effect * b) * 10;
    const c = -Math.log(1 - r.next()) * 14;
    clin.push(`P${i},${(Math.min(t, c) + 0.001 * (i + 1)).toFixed(4)},${t <= c ? 1 : 0}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}
const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", timeColumn: "time", eventColumn: "event", outcomeKind: "survival", stratifySurvival: true, lockMode: "none", outerK: 4, innerK: 3, repeats: 1, correlationThreshold: null, lassoRule: "min", lassoNLambda: 30 };
const MODELS: Record<string, Record<string, unknown>> = {
  lasso: { model: "lasso" },
  enet: { model: "enet", enetAlpha: 0.5 },
  ridge: { model: "ridge" },
  coxboost: { model: "coxboost", coxIterations: "20, 60", coxNu: 0.1 },
  rsf: { model: "rsf", rsfTrees: 25, rsfMtry: "sqrt, 3", rsfMinLeaf: "3, 5" },
};
const opts = { seed: 1, pluginVersion: "t", now: () => "2026-11-26T00:00:00Z" };
const go = (d: ReturnType<typeof synth>, params: Record<string, unknown>) => runSkill(P4_PIPELINE, d, { ...base, ...params } as never, opts);
const err = (r: Awaited<ReturnType<typeof go>>) => r.record.steps.flatMap((s) => s.errors).join(" / ");

// 同時刻のイベント vs 打ち切りも数える（lifelines と同じ規則）の総当たり
function brute(time: number[], event: number[], risk: number[]): number | null {
  let conc = 0, comp = 0;
  for (let i = 0; i < time.length; i++) {
    if (event[i] !== 1) continue;
    for (let j = 0; j < time.length; j++) {
      if (j === i || !(time[j] > time[i] || (time[j] === time[i] && event[j] === 0))) continue;
      comp++; conc += risk[i] > risk[j] ? 1 : risk[i] === risk[j] ? 0.5 : 0;
    }
  }
  return comp ? conc / comp : null;
}

describe("P4 survival models: Elastic Net, Ridge, Cox boosting", () => {
  for (const [name, m] of Object.entries(MODELS)) {
    it(`${name}: runs, finds the signal, and every outer-fold C-index equals a brute-force recount from the predictions`, async () => {
      const { record, state } = await go(synth(120, 7), m);
      expect(err({ record, state } as never), "errors").toBe("");
      expect(record.status).toBe("ok");
      const st = state.p4 as P4State;
      const cv = st.cv!;
      expect(cv.summary.mean!).toBeGreaterThan(0.62);
      const surv = st.cohort.task.kind === "survival" ? st.cohort.task.surv : null;
      expect(surv).not.toBeNull();
      const dev = st.qc!.dev;
      cv.folds.forEach((f) => {
        const pos = dev.map((_, p) => p).filter((p) => cv.outerAssign[f.repeat][p] === f.fold);
        const rows = pos.map((p) => dev[p]);
        const risk = pos.map((p) => cv.pred[f.repeat][p] as number);
        expect(brute(rows.map((i) => surv!.time[i]), rows.map((i) => surv!.event[i]), risk) as number).toBeCloseTo(f.score as number, 12);
      });
      // 最終モデルが作れ、方法の文にモデルの説明が入る
      expect(st.final!.error).toBeNull();
      const ja = methodText(record, P4_PIPELINE, "ja"), en = methodText(record, P4_PIPELINE, "en");
      const key = { lasso: "LASSO-Cox", enet: "Elastic Net-Cox", ridge: "Ridge-Cox", coxboost: "Cox ブースティング", rsf: "ランダムサバイバルフォレスト" }[name]!;
      expect(ja).toContain(key);
      expect(en).toContain({ lasso: "LASSO-Cox", enet: "elastic-net", ridge: "ridge-penalised", coxboost: "Cox boosting", rsf: "random survival forest" }[name]!);
    }, 120000);
    it(`${name}: negative control — with no relation between features and survival the mean C-index is near 0.5`, async () => {
      const cs: number[] = [];
      for (const seed of [11, 12, 13]) {
        const { record, state } = await go(synth(120, seed, 0), m);
        expect(record.status).toBe("ok");
        const mean = (state.p4 as P4State).cv!.summary.mean;
        if (mean !== null) cs.push(mean);
      }
      expect(cs.length).toBeGreaterThan(0);
      expect(Math.abs(cs.reduce((a, b) => a + b, 0) / cs.length - 0.5)).toBeLessThan(0.1);
    }, 180000);
  }
  it("Ridge keeps every feature (k = 6 in each fold); Elastic Net and LASSO may drop some", async () => {
    const d = synth(120, 7);
    const ridge = (await go(d, MODELS.ridge)).state.p4 as P4State;
    expect(ridge.cv!.folds.every((f) => f.k === 6)).toBe(true);
    const lasso = (await go(d, MODELS.lasso)).state.p4 as P4State;
    expect(lasso.cv!.folds.some((f) => (f.k ?? 6) < 6)).toBe(true);
  }, 120000);
  it("Cox boosting chooses the number of iterations from the candidates by inner cross-validation, per fold", async () => {
    const st = (await go(synth(120, 7), MODELS.coxboost)).state.p4 as P4State;
    for (const f of st.cv!.folds) {
      expect(f.params).not.toBeNull();
      expect([20, 60]).toContain(Number((f.params as Record<string, number>).iterations));
      expect(f.innerScores).toHaveLength(2);
    }
  }, 120000);
  it("RSF chooses (mtry, min leaf) from the grid per fold; the grid has 4 points; refuses an emptied candidate list", async () => {
    const st = (await go(synth(120, 7), MODELS.rsf)).state.p4 as P4State;
    for (const f of st.cv!.folds) {
      expect(f.params).not.toBeNull();
      expect(f.innerScores).toHaveLength(4);
      expect([3, 5]).toContain(Number((f.params as Record<string, number>).minLeaf));
    }
    const d = synth(60, 3);
    expect(err(await go(d, { model: "rsf", rsfTrees: null }))).toMatch(/木の数/);
    expect(err(await go(d, { model: "rsf", rsfTrees: 10, rsfMtry: "sqrt", rsfMinLeaf: null }))).toMatch(/葉の最小/);
    expect(err(await go(d, { model: "rsf", rsfTrees: 10, rsfMtry: "sqrt", rsfMinLeaf: "3, 3" }))).toMatch(/重複/);
    // mtry が特徴量の数を超えると、どの候補も評価できず、どの折も評価不能になり、理由が折ごとに残る（0 や 0.5 にしない）
    const big = (await go(d, { model: "rsf", rsfTrees: 10, rsfMtry: "99", rsfMinLeaf: "3" })).state.p4 as P4State | undefined;
    expect(big?.cv?.folds.every((f) => f.score === null && (f.error ?? "") !== "")).toBe(true);
    expect(err(await go(d, { ...MODELS.rsf, outcomeKind: "binary", positiveLabel: "x" }))).toMatch(/二値の結果では使えません/);
  }, 120000);
  it("settings: Elastic Net needs alpha strictly between 0 and 1; Cox boosting refuses emptied iteration candidates; binary outcomes refuse these models", async () => {
    const d = synth(60, 3);
    expect(err(await go(d, { model: "enet", enetAlpha: null }))).toMatch(/L1 の比率/);
    expect(err(await go(d, { model: "enet", enetAlpha: 1 }))).toMatch(/L1 の比率/);
    expect(err(await go(d, { model: "enet", enetAlpha: 0 }))).toMatch(/L1 の比率/);
    expect(err(await go(d, { model: "coxboost", coxIterations: null }))).toMatch(/反復回数/);
    expect(err(await go(d, { model: "coxboost", coxIterations: "10, 10" }))).toMatch(/重複/);
    expect(err(await go(d, { model: "coxboost", coxIterations: "10", coxNu: 0 }))).toMatch(/学習率/);
    for (const model of ["enet", "ridge", "coxboost"]) {
      const r = await go(d, { ...MODELS[model], model, outcomeKind: "binary", positiveLabel: "x" });
      expect(err(r)).toMatch(/二値の結果では使えません/);
    }
  });
});
