import { describe, expect, it } from "vitest";
import { Rng } from "../src/core/rng";
import { r5Model, readR5Params, R5_DEFAULTS, type R5Params } from "../src/core/r5";

const normal = (rng: Rng) => Math.sqrt(-2 * Math.log(rng.next() || 1e-12)) * Math.cos(2 * Math.PI * rng.next());

/** 特徴量 m 個（列ごとの配列）。signal の列だけがハザードに効く。打ち切りは指数分布。 */
function data(seed: number, n: number, m: number, signal: number[]) {
  const rng = new Rng(seed);
  const values = Array.from({ length: m }, () => Array.from({ length: n }, () => normal(rng)));
  const time: number[] = [], event: number[] = [];
  for (let i = 0; i < n; i++) {
    const eta = signal.reduce((a, b, j) => a + b * values[j][i], 0);
    const t = -Math.log(rng.next() || 1e-12) / Math.exp(eta), c = -Math.log(rng.next() || 1e-12) * 2;
    time.push(Math.min(t, c) + 1e-6 * (i + 1));
    event.push(t <= c ? 1 : 0);
  }
  return { surv: { time, event }, values: values as (number | null)[][], names: values.map((_, j) => `f${j + 1}`) };
}

const params = (o: Partial<R5Params> = {}): R5Params => {
  const p = readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9, r5BoostNu: 0.1, r5BootstrapB: 100 });
  if ("error" in p) throw new Error(p.error);
  return { ...p, ...o };
};

describe("R5 parameters", () => {
  it("needs the values the paper does not give", () => {
    expect(readR5Params({ ...R5_DEFAULTS })).toHaveProperty("error");
    expect(readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9 })).toHaveProperty("error"); // ν
    expect(readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9, r5Screen: "lasso" })).toHaveProperty("error"); // λ の選び方
    expect(readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9, r5Screen: "lasso", r5LassoRule: "1se" })).not.toHaveProperty("error");
    expect(readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9, r5BoostNu: 0.1, r5PIn: 0.2, r5POut: 0.15 })).toHaveProperty("error");
  });
});

describe("R5 bootstrap-corrected C-index", () => {
  it("is near 0.5 on random features, while the apparent C and the leaky variant are optimistic", async () => {
    const proper: number[] = [], leaky: number[] = [], apparent: number[] = [];
    for (const seed of [1, 2, 3]) {
      const d = data(seed, 100, 40, []);
      const a = await r5Model(d.surv, d.values, d.names, params(), new Rng(seed));
      const b = await r5Model(d.surv, d.values, d.names, params(), new Rng(seed), undefined, { selectOnce: true });
      if ("error" in a || "error" in b) throw new Error("failed");
      expect(a.result.selected.length).toBeGreaterThan(0);
      proper.push(a.corrected!.corrected);
      leaky.push(b.corrected!.corrected);
      apparent.push(a.cApparent!);
    }
    const mean = (v: number[]) => v.reduce((x, y) => x + y, 0) / v.length;
    // 偶然の偏り: 補正 C の SD はおよそ 0.03（100 人・イベント約 70）。3 回の平均で ±0.06 に収まることを求める
    expect(Math.abs(mean(proper) - 0.5)).toBeLessThan(0.06);
    expect(mean(apparent)).toBeGreaterThan(mean(proper) + 0.05);
    expect(mean(leaky)).toBeGreaterThan(mean(proper) + 0.03);
  }, 120000);

  it("stays well above 0.5 with real signal and finds the signal features", async () => {
    const d = data(7, 120, 20, [0.9, -0.8]);
    const r = await r5Model(d.surv, d.values, d.names, params(), new Rng(7));
    if ("error" in r) throw new Error(r.error);
    expect(r.result.selected).toEqual(expect.arrayContaining([0, 1]));
    expect(r.corrected!.corrected).toBeGreaterThan(0.65);
    expect(r.terms.find((t) => t.name === "f1")!.hr).toBeGreaterThan(1);
    expect(r.terms.find((t) => t.name === "f2")!.hr).toBeLessThan(1);
    expect(r.schoenfeld).not.toBeNull();
  }, 60000);

  it("is reproducible with the same seed and changes with another seed", async () => {
    const d = data(11, 80, 15, [0.6]);
    const p = params({ bootstrapB: 30 });
    const a = await r5Model(d.surv, d.values, d.names, p, new Rng(5));
    const b = await r5Model(d.surv, d.values, d.names, p, new Rng(5));
    const c = await r5Model(d.surv, d.values, d.names, p, new Rng(6));
    if ("error" in a || "error" in b || "error" in c) throw new Error("failed");
    expect(b.corrected).toEqual(a.corrected);
    expect(c.corrected!.optimism).not.toBe(a.corrected!.optimism);
  }, 60000);
});

describe("R5 procedure details", () => {
  it("removes correlated features, caps the model size and reports both", async () => {
    const d = data(13, 150, 8, [0.8, -0.7, 0.6, 0.5]);
    d.values.push(d.values[0].map((v) => (v as number) * 2 + 0.001));
    d.names.push("f1copy");
    const r = await r5Model(d.surv, d.values, d.names, params({ maxVars: 2, bootstrapB: 0 }), new Rng(1));
    if ("error" in r) throw new Error(r.error);
    expect(r.result.corrRemoved.map((x) => x.feature)).toContain("f1copy");
    expect(r.result.selected.length).toBeLessThanOrEqual(2);
    expect(r.result.steps.some((s) => s.action === "cap")).toBe(true);
    expect(r.notes.join("\n")).toMatch(/最大の数 2/);
  }, 60000);

  it("drops features with missing values and lists them", async () => {
    const d = data(17, 80, 6, [0.8]);
    d.values[3][5] = null;
    const r = await r5Model(d.surv, d.values, d.names, params({ bootstrapB: 0 }), new Rng(1));
    if ("error" in r) throw new Error(r.error);
    expect(r.droppedMissing).toEqual(["f4"]);
    expect(r.used).not.toContain(3);
  });

  it("works with LASSO-Cox screening", async () => {
    const d = data(19, 120, 15, [0.9, -0.8]);
    const p = readR5Params({ ...R5_DEFAULTS, r5Corr: 0.9, r5Screen: "lasso", r5LassoRule: "1se", r5LassoNLambda: 30, r5BootstrapB: 10 });
    if ("error" in p) throw new Error(p.error);
    const r = await r5Model(d.surv, d.values, d.names, p, new Rng(3));
    if ("error" in r) throw new Error(r.error);
    expect(r.result.lasso).not.toBeNull();
    expect(r.result.selected).toEqual(expect.arrayContaining([0, 1]));
    expect(r.corrected!.bootstraps + r.corrected!.failed).toBe(10);
  }, 120000);
});

describe("R5 skill end to end", async () => {
  const { runSkill } = await import("../src/core/engine");
  const { R5_MODEL } = await import("../src/core/skills");
  const { sourceOf } = await import("../src/core/delta");
  const { methodText } = await import("../src/core/methodText");
  const { r5ModelCsv, r5SchoenfeldCsv, r5ScreenCsv, r5StepsCsv } = await import("../src/core/exportR5");
  const { parseCsv } = await import("../src/core/csv");
  const synth = (n: number, seed: number) => {
    const d = data(seed, n, 8, [0.9, -0.7]);
    const head = ["CaseLabel", ...d.names, "f_gap"];
    const feat = [head.join(",")], clin = ["CaseLabel,time,event"];
    for (let i = 0; i < n; i++) {
      feat.push([`P${i}`, ...d.values.map((c) => (c[i] as number).toFixed(5)), i % 9 === 0 ? "" : "1.5"].join(","));
      clin.push(`P${i},${d.surv.time[i].toFixed(6)},${d.surv.event[i]}`);
    }
    return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
  };
  const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", timeColumn: "time", eventColumn: "event", r5Corr: 0.9, r5BoostNu: 0.1, r5BootstrapB: 20 };
  const opts = { seed: 3, pluginVersion: "0.6.0", now: () => "2026-10-27T00:00:00Z" };

  it("runs, records the steps, and writes the method text and CSV files", async () => {
    const { record, state } = await runSkill(R5_MODEL, synth(120, 21), base, opts);
    expect(record.status).toBe("ok");
    const st = state.r5 as import("../src/core/skills").R5State;
    expect(st.model!.result.selected.map((j) => st.names[j])).toEqual(expect.arrayContaining(["f1", "f2"]));
    expect(st.model!.droppedMissing).toEqual(["f_gap"]);
    expect(record.steps.find((s) => s.id === "r5-cohort")!.warnings.join()).toMatch(/mboost/);
    const ja = methodText(record, R5_MODEL, "ja"), en = methodText(record, R5_MODEL, "en");
    expect(ja).toMatch(/Cox ブースティング（線形の基底、100 反復、学習率 0\.100/);
    expect(ja).toMatch(/ブートストラップ（20 回/);
    expect(ja).toMatch(/mboost と同じ値になるかは確かめていない/);
    expect(en).toMatch(/Grambsch-Therneau/);
    const model = parseCsv(r5ModelCsv(st));
    expect(model.rows.map((r) => r[0])).toEqual(expect.arrayContaining(["f1", "f2", "c_index_corrected"]));
    expect(parseCsv(r5StepsCsv(st)).rows.some((r) => r[0] === "missing" && r[1] === "f_gap")).toBe(true);
    expect(parseCsv(r5ScreenCsv(st)).header).toEqual(["feature", "beta_per_sd", "importance", "selected_count"]);
    const sch = parseCsv(r5SchoenfeldCsv(st));
    expect(sch.rows.length).toBe(st.cohort.surv.event.reduce((a, b) => a + b, 0));
    expect(String(sch.rows[0][0])).toMatch(/^P\d+$/);
  }, 120000);

  it("stops with a clear error when a required value is missing", async () => {
    const { record } = await runSkill(R5_MODEL, synth(60, 22), { ...base, r5BoostNu: null }, opts);
    expect(record.status).not.toBe("ok");
    expect(record.steps.flatMap((s) => s.errors).join()).toMatch(/学習率/);
  });
});
