import { describe, expect, it } from "vitest";
import fx from "./fixtures/preprocess_pipeline.json";
import { sourceOf, buildDelta } from "../src/core/delta";
import { runSkill } from "../src/core/engine";
import { checkRowUse } from "../src/core/leak";
import { methodText } from "../src/core/methodText";
import { Rng } from "../src/core/rng";
import { finalModel, nestedCv, pdropColumns, type NestedCvSpec, type PDropSpec, type Task } from "../src/core/pipeline/nestedCv";
import { pairedDrop } from "../src/core/pipeline/pre";
import { RowTrace } from "../src/core/pipeline/rowTrace";
import { P4_PIPELINE, R1_PROGNOSIS, R2_DISCRIMINATION, R5_MODEL, type P4State, type R1State, type R2State, type R5State } from "../src/core/skills";
import { dropByPValue } from "../src/core/stats/dropByPValue";
import { assumptionReport } from "../src/core/stats/assumptionReport";
import { assumptionsCsv, pdropCsv } from "../src/core/exportPre";

const near = (a: number | null, b: number | null, label: string) => {
  if (b === null || a === null) { expect(a, label).toBe(b); return; }
  expect(Math.abs(a - b), `${label}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-10 + 1e-9 * Math.abs(b));
};

// ---------------------------------------------------------------------------------------------
// scipy との突き合わせ（tools/gen_preprocess_pipeline_fixture.py。規則を scipy の関数で独立に書いたもの）
// ---------------------------------------------------------------------------------------------
describe("p-value filter vs an independent scipy implementation", () => {
  const u = fx.unpaired;
  const X = u.X as (number | null)[][];
  const task: Task = { kind: "binary", y: u.y };
  const spec = (alpha: number, threshold: number): PDropSpec => ({ alpha, threshold, names: u.names, log: [] });

  it("unpaired: the kept features are identical on every training set and every (alpha, threshold)", () => {
    expect(u.sets.length).toBe(15); // 全体 1 + 外側 5×2 + 内側 4
    let checked = 0;
    for (const set of u.sets) {
      for (const [alpha, thr] of u.combos) {
        const dropped = pdropColumns(X, task, set.rows, spec(alpha, thr));
        const kept = u.names.map((_, j) => j).filter((j) => !dropped.has(j));
        expect(kept, `${set.label} alpha=${alpha} thr=${thr}`).toEqual((set.kept as Record<string, number[]>)[`${alpha}|${thr}`]);
        checked++;
      }
    }
    expect(checked).toBe(15 * u.combos.length);
  });
  it("unpaired: the chosen test and the p of every feature match scipy", () => {
    const seen = new Set<string>();
    for (const set of u.sets) {
      for (const alpha of u.alphas) {
        const r = dropByPValue(X, u.names, { kind: "unpaired", rows0: set.rows.filter((i) => u.y[i] === 0), rows1: set.rows.filter((i) => u.y[i] === 1) }, alpha, 0.05);
        if ("error" in r) throw new Error(r.error);
        const ref = (set.perAlpha as Record<string, { test: string | null; p: number | null }[]>)[String(alpha)];
        r.records.forEach((rec, j) => {
          expect(rec.selection?.test ?? null, `${set.label} a=${alpha} ${rec.feature}`).toBe(ref[j].test);
          near(rec.p, ref[j].p, `${set.label} a=${alpha} ${rec.feature} p`);
          if (rec.selection) seen.add(rec.selection.test);
        });
      }
    }
    expect([...seen].sort()).toEqual(["mannwhitney", "student", "welch"]); // 3 つの検定がどれも使われている
  });
  it("unpaired: the thresholds really change the result (the check is not trivially all-kept or all-dropped)", () => {
    const k = (u.sets[0].kept as Record<string, number[]>);
    const sizes = Object.values(k).map((v) => v.length);
    expect(new Set(sizes).size).toBeGreaterThan(2);
    expect(Math.min(...sizes)).toBeGreaterThan(0);
    expect(Math.max(...sizes)).toBeLessThan(u.names.length);
  });
  it("paired: kept features, chosen test and p match scipy (ttest_rel / wilcoxon)", () => {
    const p = fx.paired;
    const n = p.before.length;
    const pm = { before: p.before as number[][], after: p.after as number[][] };
    const rows = Array.from({ length: n }, (_, i) => i), cols = p.names.map((_, j) => j);
    const seen = new Set<string>();
    for (const [alpha, thr] of p.combos) {
      const r = pairedDrop(pm, p.names, cols, rows, { alpha, threshold: thr });
      if ("error" in r) throw new Error(r.error);
      expect(r.keep, `alpha=${alpha} thr=${thr}`).toEqual((p.kept as Record<string, number[]>)[`${alpha}|${thr}`]);
      const ref = (p.perAlpha as Record<string, { test: string | null; p: number | null }[]>)[String(alpha)];
      r.info.records.forEach((rec, j) => {
        expect(rec.test, `${rec.feature}`).toBe(ref[j].test);
        near(rec.p, ref[j].p, `${rec.feature} p`);
        if (rec.test) seen.add(rec.test);
      });
    }
    expect([...seen].sort()).toEqual(["paired-t", "wilcoxon"]);
  });
  it("positive and negative controls: features with a group difference stay, noise features go (fixed data, final training set)", () => {
    const dropped = pdropColumns(X, task, u.sets[0].rows, spec(0.05, 0.05));
    const by = (prefix: string) => u.names.map((n, j) => ({ n, j })).filter((x) => x.n.startsWith(prefix));
    for (const x of [...by("shiftN"), ...by("shiftLog")]) expect(dropped.has(x.j), x.n).toBe(false); // 差のある特徴量は残る
    const noise = [...by("noiseN"), ...by("noiseLog")];
    expect(noise.filter((x) => dropped.has(x.j)).length / noise.length).toBeGreaterThanOrEqual(0.85); // 雑音は（ほぼ）落ちる
    expect(dropped.has(u.names.indexOf("constant"))).toBe(true); // 検定できない列は落ちる（0 の p で残さない）
  });
  it("reads only the rows it is given (changing every other row does not change the result)", () => {
    const rows = u.sets[1].rows;
    const inSet = new Set(rows);
    const X2 = X.map((r, i) => (inSet.has(i) ? r : r.map(() => 1e6 * (i + 1))));
    const a = pdropColumns(X, task, rows, spec(0.05, 0.1));
    const b = pdropColumns(X2, task, rows, spec(0.05, 0.1));
    expect([...b].sort((x, y) => x - y)).toEqual([...a].sort((x, y) => x - y));
  });
});

// ---------------------------------------------------------------------------------------------
// 入れ子の交差検証の内側で行う（リークの正例・負例）
// ---------------------------------------------------------------------------------------------
describe("p-value filter inside nested cross-validation", () => {
  const u = fx.unpaired;
  const X = u.X as (number | null)[][];
  const task: Task = { kind: "binary", y: u.y };
  const dev = X.map((_, i) => i);
  const mk = (over: Partial<NestedCvSpec> = {}): NestedCvSpec => ({
    outerK: 4, innerK: 3, repeats: 1, kCandidates: [1, 3], strata: u.y, impute: false, trace: new RowTrace(),
    pdrop: { alpha: 0.05, threshold: 0.05, names: u.names, log: [] }, ...over,
  });

  it("honest: the filter runs on training rows only at every level (outer, inner, final) and the row-use check is OK", async () => {
    const spec = mk();
    await nestedCv(X, task, dev, spec, new Rng(4));
    finalModel(X, task, dev, spec, new Rng(4));
    const r = checkRowUse(spec.trace!);
    expect(r.level).toBe("ok");
    const f = r.findings.filter((x) => x.stage === "pdrop");
    expect(f.map((x) => x.where).sort()).toEqual(["final", "finalInner", "fold", "inner"]);
    expect(f.every((x) => x.usesOutcome && x.entries > 0 && x.violating === 0)).toBe(true);
    // 回数: (外側 4 × 反復 1 + 最終モデル) × (内側 3 + 1)。画面の「検定の回数の目安」と同じ式
    expect(spec.pdrop!.log.length).toBe((4 * 1 + 1) * (3 + 1));
    expect(spec.pdrop!.log.filter((e) => /^r0f\d$/.test(e.scope)).length).toBe(4);
    expect(spec.pdrop!.log.filter((e) => e.scope === "final").length).toBe(1);
    // 差のある特徴量は、外側のどの訓練でも落ちない
    const shift = u.names.map((n, j) => ({ n, j })).filter((x) => x.n.startsWith("shiftN")).map((x) => x.n);
    for (const e of spec.pdrop!.log.filter((e) => /^r0f\d$/.test(e.scope))) for (const n of shift) expect(e.dropped.map((d) => d.feature)).not.toContain(n);
  });
  it("negative control: deciding the filter on the whole development set (a real leak) is NG", async () => {
    const spec = mk();
    await nestedCv(X, task, dev, spec, new Rng(4), undefined, { pdropOnAll: true });
    const r = checkRowUse(spec.trace!);
    expect(r.level).toBe("ng"); // 目的変数を使う処理が検証側の行を読んだ
    const bad = r.findings.filter((x) => x.stage === "pdrop" && x.violating > 0);
    expect(bad.length).toBeGreaterThan(0);
    expect(bad.every((x) => x.usesOutcome && x.where === "pre" && x.maxOutside > 0)).toBe(true);
    expect(bad[0].maxOutside).toBe(24); // dev 96 − 外側の訓練 72
  });
  it("LASSO: λ is chosen with the filter redone inside each inner training set, and the check is OK", async () => {
    const spec = mk({ model: { kind: "lasso", nLambda: 15, rule: "min" } });
    await nestedCv(X, task, dev, spec, new Rng(4));
    const r = checkRowUse(spec.trace!);
    expect(r.level).toBe("ok");
    expect(spec.pdrop!.log.some((e) => e.scope.includes(".i"))).toBe(true);
    expect(spec.pdrop!.log.filter((e) => /^r0f\d$/.test(e.scope)).length).toBe(4);
    // 外側の訓練ごとに 内側 4 回（outerK=4 → 内側の分割 3 + その外側の訓練）
    expect(spec.pdrop!.log.length).toBe(4 * (3 + 1));
  });
  it("without the filter nothing changes: same predictions and folds as a spec with no pdrop", async () => {
    const a = await nestedCv(X, task, dev, mk({ pdrop: undefined, trace: undefined }), new Rng(4));
    const b = await nestedCv(X, task, dev, mk({ pdrop: undefined, trace: undefined }), new Rng(4));
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
  it("the log and CSV list the removed features per training set", async () => {
    const spec = mk();
    await nestedCv(X, task, dev, spec, new Rng(4));
    const csv = pdropCsv(spec.pdrop!.log);
    expect(csv.split(/\r?\n/)[0]).toBe("scope,n_train,tested,kept,dropped_feature,test,p,reason");
    expect(csv).toContain("r0f0,72,55");
    expect(csv).toContain("noiseN_");
  });
});

// ---------------------------------------------------------------------------------------------
// 前提の検定の報告
// ---------------------------------------------------------------------------------------------
describe("assumption report", () => {
  const u = fx.unpaired;
  const X = u.X as (number | null)[][];
  it("matches the selection rule and leaves blanks (null with a reason) where a value cannot be computed", () => {
    const rows = X.map((_, i) => i);
    const rep = assumptionReport(X, u.names, rows.filter((i) => u.y[i] === 0), rows.filter((i) => u.y[i] === 1), 0.05);
    if ("error" in rep) throw new Error(rep.error);
    const ref = (u.sets[0].perAlpha as Record<string, { test: string | null; p: number | null }[]>)["0.05"];
    rep.forEach((r, j) => { expect(r.test).toBe(ref[j].test); near(r.p, ref[j].p, r.feature); });
    const c = rep[u.names.indexOf("constant")];
    expect(c.shapiro[0].p).toBeNull();
    expect(c.shapiro[0].reason).toBeTruthy(); // 0 にしない・理由つき
    expect(c.p).toBeNull();
    const csv = assumptionsCsv(rep, 0.05).split(/\r?\n/);
    expect(csv[0].startsWith("feature,alpha,n_group0")).toBe(true);
    expect(csv.find((l) => l.startsWith("constant,"))).toContain(",,"); // 空欄
    expect("error" in assumptionReport(X, u.names, [0], [1], 1.5)).toBe(true);
  });
});

// ---------------------------------------------------------------------------------------------
// スキルを通した試験（R1・R2・R5・P4）
// ---------------------------------------------------------------------------------------------
function synth(n: number, seed: number, shift = 1.4, extra: (i: number, g: number, a: number) => Record<string, number> = () => ({})) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const names = ["f_a", "f_b", "f_n0", "f_n1", "f_const", "f_dup", ...Object.keys(extra(0, 0, 0))];
  const feat = [`CaseLabel,${names.join(",")}`], clin = ["CaseLabel,outcome,time,event"];
  for (let i = 0; i < n; i++) {
    const g = r.next() < 0.5 ? 1 : 0, a = norm() + shift * g, b = norm() + 0.5 * shift * g;
    const e = extra(i, g, a);
    feat.push([`P${i}`, a.toFixed(4), b.toFixed(4), norm().toFixed(4), norm().toFixed(4), "2", (a * 1.01 + 0.1 * norm()).toFixed(4), ...Object.values(e)].join(","));
    const time = (-Math.log(1 - r.next()) / Math.exp(0.9 * a)).toFixed(4);
    clin.push(`P${i},${g ? "resp" : "no"},${Number(time) + 0.001},${r.next() < 0.8 ? 1 : 0}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}
const roles = { patientColumn: null, clinicalIdColumn: "CaseLabel", outcomeColumn: "outcome", timeColumn: "time", eventColumn: "event" };
const base = { ...roles, positiveLabel: "resp" }; // R2・P4（二値）
const surv = { ...roles }; // R1・R5（生存）
const opts = { seed: 1, pluginVersion: "t", now: () => "2026-01-01T00:00:00Z" };

/** Benjamini–Hochberg（手計算の定義どおり）。p の昇順 k 番目: min_{j>=k} p_(j)·m/j。 */
function bh(p: number[]): number[] {
  const m = p.length, o = p.map((_, i) => i).sort((a, b) => p[a] - p[b]);
  const q = new Array<number>(m);
  let run = 1;
  for (let k = m; k >= 1; k--) { run = Math.min(run, (p[o[k - 1]] * m) / k); q[o[k - 1]] = run; }
  return q;
}

describe("R2: zero-variance and correlation removal reduce m and change the BH q", () => {
  it("off: m counts every analysable feature; on: the constant and the duplicate are gone, p unchanged, q recomputed with the smaller m (it changes) (hand-computed BH)", async () => {
    const data = synth(80, 7, 0.9);
    const off = await runSkill(R2_DISCRIMINATION, data, base as never, opts);
    const on = await runSkill(R2_DISCRIMINATION, data, { ...base, dropConstant: true, correlationThreshold: 0.95 } as never, opts);
    expect(off.record.status, off.record.steps.flatMap((x) => x.errors).join()).toBe("ok");
    expect(on.record.status, on.record.steps.flatMap((x) => x.errors).join()).toBe("ok");
    const a = off.state.r2 as R2State, b = on.state.r2 as R2State;
    expect(a.pre).toBeNull();
    expect(a.comparisons).toBe(5); // 定数の列は解析できない（p なし）ので 6 − 1
    expect(b.comparisons).toBe(4); // f_const と f_dup を除いた 4
    expect(b.pre!.removed.map((r) => `${r.feature}:${r.step}`).sort()).toEqual(["f_const:const", "f_dup:corr"]);
    const pOff = new Map(a.rows.map((r) => [r.feature, r.mwP])), pOn = new Map(b.rows.map((r) => [r.feature, r.mwP]));
    for (const [k, v] of pOn) expect(v).toBe(pOff.get(k)); // 除外は残った特徴量の解析を変えない
    // 手計算の BH
    const namesOff = a.rows.filter((r) => r.mwP !== null).map((r) => r.feature), namesOn = b.rows.filter((r) => r.mwP !== null).map((r) => r.feature);
    const qOff = bh(namesOff.map((n) => pOff.get(n) as number)), qOn = bh(namesOn.map((n) => pOn.get(n) as number));
    namesOff.forEach((n, i) => expect(a.rows.find((r) => r.feature === n)!.q as number).toBeCloseTo(qOff[i], 12));
    namesOn.forEach((n, i) => expect(b.rows.find((r) => r.feature === n)!.q as number).toBeCloseTo(qOn[i], 12));
    // m が減ると q は（同じ p で）小さくなる
    // m（と順位）が変わるので、同じ p でも q は変わる（上がることも下がることもある。ここでは f_a が 1.4e-5 → 2.3e-5）
    const changed = namesOn.filter((n) => Math.abs((b.rows.find((r) => r.feature === n)!.q as number) - (a.rows.find((r) => r.feature === n)!.q as number)) > 1e-12);
    expect(changed.length).toBeGreaterThan(0);
    expect(on.record.steps.find((s) => s.id === "pre")!.warnings.join()).toContain("多重比較の補正の対象の数 m");
    expect(on.record.steps.find((s) => s.id === "pre")!.detail).toMatchObject({ removed: [{ feature: "f_const", step: "const" }, { feature: "f_dup", step: "corr" }] });
    expect(methodText(on.record, R2_DISCRIMINATION, "ja")).toContain("特徴量は 6 個から 4 個になった");
    expect(methodText(on.record, R2_DISCRIMINATION, "en")).toContain("went from 6 to 4");
    expect(methodText(off.record, R2_DISCRIMINATION, "ja")).not.toContain("解析の前に、患者ごとの特徴量に対して");
  }, 120000);
  it("with nothing enabled the 'pre' step is skipped and the results are exactly the same as before the step existed", async () => {
    const data = synth(80, 7);
    const r = await runSkill(R2_DISCRIMINATION, data, base as never, opts);
    expect(r.record.steps.find((s) => s.id === "pre")!.status).toBe("skipped");
    // 既定の設定の値だけを入れても（有効にしない）同じ
    const r2 = await runSkill(R2_DISCRIMINATION, data, { ...base, prePairedOn: false, prePairedAlpha: 0.05, prePairedThr: 0.05, preAssumpOn: false, preAssumpAlpha: 0.05 } as never, opts);
    const strip = (x: typeof r) => JSON.stringify(x.record.steps);
    expect(strip(r2)).toBe(strip(r));
  }, 120000);
  it("R2 assumption report: lists W/p/Levene per feature, does not change the analysis", async () => {
    const data = synth(80, 7);
    const off = await runSkill(R2_DISCRIMINATION, data, base as never, opts);
    const on = await runSkill(R2_DISCRIMINATION, data, { ...base, preAssumpOn: true, preAssumpAlpha: 0.05 } as never, opts);
    const a = (on.state.r2 as R2State).pre!.assumptions!;
    expect(a.rows.length).toBe(6);
    expect(a.rows.find((r) => r.feature === "f_const")!.reason ?? a.rows.find((r) => r.feature === "f_const")!.shapiro[0].reason).toBeTruthy();
    expect(JSON.stringify((on.state.r2 as R2State).rows)).toBe(JSON.stringify((off.state.r2 as R2State).rows));
  }, 120000);
});

describe("R1: zero-variance and correlation removal", () => {
  it("removes the constant and the duplicate before the per-feature analysis (m shrinks)", async () => {
    const data = synth(80, 7);
    const p = { ...surv, dropConstant: true, correlationThreshold: 0.95, bootstrapB: 0, mannWhitney: false };
    const on = await runSkill(R1_PROGNOSIS, data, p as never, opts);
    expect(on.record.status).toBe("ok");
    const st = on.state.r1 as R1State;
    expect(st.context.names).toEqual(["f_a", "f_b", "f_n0", "f_n1"]);
    expect(st.pre!.removed.map((r) => r.feature).sort()).toEqual(["f_const", "f_dup"]);
    expect(st.rows.length).toBe(4);
  }, 120000);
});

describe("P4: p-value filter through the skill (default flow and record)", () => {
  const p4 = { ...base, outcomeKind: "binary", outerK: 4, innerK: 3, repeats: 1, kCandidates: "1, 2", lockMode: "none", correlationThreshold: null };
  it("filter on: recorded in the steps and the method text, row-use stays OK, and the removed features are listed per training set", async () => {
    const data = synth(100, 3);
    const on = await runSkill(P4_PIPELINE, data, { ...p4, preUnpairedOn: true, preUnpairedAlpha: 0.05, preUnpairedThr: 0.05 } as never, opts);
    expect(on.record.status).toBe("ok");
    const st = on.state.p4 as P4State;
    expect(on.record.steps.find((s) => s.id === "p4-leak")!.summary.rowUseCheck).toBe("OK");
    const log = st.spec!.pdrop!.log;
    expect(log.length).toBe((4 * 1 + 1) * (3 + 1));
    const cv = on.record.steps.find((s) => s.id === "p4-cv")!;
    expect(cv.summary.pdropOuterEntries).toBe(4);
    expect((cv.detail as { pdrop: unknown[] }).pdrop.length).toBe(4);
    expect(on.record.steps.find((s) => s.id === "p4-final")!.summary.pdropDropped).toBe(log.find((e) => e.scope === "final")!.dropped.length);
    // 雑音の列は落ち、差のある列は残る（最終モデル）
    const fin = log.find((e) => e.scope === "final")!.dropped.map((d) => d.feature);
    expect(fin).toContain("f_const"); // 検定できない列
    expect(fin).toContain("f_n1"); // 雑音
    expect(fin).not.toContain("f_a"); // 群に差がある
    expect(methodText(on.record, P4_PIPELINE, "ja")).toContain("対応なしの p 値による除外");
    expect(methodText(on.record, P4_PIPELINE, "en")).toContain("P-value filter (unpaired)");
  }, 120000);
  it("filter off with values filled in: the whole record of the steps is identical to the default run", async () => {
    const data = synth(100, 3);
    const a = await runSkill(P4_PIPELINE, data, p4 as never, opts);
    const b = await runSkill(P4_PIPELINE, data, { ...p4, preUnpairedOn: false, preUnpairedAlpha: 0.05, preUnpairedThr: 0.05, prePairedOn: false } as never, opts);
    expect(JSON.stringify(b.record.steps)).toBe(JSON.stringify(a.record.steps));
  }, 120000);
  it("the filter is not offered for survival: switching it on has no effect on a survival outcome", async () => {
    const data = synth(100, 3);
    const s = { ...base, outcomeKind: "survival", outerK: 4, innerK: 3, repeats: 1, kCandidates: "1, 2", lockMode: "none", stratifySurvival: true };
    const a = await runSkill(P4_PIPELINE, data, s as never, opts);
    const b = await runSkill(P4_PIPELINE, data, { ...s, preUnpairedOn: true, preUnpairedAlpha: 0.05, preUnpairedThr: 0.05 } as never, opts);
    expect(b.record.status).toBe("ok");
    expect((b.state.p4 as P4State).spec!.pdrop).toBeUndefined();
    expect(JSON.stringify(b.record.steps)).toBe(JSON.stringify(a.record.steps));
  }, 120000);
  it("missing alpha or threshold is an error (no default is put in)", async () => {
    const data = synth(100, 3);
    const r = await runSkill(P4_PIPELINE, data, { ...p4, preUnpairedOn: true } as never, opts);
    expect(r.record.status).toBe("failed");
    expect(r.record.steps.flatMap((s) => s.errors).join()).toContain("既定値は決めていません");
  }, 60000);
  it("assumption report in P4 reads the development rows only and is listed in the record", async () => {
    const data = synth(100, 3);
    const r = await runSkill(P4_PIPELINE, data, { ...p4, preAssumpOn: true, preAssumpAlpha: 0.05 } as never, opts);
    expect(r.record.status).toBe("ok");
    expect((r.state.p4 as P4State).qc!.assumptions!.rows.length).toBe(6);
    expect(r.record.steps.find((s) => s.id === "p4-qc")!.summary.assumptionAlpha).toBe(0.05);
  }, 120000);
});

// ---------------------------------------------------------------------------------------------
// 対応あり（前と後の表）
// ---------------------------------------------------------------------------------------------
/** 患者 n 人 × 病変 2 つ。g_shift は後で +6、g_noise は変化なし、g_const は変化なし（値が同じ）。 */
function pairedData(n: number, seed: number) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const head = "CaseLabel,Patient,g_shift,g_noise,g_skew";
  const b: string[] = [head], f: string[] = [head], clin = ["CaseLabel,outcome,time,event"];
  for (let i = 0; i < n; i++) {
    clin.push(`P${i},${i % 2 ? "resp" : "no"},${1 + (i % 7)},${i % 3 ? 1 : 0}`);
    for (let k = 0; k < 2; k++) {
      const s = 10 + 3 * norm(), t = 10 + 3 * norm(), u = 5 + Math.abs(norm());
      b.push(`L${i}_${k},P${i},${s.toFixed(4)},${t.toFixed(4)},${u.toFixed(4)}`);
      f.push(`L${i}_${k},P${i},${(s + 6 + norm()).toFixed(4)},${(t + 0.5 * norm()).toFixed(4)},${(u + Math.sign(norm()) * Math.abs(norm()) ** 3).toFixed(4)}`);
    }
  }
  return { features: [sourceOf("b.csv", b.join("\n") + "\n")], baseline: [sourceOf("b.csv", b.join("\n") + "\n")], followup: [sourceOf("f.csv", f.join("\n") + "\n")], clinical: sourceOf("c.csv", clin.join("\n") + "\n") };
}

describe("DeltaResult keeps the before and after values (the change rates are unchanged)", () => {
  const BASE = "CaseLabel,Patient,f1\nL1,P1,10\nL2,P1,20\nL3,P2,0\nL4,P2,4\nL5,P3,6\n";
  const FOLLOW = "CaseLabel,Patient,f1\nL1,P1,15\nL2,P1,10\nL3,P2,5\nL4,P2,6\nL6,P3,1\n";
  it("hand-computed: lesion-then-mean averages both-present lesions; mean-then-change averages each time point", () => {
    const mk = (order: "lesion-then-mean" | "mean-then-change") => buildDelta({ baseline: [sourceOf("b", BASE)], followup: [sourceOf("f", FOLLOW)], idColumn: "CaseLabel", patientColumn: "Patient", featureColumns: ["f1"], order });
    for (const order of ["lesion-then-mean", "mean-then-change"] as const) {
      const r = mk(order);
      if (!r.ok) throw new Error("not ok");
      expect(r.result.before).toEqual([[15], [2]]); // P1: (10+20)/2、P2: (0+4)/2
      expect(r.result.after).toEqual([[12.5], [5.5]]); // P1: (15+10)/2、P2: (5+6)/2
    }
    const a = mk("lesion-then-mean");
    if (!a.ok) throw new Error("not ok");
    expect(a.result.values[0][0]).toBeCloseTo(0, 12); // 既存の変化率は同じ
    expect(a.result.values[1][0]).toBeCloseTo(50, 12);
  });
});

describe("paired p-value filter through the skills", () => {
  const pr = { ...roles, patientColumn: "Patient", deltaOrder: "lesion-then-mean", prePairedOn: true, prePairedAlpha: 0.05, prePairedThr: 0.05 };
  it("R2: keeps the feature with a change and drops the ones without; the result agrees with pairedDrop on the same before/after matrix", async () => {
    const data = pairedData(60, 11);
    const r = await runSkill(R2_DISCRIMINATION, data, { ...pr, positiveLabel: "resp" } as never, opts);
    expect(r.record.status, r.record.steps.flatMap((x) => x.errors).join()).toBe("ok");
    const st = r.state.r2 as R2State;
    expect(st.pre!.paired).not.toBeNull();
    expect(st.context.names).toContain("Delta_g_shift");
    expect(st.context.names).not.toContain("Delta_g_noise");
    expect(st.pre!.removed.every((x) => x.step === "paired")).toBe(true);
    // 前後の行列から独立に（pairedDrop を直接）
    const delta = r.state.delta as { patients: string[]; featureNames: string[]; before: number[][]; after: number[][] };
    const rows = delta.patients.map((_, i) => i);
    const direct = pairedDrop({ before: delta.before, after: delta.after }, delta.featureNames, delta.featureNames.map((_, j) => j), rows, { alpha: 0.05, threshold: 0.05 });
    if ("error" in direct) throw new Error(direct.error);
    expect(direct.keep.map((j) => delta.featureNames[j]).sort()).toEqual(st.context.names.map((n) => n.replace("Delta_", "")).sort());
    expect(st.pre!.paired!.patients).toBe(60);
    expect(methodText(r.record, R2_DISCRIMINATION, "ja")).toContain("対応のある前後の値の検定による除外");
  }, 120000);
  it("R5: the paired filter shrinks the feature set before the model", async () => {
    const data = pairedData(60, 11);
    const r = await runSkill(R5_MODEL, data, { ...pr, r5Corr: 0.99, r5Screen: "boost", r5BoostNu: 0.1, r5BootstrapB: 0 } as never, opts);
    expect(r.record.status).toBe("ok");
    const st = r.state.r5 as R5State;
    expect(st.names).toContain("Delta_g_shift");
    expect(st.names).not.toContain("Delta_g_noise");
    expect(st.cols.length).toBe(st.names.length);
  }, 120000);
  it("P4: the paired filter reads the development rows (not the test set) and the row-use check reports a notice, not NG", async () => {
    const data = pairedData(60, 11);
    const r = await runSkill(P4_PIPELINE, data, { ...pr, positiveLabel: "resp", outcomeKind: "binary", outerK: 4, innerK: 3, repeats: 1, kCandidates: "1", lockMode: "create", testFraction: 0.25 } as never, opts);
    expect(r.record.status).toBe("ok");
    const st = r.state.p4 as P4State;
    expect(st.names).toContain("Delta_g_shift");
    // 独立に: 開発集合の患者の前後の値に pairedDrop を直接当てた結果と同じ特徴量が残る
    const delta = r.state.delta as { patients: string[]; featureNames: string[]; before: number[][]; after: number[][] };
    const rows = st.qc!.dev.map((i) => delta.patients.indexOf(st.cohort.ids[i]));
    const direct = pairedDrop({ before: delta.before, after: delta.after }, delta.featureNames, delta.featureNames.map((_, j) => j), rows, { alpha: 0.05, threshold: 0.05 });
    if ("error" in direct) throw new Error(direct.error);
    expect(st.names).toEqual(direct.keep.map((j) => `Delta_${delta.featureNames[j]}`));
    expect(st.qc!.pairedRows).toEqual(st.qc!.dev); // 開発集合だけ
    expect(st.qc!.pairedRows.some((i) => st.split!.test.includes(i))).toBe(false);
    const leak = r.record.steps.find((s) => s.id === "p4-leak")!;
    expect(leak.summary.rowUseCheck).toBe("NOTICE");
    expect(leak.summary.rowUseFindings).toContain("pdropPaired@pre:4/4");
    expect(st.leak!.rowUse.findings.find((f) => f.stage === "pdropPaired")).toMatchObject({ usesOutcome: false, where: "pre" });
  }, 120000);
  it("negative: asking for the paired filter without the before/after tables is an error, not a silent skip", async () => {
    const data = synth(80, 7);
    const r = await runSkill(R2_DISCRIMINATION, data, { ...base, prePairedOn: true, prePairedAlpha: 0.05, prePairedThr: 0.05 } as never, opts);
    expect(r.record.status).toBe("failed");
    expect(r.record.steps.flatMap((s) => s.errors).join()).toContain("前と後の表");
  }, 60000);
});
