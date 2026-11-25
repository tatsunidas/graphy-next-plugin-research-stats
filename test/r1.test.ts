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

/** 特徴量 nf 個（無関係な乱数）。イベントは先頭から数えて ev 件（時間は無関係）。共変量は age と sex。 */
function plain(n: number, ev: number, nf: number, seed: number, extra?: (i: number, r: Rng) => Record<string, string>) {
  const r = new Rng(seed);
  const names = Array.from({ length: nf }, (_, j) => `g${j}`);
  const extraNames = extra ? Object.keys(extra(0, new Rng(1))) : [];
  const feat = [["CaseLabel", ...names, ...extraNames].join(",")], clin = ["CaseLabel,time,event,age,sex"];
  const order = Array.from({ length: n }, (_, i) => i).sort(() => r.next() - 0.5);
  for (let i = 0; i < n; i++) {
    const e = extra ? extra(i, r) : {};
    feat.push([`P${i}`, ...names.map(() => r.next().toFixed(4)), ...extraNames.map((k) => e[k])].join(","));
    const isEv = order[i] < ev ? 1 : 0;
    clin.push(`P${i},${(1 + Math.round(r.next() * 500) / 10).toFixed(1)},${isEv},${40 + Math.floor(r.next() * 40)},${i % 2}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}

describe("R1: stratification QC and groups without an event", () => {
  it("a feature that cannot be split into the requested strata is excluded by QC, and counted apart from failures", async () => {
    const data = plain(60, 20, 3, 5, (i, r) => ({ f_ties: i % 10 < 7 ? "0" : r.next().toFixed(3) }));
    const { record, state } = await runSkill(R1_PROGNOSIS, data, { ...base, groupMode: "tertile", bootstrapB: 0 }, opts());
    expect(record.status).toBe("ok");
    const row = (state.r1 as R1State).rows.find((x) => x.feature === "f_ties")!;
    expect(row.qcExcluded).toBe(true);
    expect(row.error).toMatch(/層に分けられません/);
    expect(row.pUsed).toBeNull();
    const uni = record.steps.find((s) => s.id === "univariate")!;
    expect(uni.summary.excluded).toBe(1);
    expect(uni.summary.failed).toBe(0);
    // 負例: 普通の特徴量は除外されない
    expect((state.r1 as R1State).rows.filter((x) => x.feature.startsWith("g") && x.qcExcluded)).toHaveLength(0);
  });
  it("a group with no event: the group HR is 'not estimable', the log-rank p stays, and the feature is still selected", async () => {
    // f_sep: 値が低い三分の一は全員が打ち切り。残りはイベントが多い
    const n = 90, feat = ["CaseLabel,f_sep,g0"], clin = ["CaseLabel,time,event,age,sex"];
    const r = new Rng(9);
    for (let i = 0; i < n; i++) {
      const v = i / n; // 0..1
      feat.push(`P${i},${v.toFixed(4)},${r.next().toFixed(4)}`);
      const ev = v < 1 / 3 ? 0 : (r.next() < 0.8 ? 1 : 0);
      clin.push(`P${i},${(5 + (1 - v) * 40 + r.next() * 5).toFixed(2)},${ev},${50 + (i % 20)},${i % 2}`);
    }
    const data = { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
    const { record, state } = await runSkill(R1_PROGNOSIS, data, { ...base, groupMode: "tertile", bootstrapB: 0 }, opts());
    expect(record.status).toBe("ok");
    const st = state.r1 as R1State;
    const row = st.rows.find((x) => x.feature === "f_sep")!;
    expect(row.groupEvents[0]).toBe(0);
    expect(row.error).toBeNull();
    expect(row.hrGroup).toBeNull();
    expect(row.hrGroupNote).toMatch(/群 1 に死亡なし/);
    expect(row.logrankP).not.toBeNull();
    expect(row.q as number).toBeLessThan(0.05);
    expect(st.model!.screened).toContain("f_sep");
    expect(st.model!.selected).toContain("f_sep"); // 多変量は連続値（z 化）を使うので、群の HR が無くても入る
    // 負例: 普通の特徴量には注記が付かない
    expect(st.rows.find((x) => x.feature === "g0")!.hrGroupNote).toBeNull();
  });
});

describe("R1: the multivariable model keeps 10 events per variable and never takes the whole skill down", () => {
  const loose = { ...base, covariates: ["age", "sex"], multiplicity: "none", alpha: 0.5, bootstrapB: 0, groupMode: "median" };
  it("15 events, 2 covariates, 34 features, many pass a loose screen: no feature fits in; univariate results are kept; the skill is not failed", async () => {
    const { record, state } = await runSkill(R1_PROGNOSIS, plain(56, 15, 34, 21), loose, opts());
    expect(record.status).toBe("ok");
    const st = state.r1 as R1State;
    expect(st.rows.filter((x) => x.error === null).length).toBe(34);
    expect(st.model!.screened.length).toBeGreaterThan(5);
    expect(st.model!.selected).toHaveLength(0);
    expect(st.model!.notEntered).toHaveLength(st.model!.screened.length);
    expect(st.model!.error).toBeNull();
    expect(st.model!.notes.join(" ")).toMatch(/EPV 10/);
  });
  it("30 events, no covariate: at most 3 features enter, the ones with the smallest p, and the bootstrap uses the same cap", async () => {
    const { record, state } = await runSkill(R1_PROGNOSIS, plain(80, 30, 20, 4), { ...base, multiplicity: "none", alpha: 0.5, bootstrapB: 40, groupMode: "median" }, opts());
    expect(record.status).toBe("ok");
    const st = state.r1 as R1State, m = st.model!;
    expect(m.maxVars).toBe(3);
    expect(m.selected).toHaveLength(3);
    const byP = st.rows.filter((x) => m.screened.includes(x.feature)).sort((a, b) => (a.pUsed as number) - (b.pUsed as number)).map((x) => x.feature);
    expect(m.selected).toEqual(byP.slice(0, 3));
    expect(m.notEntered).toEqual(byP.slice(3));
    expect(m.fit).not.toBeNull();
    expect(m.corrected).not.toBeNull();
    expect(m.corrected!.failed).toBeLessThan(m.corrected!.bootstraps + m.corrected!.failed); // 上限が無いと大半が失敗する
  });
  it("enough events: nothing is held back (no cap note)", async () => {
    const { state } = await runSkill(R1_PROGNOSIS, synth(120, 11), { ...base, covariates: ["age"], bootstrapB: 0 }, opts());
    const m = (state.r1 as R1State).model!;
    expect(m.notEntered).toHaveLength(0);
    expect(m.notes.join(" ")).not.toMatch(/EPV/);
  });
});
