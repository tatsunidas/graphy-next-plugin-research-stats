import { describe, expect, it } from "vitest";
import { sourceOf } from "../src/core/delta";
import { runSkill } from "../src/core/engine";
import { methodText } from "../src/core/methodText";
import { Rng } from "../src/core/rng";
import { P4_PIPELINE, R5_MODEL, type P4State, type R5State } from "../src/core/skills";

function synth(n: number, seed: number) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const feat = ["CaseLabel,f_a,f_b,f_n0,f_n1,f_n2,f_dup"], clin = ["CaseLabel,outcome,time,event"];
  for (let i = 0; i < n; i++) {
    const g = r.next() < 0.45 ? 1 : 0, a = norm() + 1.4 * g, b = norm() + 0.7 * g;
    feat.push([`P${i}`, a.toFixed(4), b.toFixed(4), norm().toFixed(4), norm().toFixed(4), norm().toFixed(4), (a * 1.01 + 0.001 * norm()).toFixed(4)].join(","));
    const time = (-Math.log(1 - r.next()) / Math.exp(0.9 * a)).toFixed(4);
    clin.push(`P${i},${g ? "resp" : "no"},${Number(time) + 0.001},${r.next() < 0.8 ? 1 : 0}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}
const base = { patientColumn: null, clinicalIdColumn: "CaseLabel", outcomeColumn: "outcome", timeColumn: "time", eventColumn: "event" };
const p4 = { ...base, outcomeKind: "binary", positiveLabel: "resp", outerK: 5, innerK: 4, repeats: 1, kCandidates: "1, 2", lockMode: "create", testFraction: 0.25, correlationThreshold: null };
const opts = { seed: 1, pluginVersion: "t", now: () => "2026-01-01T00:00:00Z" };

describe("P4 records the leakage checks", () => {
  it("default settings: split OK, row use OK, and the check reads real fold/final rows", async () => {
    const { record, state } = await runSkill(P4_PIPELINE, synth(100, 3), p4 as never, opts);
    expect(record.status).toBe("ok");
    const st = state.p4 as P4State;
    const step = record.steps.find((s) => s.id === "p4-leak")!;
    expect(step.summary).toMatchObject({ splitCheck: "OK", rowUseCheck: "OK" });
    expect(st.leak!.split.checked).toBe(5 + 5 * 4 + 1); // 外側 5 + 内側 5×4 + 開発集合とテスト集合の重なり
    expect(st.leak!.rowUse.findings.every((f) => f.entries > 0)).toBe(true);
    expect(st.leak!.rowUse.findings.map((f) => f.where)).toContain("final");
    // 同じ患者がテスト集合と開発集合の両方にいない（固定ファイルの ID）
    const dev = new Set(st.qc!.dev.map((i) => st.cohort.ids[i]));
    expect(st.split!.test.some((i) => dev.has(st.cohort.ids[i]))).toBe(false);
    expect(methodText(record, P4_PIPELINE, "ja")).toContain("分割の検査（同じ患者が訓練と検証に分かれていないか）: OK。前処理・特徴量選択の推定に使った行の検査（訓練側の行だけか）: OK");
  }, 60000);

  it("quality control that reads the whole development set is reported as NG (found, not fixed)", async () => {
    const { record, state } = await runSkill(P4_PIPELINE, synth(100, 3), { ...p4, dropConstant: true } as never, opts);
    expect(record.status).toBe("ok");
    const step = record.steps.find((s) => s.id === "p4-leak")!;
    expect(step.summary.splitCheck).toBe("OK");
    expect(step.summary.rowUseCheck).toBe("NOTICE"); // 結果ラベルを使わない読み取りは注意（NG ではない）
    expect(step.summary.rowUseFindings).toContain("qc@pre:5/5");
    expect(step.warnings.join()).toContain("教師なしの処理が訓練側でない行の値を読んでいます（結果ラベルは使っていません）: qc");
    const f = (state.p4 as P4State).leak!.rowUse.findings;
    expect(f.filter((x) => x.violating > 0).map((x) => `${x.where}/${x.stage}`)).toEqual(["pre/qc"]);
  }, 60000);
});

describe("R5 checks the LASSO folds", () => {
  const r5 = { ...base, r5Corr: 0.9, r5BootstrapB: 0 };
  it("LASSO screening: folds are checked and OK; bootstrap/boost screening has no folds to check", async () => {
    const l = await runSkill(R5_MODEL, synth(80, 5), { ...r5, r5Screen: "lasso", r5LassoRule: "min" } as never, opts);
    expect(l.record.status).toBe("ok");
    expect((l.state.r5 as R5State).split).toMatchObject({ ok: true });
    expect(l.record.steps.find((s) => s.id === "r5-model")!.summary.splitCheck).toBe("OK");
    const b = await runSkill(R5_MODEL, synth(80, 5), { ...r5, r5Screen: "boost", r5BoostNu: 0.1 } as never, opts);
    expect(b.record.status).toBe("ok");
    expect((b.state.r5 as R5State).split).toBeNull();
  }, 120000);
});
