import { describe, expect, it } from "vitest";
import { runSkill } from "../src/core/engine";
import { R3_REPRODUCIBILITY, type R3State } from "../src/core/skills";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { methodText } from "../src/core/methodText";
import { blandAltmanFor } from "../src/core/r3";
import { blandAltmanCsv, exclusionCsv, r3ResultsCsv } from "../src/core/exportR3";

/** 3 条件。f_same は全条件で同じ、f_noisy は条件ごとに大きなばらつき、f_shift は条件 C が +5 ずれる、f_rand は無関係な乱数。 */
function synth(n = 40, seed = 1) {
  const r = new Rng(seed);
  const rows = Array.from({ length: n }, (_, i) => ({ id: `L${i}`, base: r.next() * 10 }));
  const make = (cond: number) => {
    const lines = ["CaseLabel,f_same,f_noisy,f_shift,f_rand,f_gap,f_text,Roi_Voxels"];
    rows.forEach((x, i) => {
      const noisy = x.base + (r.next() - 0.5) * 8;
      lines.push([x.id, x.base.toFixed(5), noisy.toFixed(5), (x.base + (cond === 2 ? 5 : 0) + (r.next() - 0.5) * 0.2).toFixed(5), (r.next() * 10).toFixed(5), i % 5 === 0 && cond === 1 ? "" : (x.base + r.next() * 0.1).toFixed(5), i === 0 ? "oops" : "1", String(100 + i)].join(","));
    });
    return sourceOf(`cond${"ABC"[cond]}.csv`, lines.join("\n") + "\n", "b".repeat(64));
  };
  return { features: [make(0), make(1), make(2)] };
}
const opts = (seed = 1) => ({ seed, pluginVersion: "0.4.0", now: () => "2026-10-27T00:00:00Z" });

describe("R3 reproducibility skill", () => {
  it("grades the planted features and runs without a clinical table", async () => {
    const { record, state } = await runSkill(R3_REPRODUCIBILITY, synth(), { minCategory: "good" }, opts());
    expect(record.status).toBe("ok");
    const st = state.r3 as R3State;
    const by = (f: string) => st.rows.find((r) => r.feature === f)!;
    expect(by("f_same").result!.single.icc).toBeGreaterThan(0.999);
    expect(by("f_same").result!.category).toBe("excellent");
    expect(by("f_noisy").result!.category).not.toBe("excellent");
    expect(by("f_rand").result!.category).toBe("poor");
    expect(by("f_rand").result!.singleTruncated.lower).toBeGreaterThanOrEqual(0);
    // 条件 C が +5 ずれる: 条件間の相関は高いが absolute agreement の ICC は下がる
    expect(by("f_shift").result!.single.icc).toBeLessThan(by("f_same").result!.single.icc - 0.05);
    expect(by("f_gap").dropped).toBeGreaterThan(0);
    expect(by("f_gap").n + by("f_gap").dropped).toBe(40);
    expect(st.candidates.map((r) => r.feature)).toContain("f_rand");
    expect(st.candidates.map((r) => r.feature)).not.toContain("f_same");
    expect(record.inputs.length).toBe(3);
  });
  it("lists the columns it could not use, with the reason (no silent dropping)", async () => {
    const { state } = await runSkill(R3_REPRODUCIBILITY, synth(), {}, opts());
    const ex = (state.r3 as R3State).data.report.excludedColumns;
    expect(ex.find((e) => e.name === "f_text")!.reason).toMatch(/数でない値/);
    expect(ex.find((e) => e.name === "Roi_Voxels")).toBeUndefined(); // 数値の列なので共通の列として使われる（除外は利用者が決める）
  });
  it("lists rows missing in some condition and does not use them", async () => {
    const s = synth(30, 2);
    s.features[1] = sourceOf("condB.csv", s.features[1].table.rows.slice(2).reduce((acc, r) => `${acc}${r.join(",")}\n`, `${s.features[1].table.header.join(",")}\n`));
    const { state, record } = await runSkill(R3_REPRODUCIBILITY, s, {}, opts());
    const rep = (state.r3 as R3State).data.report;
    expect(rep.matched).toBe(28);
    expect(rep.unmatched.map((u) => u.id).sort()).toEqual(["L0", "L1"]);
    expect(rep.unmatched[0].missingIn).toEqual(["condB"]);
    expect(record.steps[0].warnings.join()).toMatch(/揃っていない行が 2 個/);
  });
  it("stops with a reason: one condition, duplicate IDs, a missing ID column (negative)", async () => {
    const one = await runSkill(R3_REPRODUCIBILITY, { features: [synth().features[0]] }, {}, opts());
    expect(one.record.steps[0].errors.join()).toMatch(/2 つ以上/);
    const d = synth();
    const dup = { features: [d.features[0], sourceOf("dup.csv", `${d.features[1].table.header.join(",")}\n${d.features[1].table.rows.concat([d.features[1].table.rows[0]]).map((r) => r.join(",")).join("\n")}\n`)] };
    expect((await runSkill(R3_REPRODUCIBILITY, dup, {}, opts())).record.steps[0].errors.join()).toMatch(/重複/);
    expect((await runSkill(R3_REPRODUCIBILITY, synth(), { reliabilityIdColumn: "nope" }, opts())).record.steps[0].errors.join()).toMatch(/列「nope」がありません/);
    expect((await runSkill(R3_REPRODUCIBILITY, synth(), { minCategory: "great" }, opts())).record.steps[0].errors.join()).toMatch(/除外の基準/);
  });
  it("does not make a candidate list unless a criterion is chosen", async () => {
    const { state } = await runSkill(R3_REPRODUCIBILITY, synth(), {}, opts());
    expect((state.r3 as R3State).candidates).toEqual([]);
  });
  it("is reproducible and gives the same ICC when the conditions are listed in another order", async () => {
    const a = await runSkill(R3_REPRODUCIBILITY, synth(), {}, opts()), b = await runSkill(R3_REPRODUCIBILITY, synth(), {}, opts());
    expect(b.record).toEqual(a.record);
    const s = synth();
    const rev = await runSkill(R3_REPRODUCIBILITY, { features: [...s.features].reverse() }, {}, opts());
    const x = (a.state.r3 as R3State).rows[0].result!.single.icc, y = (rev.state.r3 as R3State).rows[0].result!.single.icc;
    expect(Math.abs(x - y)).toBeLessThan(1e-12);
  });
  it("Bland–Altman for a chosen pair, with exports and the method text", async () => {
    const { record, state } = await runSkill(R3_REPRODUCIBILITY, synth(), { minCategory: "moderate" }, opts());
    const st = state.r3 as R3State;
    const ba = blandAltmanFor(st.data, "f_shift", 0, 2);
    if ("error" in ba) throw new Error(ba.error);
    expect(ba.bias).toBeGreaterThan(4.8); expect(ba.bias).toBeLessThan(5.2); // 条件 C は +5
    expect("error" in blandAltmanFor(st.data, "f_shift", 1, 1)).toBe(true);
    expect(blandAltmanCsv(ba, "f_shift", "condA", "condC")).toContain("mean,difference");
    expect(r3ResultsCsv(st.rows).split("\r\n")[0]).toContain("icc_lower_truncated");
    expect(exclusionCsv(st.candidates, "moderate")).toContain("f_rand");
    const ja = methodText(record, R3_REPRODUCIBILITY, "ja"), en = methodText(record, R3_REPRODUCIBILITY, "en");
    expect(ja).toContain("absolute agreement");
    expect(ja).toContain("moderate より低い");
    expect(en).toContain("McGraw–Wong");
  });
});
