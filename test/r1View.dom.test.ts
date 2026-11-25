// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { runSkill } from "../src/core/engine";
import { R1_PROGNOSIS, type R1State } from "../src/core/skills";
import { sourceOf } from "../src/core/delta";
import { Rng } from "../src/core/rng";
import { makeT } from "../src/ui/i18n";
import { r1Panel } from "../src/ui/r1View";

// f_sep: 値が低い三分の一は全員が打ち切り（群 1 に死亡なし）。f_ties: 同じ値が 7 割で三分位に分けられない。g0: 無関係
function data() {
  const n = 90, r = new Rng(9);
  const feat = ["CaseLabel,f_sep,f_ties,g0"], clin = ["CaseLabel,time,event,age"];
  for (let i = 0; i < n; i++) {
    const v = i / n;
    feat.push(`P${i},${v.toFixed(4)},${i % 10 < 7 ? "0" : r.next().toFixed(3)},${r.next().toFixed(4)}`);
    clin.push(`P${i},${(5 + (1 - v) * 40 + r.next() * 5).toFixed(2)},${v < 1 / 3 ? 0 : r.next() < 0.8 ? 1 : 0},${50 + (i % 20)}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}

describe("R1 result panel", () => {
  it("shows 'not estimable' for the group HR and the QC reason, and the multivariable cap note (ja and en)", async () => {
    const { state, record } = await runSkill(R1_PROGNOSIS, data(), { patientColumn: null, clinicalIdColumn: "CaseLabel", timeColumn: "time", eventColumn: "event", covariates: ["age"], groupMode: "tertile", bootstrapB: 0 }, { seed: 1, pluginVersion: "t", now: () => "2026-11-25T00:00:00Z" });
    expect(record.status).toBe("ok");
    for (const loc of ["ja", "en"] as const) {
      const el = r1Panel(state.r1 as R1State, makeT(loc), null, () => {});
      document.body.replaceChildren(el);
      const rowOf = (f: string) => el.querySelector(`[data-testid="rs-r1-row"][data-feature="${f}"]`) as HTMLElement;
      expect(rowOf("f_sep").textContent).toContain("推定不能（群 1 に死亡なし）");
      expect(rowOf("f_sep").querySelector('[data-testid="rs-r1-q"]')!.textContent).not.toBe("—"); // log-rank の q は出ている
      expect(rowOf("f_ties").textContent).toContain("層に分けられません");
      expect(rowOf("g0").textContent).not.toContain("推定不能");
    }
  });
});
