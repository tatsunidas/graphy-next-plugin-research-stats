import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { runSkill } from "../src/core/engine";
import { DATASET_CHECK, P4_PIPELINE, R1_PROGNOSIS, R2_DISCRIMINATION, R5_MODEL } from "../src/core/skills";
import { methodText } from "../src/core/methodText";
import { multiRow, opts, roles } from "./helpers/multiRow";

/**
 * 「行の扱い」を足す前（平均のみ）のコードで作った実行記録。既定（平均）の実行は、記録がバイト単位で同じでなければならない。
 * GOLDEN_OUT=<path> を付けて実行すると、今のコードの記録をそのパスへ書く（足す前のコードで 1 度だけ使った）。
 */
const GOLDEN = new URL("./fixtures/default_records.json", import.meta.url);
const p4 = { ...roles, outcomeKind: "binary", positiveLabel: "resp", outerK: 4, innerK: 3, repeats: 1, kCandidates: "1, 2" };

async function records() {
  const d = multiRow(70, 5);
  const runs: [string, () => ReturnType<typeof runSkill>][] = [
    ["check", () => runSkill(DATASET_CHECK, d, { ...roles, timeColumn: null, eventColumn: null, outcomeColumn: null } as never, opts)],
    ["r1", () => runSkill(R1_PROGNOSIS, d, { ...roles, bootstrapB: 0, mannWhitney: false } as never, opts)],
    ["r2", () => runSkill(R2_DISCRIMINATION, d, { ...roles, positiveLabel: "resp" } as never, opts)],
    ["r5", () => runSkill(R5_MODEL, d, { ...roles, r5Corr: 0.9, r5Screen: "boost", r5BoostNu: 0.1, r5BootstrapB: 0 } as never, opts)],
    ["p4-topk-none", () => runSkill(P4_PIPELINE, d, { ...p4, lockMode: "none" } as never, opts)],
    ["p4-topk-create", () => runSkill(P4_PIPELINE, d, { ...p4, lockMode: "create", testFraction: 0.25 } as never, opts)],
    ["p4-lasso", () => runSkill(P4_PIPELINE, d, { ...p4, kCandidates: null, model: "lasso", lassoRule: "1se", lassoNLambda: 30, lockMode: "none" } as never, opts)],
    ["p4-rf", () => runSkill(P4_PIPELINE, d, { ...p4, kCandidates: null, model: "rf", rfTrees: 20, rfMtry: "sqrt", lockMode: "none" } as never, opts)],
    ["p4-survival", () => runSkill(P4_PIPELINE, d, { ...p4, outcomeKind: "survival", positiveLabel: null, stratifySurvival: true, lockMode: "create", testFraction: 0.25 } as never, opts)],
  ];
  const out: Record<string, unknown> = {};
  for (const [name, f] of runs) {
    const r = await f();
    const skill = name.startsWith("p4") ? P4_PIPELINE : name === "r1" ? R1_PROGNOSIS : name === "r2" ? R2_DISCRIMINATION : name === "r5" ? R5_MODEL : DATASET_CHECK;
    out[name] = { record: r.record, ja: methodText(r.record, skill, "ja"), en: methodText(r.record, skill, "en") };
  }
  return JSON.stringify(out, null, 1);
}

describe("default (averaged) runs are byte-identical to the records made before the row-mode option", () => {
  it("matches the golden records", async () => {
    const now = await records();
    if (process.env.GOLDEN_OUT) { writeFileSync(process.env.GOLDEN_OUT, now); return; }
    expect(now).toBe(readFileSync(GOLDEN, "utf-8"));
    expect(JSON.parse(now)["p4-topk-none"].record.status).toBe("ok");
    expect(JSON.parse(now)["r5"].record.status).toBe("ok");
  }, 300000);
});
