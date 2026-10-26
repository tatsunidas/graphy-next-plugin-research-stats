import { beforeAll, describe, expect, it } from "vitest";
import { runRecipe } from "../src/core/engine";
import { DATASET_CHECK } from "../src/core/recipes";
import { methodText } from "../src/core/methodText";
import { sourceOf } from "../src/core/delta";
import type { RunInputs } from "../src/core/recipe";

const FEAT = "CaseLabel,Patient,f1\nL1,P1,1\nL2,P1,3\nL3,P2,5\n";
const CLIN = "Patient,time,event\nP1,12,1\nP2,8,0\nP9,4,1\n";
const inputs = (): RunInputs => ({
  features: [sourceOf("feat.csv", FEAT, "a".repeat(64))],
  clinical: sourceOf("clin.csv", CLIN, null),
});
const params = { patientColumn: "Patient", clinicalIdColumn: "Patient", timeColumn: "time", eventColumn: "event" };
const opts = { seed: 42, pluginVersion: "0.1.0", now: () => "2026-10-26T00:00:00Z" };

describe("runRecipe", () => {
  it("runs the join, skips the delta step and records everything needed to reproduce", async () => {
    const { record, state } = await runRecipe(DATASET_CHECK, inputs(), params, opts);
    expect(record.status).toBe("ok");
    expect(record.steps.map((s) => `${s.id}:${s.status}`)).toEqual(["delta:skipped", "join:ok"]);
    expect(record.steps[1].summary).toMatchObject({ joined: 2, featureOnly: 0, clinicalOnly: 1, survivalUsable: 2, features: 1 });
    expect(record.seed).toBe(42);
    expect(record.pluginVersion).toBe("0.1.0");
    expect(record.inputs).toEqual([
      { name: "feat.csv", sha256: "a".repeat(64), rows: 3 },
      { name: "clin.csv", sha256: null, rows: 3 },
    ]);
    expect(record.changedFromDefault.sort()).toEqual(["clinicalIdColumn", "eventColumn", "patientColumn", "timeColumn"]);
    expect(state.dataset).toBeTruthy();
  });
  it("gives the same record for the same inputs, settings and seed, and records a changed seed", async () => {
    const a = (await runRecipe(DATASET_CHECK, inputs(), params, opts)).record;
    const b = (await runRecipe(DATASET_CHECK, inputs(), params, opts)).record;
    expect(b).toEqual(a);
    expect((await runRecipe(DATASET_CHECK, inputs(), params, { ...opts, seed: 43 })).record.seed).toBe(43);
  });
  it("stops at the failed step, keeps its errors and does not run the later steps (negative)", async () => {
    const { record } = await runRecipe(DATASET_CHECK, inputs(), { ...params, eventColumn: "missingcol" }, opts);
    expect(record.status).toBe("failed");
    const join = record.steps.find((s) => s.id === "join")!;
    expect(join.status).toBe("failed");
    expect(join.errors.join()).toMatch(/列「missingcol」がありません/);
  });
  it("refuses a setting the recipe does not have (negative)", async () => {
    await expect(runRecipe(DATASET_CHECK, inputs(), { nope: 1 }, opts)).rejects.toThrow(/無い設定/);
  });
  it("runs the delta step when two time points are given and feeds the join with it", async () => {
    const base = sourceOf("base.csv", "CaseLabel,Patient,f1\nL1,P1,10\nL2,P2,20\n");
    const fol = sourceOf("fol.csv", "CaseLabel,Patient,f1\nL1,P1,15\nL2,P2,10\n");
    const { record, state } = await runRecipe(DATASET_CHECK, { ...inputs(), baseline: [base], followup: [fol] }, { ...params, featureIdColumn: "CaseLabel" }, opts);
    expect(record.status).toBe("ok");
    expect(record.steps.map((s) => s.status)).toEqual(["ok", "ok"]);
    const ds = state.dataset as { featureNames: string[]; features: number[][] };
    expect(ds.featureNames).toEqual(["Delta_f1"]);
    expect(ds.features).toEqual([[50], [-50]]);
  });
});

describe("methodText", () => {
  let run: Awaited<ReturnType<typeof runRecipe>>;
  beforeAll(async () => { run = await runRecipe(DATASET_CHECK, inputs(), params, opts); });
  it("writes the version, seed, counts and changed settings in both languages", async () => {
    const ja = methodText(run.record, DATASET_CHECK, "ja");
    expect(ja).toContain("research-stats 0.1.0");
    expect(ja).toContain("乱数シード: 42");
    expect(ja).toContain("2 患者を解析の対象とした");
    expect(ja).toContain("timeColumn=time");
    expect(ja).toContain("aaaaaaaaaaaa"); // SHA-256 の先頭 12 桁
    expect(ja).toContain("未取得"); // 取れなかったハッシュは「未取得」
    expect(ja).toContain("診断・治療の判断に用いない");
    const en = methodText(run.record, DATASET_CHECK, "en");
    expect(en).toContain("Random seed: 42");
    expect(en).toContain("2 patients were analysed");
    expect(en).toContain("not available");
  });
  it("is the same text for the same record", async () => {
    expect(methodText(run.record, DATASET_CHECK, "ja")).toBe(methodText(structuredClone(run.record), DATASET_CHECK, "ja"));
  });
  it("says the settings were the defaults when nothing was changed", async () => {
    const r = await runRecipe(DATASET_CHECK, { features: [sourceOf("f.csv", "CaseLabel,f1\nA,1\n")], clinical: sourceOf("c.csv", "CaseLabel\nA\n") }, {}, opts);
    expect(methodText(r.record, DATASET_CHECK, "ja")).toContain("設定は既定値のままであった");
  });
});

describe("recipes that need a clinical table", () => {
  it("stop with a reason, not an exception, when there is none", async () => {
    const { record } = await runRecipe(DATASET_CHECK, { features: [sourceOf("f.csv", "CaseLabel,f1\nA,1\n")] }, {}, opts);
    expect(record.status).toBe("failed");
    expect(record.steps.at(-1)!.errors.join()).toMatch(/臨床の表がありません/);
    expect(record.inputs.length).toBe(1);
  });
});
