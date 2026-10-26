/**
 * レシピの実行エンジン。手順を順に実行し、再現に要る情報（版・シード・入力の SHA-256・設定・
 * 既定値からの変更・手順ごとの要約と警告）を RunRecord に残す。同じ入力・設定・シードなら記録も同じ。
 */
import { Rng } from "./rng";
import type { ParamValue, Recipe, RunInputs, StepOutcome } from "./recipe";

export interface RecordedInput {
  name: string;
  sha256: string | null;
  rows: number;
}

export interface RecordedStep {
  id: string;
  status: "ok" | "failed" | "skipped";
  summary: StepOutcome["summary"];
  warnings: string[];
  errors: string[];
}

export interface RunRecord {
  recipeId: string;
  pluginVersion: string;
  startedAt: string;
  seed: number;
  inputs: RecordedInput[];
  params: Record<string, ParamValue>;
  changedFromDefault: string[];
  steps: RecordedStep[];
  status: "ok" | "failed";
}

export interface RunOptions {
  seed: number;
  pluginVersion: string;
  /** 開始時刻の文字列。試験で固定できるよう外から渡す。 */
  now: () => string;
}

const same = (a: ParamValue, b: ParamValue) => JSON.stringify(a) === JSON.stringify(b);

export function runRecipe(
  recipe: Recipe, inputs: RunInputs, params: Record<string, ParamValue>, opts: RunOptions,
): { record: RunRecord; state: Record<string, unknown> } {
  const merged: Record<string, ParamValue> = { ...recipe.defaults, ...params };
  const unknown = Object.keys(params).filter((k) => !(k in recipe.defaults));
  if (unknown.length) throw new Error(`レシピ ${recipe.id} に無い設定です: ${unknown.join(", ")}`);
  const changed = Object.keys(merged).filter((k) => !same(merged[k], recipe.defaults[k]));

  const all = [...inputs.features, inputs.clinical, ...(inputs.baseline ?? []), ...(inputs.followup ?? [])];
  const record: RunRecord = {
    recipeId: recipe.id,
    pluginVersion: opts.pluginVersion,
    startedAt: opts.now(),
    seed: opts.seed,
    inputs: all.map((s) => ({ name: s.name, sha256: s.sha256, rows: s.table.rows.length })),
    params: merged,
    changedFromDefault: changed,
    steps: [],
    status: "ok",
  };
  const state: Record<string, unknown> = {};
  const rng = new Rng(opts.seed);
  for (const step of recipe.steps) {
    let out: StepOutcome;
    try {
      out = step.run({ inputs, params: merged, rng, state });
    } catch (e) {
      out = { summary: {}, warnings: [], errors: [`手順「${step.id}」が例外で止まりました: ${e instanceof Error ? e.message : String(e)}`] };
    }
    const failed = !!out.errors?.length;
    record.steps.push({
      id: step.id,
      status: failed ? "failed" : out.skipped ? "skipped" : "ok",
      summary: out.summary,
      warnings: out.warnings,
      errors: out.errors ?? [],
    });
    if (failed) {
      record.status = "failed";
      break;
    }
  }
  return { record, state };
}
