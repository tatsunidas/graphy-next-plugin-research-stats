/**
 * 組み込みのレシピ。この段（基盤）は「データセットの確認」だけで、統計の手法は持たない。
 * 結合・集約・変化率の結果を確かめ、結合済みのデータセットを書き出すために使う。
 */
import { buildDataset, type Dataset, type JoinReport } from "./dataset";
import { buildDelta, deltaToTable, type DeltaOrder, type DeltaResult } from "./delta";
import type { ParamValue, Recipe, StepOutcome } from "./recipe";

const str = (v: ParamValue, name: string): string | null => {
  if (v === null || v === "") return null;
  if (typeof v !== "string") throw new Error(`設定「${name}」は文字列にしてください`);
  return v;
};

export const DATASET_CHECK: Recipe = {
  id: "dataset-check",
  title: { ja: "データセットの確認", en: "Dataset check" },
  question: {
    ja: "特徴量の表と臨床の表は、患者の単位で正しく結合できているか。",
    en: "Do the feature table and the clinical table join correctly at the patient level?",
  },
  sources: [],
  defaults: {
    featureIdColumn: "CaseLabel",
    patientColumn: null,
    featureColumns: null,
    clinicalIdColumn: "CaseLabel",
    timeColumn: null,
    eventColumn: null,
    outcomeColumn: null,
    covariates: [],
    deltaOrder: "lesion-then-mean",
  },
  steps: [
    {
      id: "delta",
      title: { ja: "治療前後の変化率", en: "Percent change between time points" },
      run(ctx): StepOutcome {
        const { baseline, followup } = ctx.inputs;
        if (!baseline?.length || !followup?.length) return { summary: {}, warnings: [], skipped: true };
        const order = ctx.params.deltaOrder as DeltaOrder;
        const featureColumns = ctx.params.featureColumns;
        const r = buildDelta({
          baseline, followup, idColumn: str(ctx.params.featureIdColumn, "featureIdColumn")!,
          patientColumn: str(ctx.params.patientColumn, "patientColumn"),
          featureColumns: Array.isArray(featureColumns) ? featureColumns : null, order,
        });
        if (!r.ok) return { summary: {}, warnings: [], errors: r.errors };
        ctx.state.delta = r.result;
        const d: DeltaResult = r.result;
        const warnings = d.unmatched.length ? [`片方の時点にしか無い病変が ${d.unmatched.length} 個あります（変化率に使いません）`] : [];
        if (d.problems.length) warnings.push(`変化率を作れなかった値が ${d.problems.length} 個あります（欠損にしました）`);
        return { summary: { order, patients: d.patients.length, unmatchedLesions: d.unmatched.length, impossible: d.problems.length }, warnings };
      },
    },
    {
      id: "join",
      title: { ja: "結合と集約", en: "Join and aggregate" },
      run(ctx): StepOutcome {
        const p = ctx.params;
        const delta = ctx.state.delta as DeltaResult | undefined;
        const idCol = str(p.featureIdColumn, "featureIdColumn")!;
        // 変化率があるときは、患者ごとの変化率の表を特徴量として結合する（1 行 = 1 患者）
        const features = delta
          ? [{ name: "delta", sha256: null, table: deltaToTable(delta, idCol) }]
          : ctx.inputs.features;
        const r = buildDataset({
          features, featureIdColumn: idCol, patientColumn: delta ? null : str(p.patientColumn, "patientColumn"),
          featureColumns: delta ? null : Array.isArray(p.featureColumns) ? p.featureColumns : null,
          clinical: ctx.inputs.clinical, clinicalIdColumn: str(p.clinicalIdColumn, "clinicalIdColumn")!,
          roles: {
            time: str(p.timeColumn, "timeColumn"), event: str(p.eventColumn, "eventColumn"),
            outcome: str(p.outcomeColumn, "outcomeColumn"), covariates: Array.isArray(p.covariates) ? p.covariates : [],
          },
        });
        if (!r.ok) return { summary: {}, warnings: [], errors: r.errors };
        ctx.state.dataset = r.dataset satisfies Dataset;
        ctx.state.report = r.report satisfies JoinReport;
        const rep = r.report;
        return {
          summary: {
            featureRows: rep.featureRows, patientsInFeatures: rep.patientsInFeatures, patientsInClinical: rep.patientsInClinical,
            joined: rep.joined, featureOnly: rep.featureOnly.length, clinicalOnly: rep.clinicalOnly.length,
            features: r.dataset.featureNames.length, survivalUsable: rep.survivalUsable,
          },
          warnings: rep.warnings,
        };
      },
    },
  ],
};

export const RECIPES: Recipe[] = [DATASET_CHECK];
