/**
 * 組み込みのレシピ。この段（基盤）は「データセットの確認」だけで、統計の手法は持たない。
 * 結合・集約・変化率の結果を確かめ、結合済みのデータセットを書き出すために使う。
 */
import { buildDataset, type Dataset, type JoinReport } from "./dataset";
import { buildDelta, deltaToTable, type DeltaOrder, type DeltaResult } from "./delta";
import type { ParamValue, Recipe, StepOutcome } from "./recipe";
import {
  analyzeFeature, applyMultiplicity, covariatesOf, isSelected, multivariate, R1_DEFAULTS, readR1Params, survivalCohort,
  type CovariateData, type FeatureContext, type FeatureResult, type MultivariateResult, type R1Params, type SurvivalCohort,
} from "./r1";
import {
  analyzeFeatures, binaryCohort, combination, decisionTree, meetsCriterion, R2_DEFAULTS, readR2Params,
  type BinaryCohort, type ComboResult, type DiscrimContext, type FeatureRow as DiscrimRow, type R2Params, type TreeResult,
} from "./r2";
import { treeNodes } from "./discrim/tree";
import {
  analyzeReliability, buildConditions, exclusionCandidates, R3_DEFAULTS, readR3Params,
  type ConditionData, type R3Params, type ReliabilityRow,
} from "./r3";

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
        if (!ctx.inputs.clinical) return { summary: {}, warnings: [], errors: ["臨床の表がありません（このレシピには臨床の CSV が要ります）"] };
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

export interface R1State {
  params: R1Params;
  cohort: SurvivalCohort;
  context: FeatureContext;
  covariates: CovariateData;
  rows: FeatureResult[];
  comparisons: number;
  model: MultivariateResult | null;
}

export const R1_PROGNOSIS: Recipe = {
  id: "r1-prognosis",
  title: { ja: "予後との関係（R1）", en: "Association with survival (R1)" },
  question: {
    ja: "この特徴量は、生存（または無増悪生存など）と関係するか。",
    en: "Is this feature associated with survival (or progression-free survival, etc.)?",
  },
  sources: [
    "Ganeshan B et al. Eur Radiol 2017 (PMID 27380902)",
    "Salazar P et al. PLoS One 2024 (PMID 39739866)",
    "Haider MA et al. Cancer Imaging 2017 (PMID 28114978)",
  ],
  defaults: { ...DATASET_CHECK.defaults, ...R1_DEFAULTS },
  steps: [
    ...DATASET_CHECK.steps,
    {
      id: "cohort",
      title: { ja: "生存解析の対象", en: "Survival cohort" },
      run(ctx): StepOutcome {
        const d = ctx.state.dataset as Dataset | undefined;
        if (!d) return { summary: {}, warnings: [], errors: ["データセットがありません"] };
        const pr = readR1Params(ctx.params);
        if ("error" in pr) return { summary: {}, warnings: [], errors: [pr.error] };
        const cohort = survivalCohort(d);
        if ("error" in cohort) return { summary: {}, warnings: [], errors: [cohort.error] };
        const cov = covariatesOf(d, cohort);
        if ("error" in cov) return { summary: {}, warnings: [], errors: [cov.error] };
        const context: FeatureContext = {
          surv: cohort.surv, names: d.featureNames, event: cohort.surv.event,
          values: d.featureNames.map((_, j) => cohort.patients.map((i) => d.features[i][j])),
        };
        ctx.state.r1 = { params: pr, cohort, context, covariates: cov, rows: [], comparisons: 0, model: null } satisfies R1State;
        const events = cohort.surv.event.reduce((a, b) => a + b, 0);
        const warnings = [];
        if (d.ids.length - cohort.patients.length > 0) warnings.push(`時間かイベントが欠けている患者 ${d.ids.length - cohort.patients.length} 人を除きました`);
        if (pr.groupMode === "optimal") warnings.push("最適しきい値は探索的です。素の p は多重比較で小さく出るため、並べ替え検定の補正 p を使います");
        if (pr.multiplicity === "none") warnings.push("多重比較の補正をしません（探索的な解析）");
        return { summary: { patients: cohort.patients.length, events, features: d.featureNames.length, covariates: cov.names.length }, warnings };
      },
    },
    {
      id: "univariate",
      title: { ja: "特徴量ごとの解析", en: "Per-feature analysis" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r1 as R1State;
        const rows: FeatureResult[] = [];
        const m = st.context.names.length;
        for (let j = 0; j < m; j++) {
          const label = `${st.context.names[j]} (${j + 1}/${m})`;
          rows.push(await analyzeFeature(
            st.context.names[j], st.context.values[j], st.context.surv, st.params, ctx.rng,
            (l, f) => ctx.tick(l, (j + f) / m), label,
          ));
        }
        st.rows = rows;
        st.comparisons = applyMultiplicity(rows, st.params.multiplicity);
        const failed = rows.filter((r) => r.error !== null);
        const selected = rows.filter((r) => isSelected(r, st.params));
        return {
          summary: { features: m, comparisons: st.comparisons, failed: failed.length, selected: selected.length },
          warnings: failed.length ? [`${failed.length} 個の特徴量は解析できませんでした（理由は結果の表にあります。p は空欄のままで、0 や 1 にはしていません）`] : [],
        };
      },
    },
    {
      id: "multivariate",
      title: { ja: "多変量 Cox と補正 C-index", en: "Multivariable Cox and corrected C-index" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r1 as R1State;
        const mv = await multivariate(st.context, st.rows, st.covariates, st.params, ctx.rng, ctx.tick);
        st.model = mv;
        if (mv.error) return { summary: {}, warnings: mv.notes, errors: [mv.error] };
        return {
          summary: {
            selected: mv.selected.length, nUsed: mv.nUsed, eventsUsed: mv.eventsUsed,
            cIndexApparent: mv.cIndexApparent, cIndexCorrected: mv.corrected?.corrected ?? null,
            optimism: mv.corrected?.optimism ?? null, bootstraps: mv.corrected?.bootstraps ?? null,
            addedValueP: mv.addedValue?.p ?? null,
          },
          warnings: mv.notes,
          skipped: mv.selected.length === 0,
        };
      },
    },
  ],
};

export interface R2State {
  params: R2Params;
  cohort: BinaryCohort;
  context: DiscrimContext;
  rows: DiscrimRow[];
  comparisons: number;
  combo: ComboResult | null;
  tree: TreeResult | null;
  treeNote: string | null;
}

export const R2_DISCRIMINATION: Recipe = {
  id: "r2-discrimination",
  title: { ja: "2 群の判別（R2）", en: "Discrimination of two groups (R2)" },
  question: {
    ja: "この特徴量で、結果ラベルの 2 群を判別できるか。",
    en: "Can this feature discriminate the two outcome groups?",
  },
  sources: [
    "Weiss GJ et al. PLoS One 2014 (PMID 24987838)",
    "Sidhu HS et al. Eur Radiol 2017 (PMID 27620864)",
  ],
  defaults: { ...DATASET_CHECK.defaults, ...R2_DEFAULTS },
  steps: [
    ...DATASET_CHECK.steps,
    {
      id: "outcome",
      title: { ja: "結果ラベルの確認", en: "Outcome labels" },
      run(ctx): StepOutcome {
        const d = ctx.state.dataset as Dataset | undefined;
        if (!d) return { summary: {}, warnings: [], errors: ["データセットがありません"] };
        const pr = readR2Params(ctx.params);
        if ("error" in pr) return { summary: {}, warnings: [], errors: [pr.error] };
        const cohort = binaryCohort(d, pr.positiveLabel);
        if ("error" in cohort) return { summary: {}, warnings: [], errors: [cohort.error] };
        const context: DiscrimContext = {
          y: cohort.y, labels: cohort.labels, names: d.featureNames,
          values: d.featureNames.map((_, j) => cohort.patients.map((i) => d.features[i][j])),
        };
        ctx.state.r2 = { params: pr, cohort, context, rows: [], comparisons: 0, combo: null, tree: null, treeNote: null } satisfies R2State;
        const pos = cohort.y.reduce((a, b) => a + b, 0);
        const warnings = d.ids.length - cohort.patients.length > 0 ? [`結果ラベルが欠けている患者 ${d.ids.length - cohort.patients.length} 人を除きました`] : [];
        if (pr.multiplicity === "none") warnings.push("多重比較の補正をしません（探索的な解析）");
        return { summary: { patients: cohort.patients.length, positive: pos, negative: cohort.y.length - pos, positiveLabel: pr.positiveLabel, negativeLabel: cohort.labels[0], features: d.featureNames.length }, warnings };
      },
    },
    {
      id: "features",
      title: { ja: "特徴量ごとの ROC", en: "Per-feature ROC" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r2 as R2State;
        await ctx.tick("ROC", 0);
        const { rows, comparisons } = analyzeFeatures(st.context, st.params);
        st.rows = rows;
        st.comparisons = comparisons;
        const failed = rows.filter((r) => r.error !== null).length;
        return {
          summary: { features: rows.length, comparisons, failed, selected: rows.filter((r) => meetsCriterion(r, st.params)).length },
          warnings: failed ? [`${failed} 個の特徴量は解析できませんでした（理由は結果の表にあります。AUC・p は空欄のままで、0 にはしていません）`] : [],
        };
      },
    },
    {
      id: "combination",
      title: { ja: "組み合わせと LOO 検証", en: "Combination and leave-one-out" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r2 as R2State;
        const c = await combination(st.context, st.rows, st.params, ctx.tick);
        st.combo = c;
        if (c.error) return { summary: {}, warnings: c.notes, errors: [c.error] };
        return {
          summary: { features: c.features.join(" + "), nUsed: c.nUsed, apparentAuc: c.apparent?.auc ?? null, looAuc: c.loo?.auc ?? null, looCohort: c.looCohort, looFailed: c.looFailed },
          warnings: c.notes, skipped: c.features.length === 0,
        };
      },
    },
    {
      id: "tree",
      title: { ja: "決定木", en: "Decision tree" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r2 as R2State;
        if (!st.params.tree) return { summary: {}, warnings: [], skipped: true };
        await ctx.tick("tree", 0);
        const r = decisionTree(st.context, st.rows, st.params, ctx.rng);
        if ("note" in r) { st.treeNote = r.note; return { summary: {}, warnings: [r.note], skipped: true }; }
        st.tree = r;
        await ctx.tick("tree", 1);
        return {
          summary: { candidates: r.candidates.length, nUsed: r.nUsed, nodes: treeNodes(r.root).filter((n) => n.split).length, accuracy: r.perf.accuracy.value },
          warnings: r.notes,
        };
      },
    },
  ],
};

export interface R3State {
  params: R3Params;
  data: ConditionData;
  rows: ReliabilityRow[];
  candidates: ReliabilityRow[];
}

export const R3_REPRODUCIBILITY: Recipe = {
  id: "r3-reproducibility",
  title: { ja: "条件を変えても同じか（R3 再現性）", en: "Reproducibility across conditions (R3)" },
  question: {
    ja: "この特徴量は、ソフト・設定・再構成などの条件を変えても同じ値になるか。",
    en: "Does this feature give the same value when the software, settings or reconstruction change?",
  },
  sources: ["Fornacon-Wood I et al. Eur Radiol 2020 (PMID 32483644)"],
  defaults: { ...R3_DEFAULTS },
  steps: [
    {
      id: "conditions",
      title: { ja: "条件の突き合わせ", en: "Match the conditions" },
      run(ctx): StepOutcome {
        const pr = readR3Params(ctx.params);
        if ("error" in pr) return { summary: {}, warnings: [], errors: [pr.error] };
        const r = buildConditions(ctx.inputs.features, pr.idColumn);
        if (!r.ok) return { summary: {}, warnings: [], errors: r.errors };
        ctx.state.r3 = { params: pr, data: r.data, rows: [], candidates: [] } satisfies R3State;
        const rep = r.data.report;
        const warnings: string[] = [];
        if (rep.unmatched.length) warnings.push(`全条件に揃っていない行が ${rep.unmatched.length} 個あります（解析に使いません。一覧は結果にあります）`);
        if (rep.excludedColumns.length) warnings.push(`使えなかった列が ${rep.excludedColumns.length} 個あります（理由は結果にあります）`);
        return { summary: { conditions: rep.conditions.length, matched: rep.matched, unmatched: rep.unmatched.length, features: r.data.features.length, excludedColumns: rep.excludedColumns.length }, warnings };
      },
    },
    {
      id: "icc",
      title: { ja: "ICC と区分", en: "ICC and categories" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r3 as R3State;
        await ctx.tick("ICC", 0);
        st.rows = analyzeReliability(st.data);
        st.candidates = exclusionCandidates(st.rows, st.params.minCategory);
        const by = (c: string) => st.rows.filter((r) => r.result?.category === c).length;
        const failed = st.rows.filter((r) => r.error !== null).length;
        return {
          summary: { features: st.rows.length, poor: by("poor"), moderate: by("moderate"), good: by("good"), excellent: by("excellent"), failed, minCategory: st.params.minCategory, candidates: st.candidates.length },
          warnings: failed ? [`${failed} 個の特徴量は ICC を計算できませんでした（理由は結果の表にあります。ICC は空欄のままで、0 や 1 にはしていません）`] : [],
        };
      },
    },
  ],
};

export const RECIPES: Recipe[] = [DATASET_CHECK, R1_PROGNOSIS, R2_DISCRIMINATION, R3_REPRODUCIBILITY];
