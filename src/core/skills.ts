/**
 * 組み込みのスキル。この段（基盤）は「データセットの確認」だけで、統計の手法は持たない。
 * 結合・集約・変化率の結果を確かめ、結合済みのデータセットを書き出すために使う。
 */
import { buildDataset, type Dataset, type JoinReport, type RowFilter } from "./dataset";
import { checkPatientSplit, checkRowUse, idLikeColumns, targetLeaks, type RowUseCheck, type SplitCheck } from "./leak";
import { RowTrace } from "./pipeline/rowTrace";
import { buildDelta, deltaToTable, type DeltaOrder, type DeltaResult } from "./delta";
import type { ParamValue, Skill, StepContext, StepOutcome } from "./skill";
import { pairedFromDelta, preActive, preDetail, preSummary, readPre, runPre, PRE_PAIRED_DEFAULTS, PRE_QC_DEFAULTS, PRE_ASSUMP_DEFAULTS, type PairedMatrix, type PreCaps, type PreInfo } from "./pipeline/pre";
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
import {
  buildCohort, checkSpec, evaluateTest, exclusionNames, featuresFingerprint, finalModel, fitCount, nestedCv, P4_DEFAULTS, qcAndMissing, readP4Params, specOf, splitCohort, subMatrix,
  type FinalModel, type NestedCvResult, type P4Cohort, type P4Params, type QcStepResult, type SplitResult, type TestResult,
} from "./p4";
import { gridOf, gridText, hyperText, type NestedCvSpec, type PDropEntry } from "./pipeline/nestedCv";
import { testCounts } from "./stats/assumptionReport";
import { r5Model, R5_DEFAULTS, readR5Params, type R5Model, type R5Params } from "./r5";

const strList = (v: ParamValue): string[] | null => (Array.isArray(v) && v.every((x) => typeof x === "string") ? (v as string[]) : null);
const rowFilterOf = (v: ParamValue): RowFilter[] => (Array.isArray(v) ? (v.filter((x) => typeof x === "object") as RowFilter[]) : []);
export const filterText = (f: readonly RowFilter[]) => f.map((x) => `${x.column}=${x.values.join("|")}`).join(";");

const str = (v: ParamValue, name: string): string | null => {
  if (v === null || v === "") return null;
  if (typeof v !== "string") throw new Error(`設定「${name}」は文字列にしてください`);
  return v;
};

/** 前処理の見出し（記録の手順名）。 */
const PRE_TITLE = { ja: "前処理", en: "Preprocessing" };

/**
 * R1・R2・R5 の前処理（患者ごとの特徴量の行列に対して、解析の前に行う）。何も有効でなければ null（何も読まない）。
 * values は [特徴量][患者]、patients は dataset の患者の番号（delta の前後の値を引くのに使う）。
 */
function cohortPre(ctx: StepContext, caps: PreCaps, names: string[], values: (number | null)[][], patients: number[], groups: { rows0: number[]; rows1: number[] } | null, qc = true): { keep: number[]; info: PreInfo } | { error: string } | null {
  const spec = readPre(ctx.params, caps);
  if ("error" in spec) return spec;
  if (!preActive(spec)) return null;
  const X = patients.map((_, i) => names.map((_, j) => values[j][i]));
  let pm: PairedMatrix | null = null;
  if (spec.paired) {
    const delta = ctx.state.delta as DeltaResult | undefined, d = ctx.state.dataset as Dataset | undefined;
    if (!delta || !d) return { error: "対応ありの p 値による除外には、前と後の表が要ります" };
    const r = pairedFromDelta(d, delta, patients);
    if ("error" in r) return r;
    pm = r;
  }
  return runPre({ names, X, rows: patients.map((_, i) => i), spec, paired: pm, groups, qc, wholeCohort: true });
}

/** 前処理で特徴量が減ったことの注記（多重比較の補正の対象の数）。 */
const mNote = (info: PreInfo, corrected: boolean) =>
  `前処理で特徴量を ${info.before - info.after} 個除きました（${info.before} → ${info.after}。除いた特徴量と理由は結果と記録にあります）${corrected ? `。多重比較の補正の対象の数 m は、この除外の後の特徴量で数えます` : ""}`;

export const DATASET_CHECK: Skill = {
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
    rowFilter: [],
    excludeFeatureColumns: [],
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
        const featureColumns = strList(ctx.params.featureColumns);
        const r = buildDelta({
          baseline, followup, idColumn: str(ctx.params.featureIdColumn, "featureIdColumn")!,
          patientColumn: str(ctx.params.patientColumn, "patientColumn"),
          featureColumns, order,
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
        if (!ctx.inputs.clinical) return { summary: {}, warnings: [], errors: ["臨床の表がありません（このスキルには臨床の CSV が要ります）"] };
        const idCol = str(p.featureIdColumn, "featureIdColumn")!;
        // 変化率があるときは、患者ごとの変化率の表を特徴量として結合する（1 行 = 1 患者）
        const features = delta
          ? [{ name: "delta", sha256: null, table: deltaToTable(delta, idCol) }]
          : ctx.inputs.features;
        const r = buildDataset({
          features, featureIdColumn: idCol, patientColumn: delta ? null : str(p.patientColumn, "patientColumn"),
          featureColumns: delta ? null : strList(p.featureColumns),
          excludeColumns: delta ? [] : (strList(p.excludeFeatureColumns) ?? []),
          rowFilter: delta ? [] : rowFilterOf(p.rowFilter),
          clinical: ctx.inputs.clinical, clinicalIdColumn: str(p.clinicalIdColumn, "clinicalIdColumn")!,
          roles: {
            time: str(p.timeColumn, "timeColumn"), event: str(p.eventColumn, "eventColumn"),
            outcome: str(p.outcomeColumn, "outcomeColumn"), covariates: strList(p.covariates) ?? [],
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
            rowFilter: delta || !rowFilterOf(p.rowFilter).length ? null : filterText(rowFilterOf(p.rowFilter)), filteredRows: rep.filteredRows,
            averagedPatients: rep.averagedPatients, averagedRows: rep.averagedRows,
            lesionsMin: rep.lesionsPerPatient?.min ?? null, lesionsMax: rep.lesionsPerPatient?.max ?? null,
            excludedFeatureColumns: delta ? null : (strList(p.excludeFeatureColumns) ?? []).join(", ") || null,
          },
          warnings: rep.warnings,
        };
      },
    },
    {
      id: "leak",
      title: { ja: "データリークの検査（列）", en: "Data leakage check (columns)" },
      run(ctx): StepOutcome {
        const d = ctx.state.dataset as Dataset | undefined;
        if (!d) return { summary: {}, warnings: [], skipped: true };
        const warnings: string[] = [];
        let idLike: string[] = [];
        if (!ctx.state.delta) {
          const p = ctx.params;
          idLike = idLikeColumns(ctx.inputs.features.map((f) => f.table), [str(p.featureIdColumn, "featureIdColumn"), str(p.patientColumn, "patientColumn")], d.featureNames)
            .filter((c) => c.used).map((c) => c.column);
        }
        if (idLike.length) warnings.push(`ID・由来・記録に当たる列が特徴量に入っています: ${idLike.slice(0, 8).join(", ")}${idLike.length > 8 ? " …" : ""}`);
        const tl = targetLeaks(d);
        if (tl.length) warnings.push(`目的変数の混入の疑いがある特徴量があります: ${tl.slice(0, 5).map((x) => `${x.feature}（${x.target}: ${x.kinds.join("・")}）`).join(", ")}${tl.length > 5 ? " …" : ""}`);
        return {
          summary: {
            idLikeUsed: idLike.length, idLikeColumns: idLike.join(", ") || null,
            targetLeaks: tl.length, targetLeakDetail: tl.map((x) => `${x.feature}:${x.target}:${x.kinds.join("+")}`).join("; ") || null,
          },
          warnings,
        };
      },
    },
  ],
};

export interface R1State {
  /** 前処理の結果（何も有効にしなければ null）。 */
  pre: PreInfo | null;
  params: R1Params;
  cohort: SurvivalCohort;
  context: FeatureContext;
  covariates: CovariateData;
  rows: FeatureResult[];
  comparisons: number;
  model: MultivariateResult | null;
}

export const R1_PROGNOSIS: Skill = {
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
  defaults: { ...DATASET_CHECK.defaults, ...R1_DEFAULTS, ...PRE_QC_DEFAULTS, ...PRE_PAIRED_DEFAULTS },
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
        ctx.state.r1 = { pre: null, params: pr, cohort, context, covariates: cov, rows: [], comparisons: 0, model: null } satisfies R1State;
        const events = cohort.surv.event.reduce((a, b) => a + b, 0);
        const warnings = [];
        if (d.ids.length - cohort.patients.length > 0) warnings.push(`時間かイベントが欠けている患者 ${d.ids.length - cohort.patients.length} 人を除きました`);
        if (pr.groupMode === "optimal") warnings.push("最適しきい値は探索的です。素の p は多重比較で小さく出るため、並べ替え検定の補正 p を使います");
        if (pr.multiplicity === "none") warnings.push("多重比較の補正をしません（探索的な解析）");
        return { summary: { patients: cohort.patients.length, events, features: d.featureNames.length, covariates: cov.names.length }, warnings };
      },
    },
    {
      id: "pre",
      title: PRE_TITLE,
      run(ctx): StepOutcome {
        const st = ctx.state.r1 as R1State;
        const r = cohortPre(ctx, { qc: true, paired: true, unpaired: false, assumption: false }, st.context.names, st.context.values, st.cohort.patients, null);
        if (r === null) return { summary: {}, warnings: [], skipped: true };
        if ("error" in r) return { summary: {}, warnings: [], errors: [r.error] };
        st.pre = r.info;
        st.context = { ...st.context, names: r.keep.map((j) => st.context.names[j]), values: r.keep.map((j) => st.context.values[j]) };
        return { summary: preSummary(r.info), warnings: r.info.removed.length ? [mNote(r.info, st.params.multiplicity === "bh")] : [], detail: preDetail(r.info) };
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
  /** 前処理の結果（何も有効にしなければ null）。 */
  pre: PreInfo | null;
  params: R2Params;
  cohort: BinaryCohort;
  context: DiscrimContext;
  rows: DiscrimRow[];
  comparisons: number;
  combo: ComboResult | null;
  tree: TreeResult | null;
  treeNote: string | null;
}

export const R2_DISCRIMINATION: Skill = {
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
  defaults: { ...DATASET_CHECK.defaults, ...R2_DEFAULTS, ...PRE_QC_DEFAULTS, ...PRE_PAIRED_DEFAULTS, ...PRE_ASSUMP_DEFAULTS },
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
        ctx.state.r2 = { pre: null, params: pr, cohort, context, rows: [], comparisons: 0, combo: null, tree: null, treeNote: null } satisfies R2State;
        const pos = cohort.y.reduce((a, b) => a + b, 0);
        const warnings = d.ids.length - cohort.patients.length > 0 ? [`結果ラベルが欠けている患者 ${d.ids.length - cohort.patients.length} 人を除きました`] : [];
        if (pr.multiplicity === "none") warnings.push("多重比較の補正をしません（探索的な解析）");
        return { summary: { patients: cohort.patients.length, positive: pos, negative: cohort.y.length - pos, positiveLabel: pr.positiveLabel, negativeLabel: cohort.labels[0], features: d.featureNames.length }, warnings };
      },
    },
    {
      id: "pre",
      title: PRE_TITLE,
      run(ctx): StepOutcome {
        const st = ctx.state.r2 as R2State;
        const y = st.context.y;
        const groups = { rows0: y.map((_, i) => i).filter((i) => y[i] === 0), rows1: y.map((_, i) => i).filter((i) => y[i] === 1) };
        const r = cohortPre(ctx, { qc: true, paired: true, unpaired: false, assumption: true }, st.context.names, st.context.values, st.cohort.patients, groups);
        if (r === null) return { summary: {}, warnings: [], skipped: true };
        if ("error" in r) return { summary: {}, warnings: [], errors: [r.error] };
        st.pre = r.info;
        st.context = { ...st.context, names: r.keep.map((j) => st.context.names[j]), values: r.keep.map((j) => st.context.values[j]) };
        return { summary: preSummary(r.info), warnings: r.info.removed.length ? [mNote(r.info, st.params.multiplicity === "bh")] : [], detail: preDetail(r.info) };
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

export const R3_REPRODUCIBILITY: Skill = {
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

export interface P4State {
  params: P4Params;
  cohort: P4Cohort;
  split: SplitResult | null;
  qc: QcStepResult | null;
  /** 品質管理で残した列の名前と、その列だけの行列（対象の症例の並び）。 */
  names: string[];
  X: (number | null)[][];
  spec: NestedCvSpec | null;
  cv: NestedCvResult | null;
  final: FinalModel | null;
  test: TestResult | null;
  /** リークの検査（患者の分割・前処理の行）。最終モデルの後に作る。 */
  leak: { split: SplitCheck; rowUse: RowUseCheck } | null;
}

const WHERE_JA = { pre: "交差検証の前", fold: "外側の分割", inner: "内側の分割", final: "最終モデル", finalInner: "最終モデルの内側の分割" } as const;

/** p 値による除外の記録を、外側の分割ごとの要約にする（平らな値）。 */
function pdropSummary(log: PDropEntry[] | null): Record<string, string | number | null> {
  if (!log) return {};
  const outer = log.filter((e) => /^r\d+f\d+$/.test(e.scope)), inner = log.filter((e) => e.scope.includes(".i"));
  const mean = (xs: PDropEntry[]) => (xs.length ? Number((xs.reduce((a, e) => a + e.dropped.length, 0) / xs.length).toFixed(3)) : null);
  return { pdropOuterEntries: outer.length, pdropOuterMeanDropped: mean(outer), pdropInnerEntries: inner.length, pdropInnerMeanDropped: mean(inner) };
}

/** 記録の詳細: 訓練の行ごとに、落とした特徴量・検定の選択の件数。 */
const pdropDetail = (log: PDropEntry[]) => log.map((e) => ({ scope: e.scope, nTrain: e.nTrain, tested: e.tested, kept: e.kept, tests: e.tests, dropped: e.dropped.map((x) => ({ feature: x.feature, test: x.test, p: x.p })) }));

const fmt = (v: number | null) => (v === null ? null : Number(v.toFixed(6)));

export const P4_PIPELINE: Skill = {
  id: "p4-pipeline",
  title: { ja: "予測モデルの性能（入れ子の交差検証とテスト集合）", en: "Prediction model performance (nested cross-validation and test set)" },
  question: {
    ja: "特徴量から作ったモデルは、作るのに使っていない症例でどのくらい当たるか。",
    en: "How well does a model built from the features predict cases that were not used to build it?",
  },
  sources: [],
  defaults: { ...DATASET_CHECK.defaults, ...P4_DEFAULTS },
  steps: [
    ...DATASET_CHECK.steps,
    {
      id: "p4-cohort",
      title: { ja: "対象と結果", en: "Cohort and outcome" },
      async run(ctx): Promise<StepOutcome> {
        const d = ctx.state.dataset as Dataset | undefined;
        if (!d) return { summary: {}, warnings: [], errors: ["データセットがありません"] };
        const pr = readP4Params(ctx.params);
        if ("error" in pr) return { summary: {}, warnings: [], errors: [pr.error] };
        const c = await buildCohort(d, pr, (ctx.state.delta as DeltaResult | undefined) ?? null);
        if ("error" in c) return { summary: {}, warnings: [], errors: [c.error] };
        ctx.state.p4 = { params: pr, cohort: c, split: null, qc: null, names: [], X: [], spec: null, cv: null, final: null, test: null, leak: null } satisfies P4State;
        const left = d.ids.length - c.ids.length;
        const ev = c.task.kind === "binary" ? c.task.y.reduce((a, b) => a + b, 0) : c.task.surv.event.reduce((a, b) => a + b, 0);
        return {
          summary: { outcomeKind: pr.outcomeKind, patients: c.ids.length, positivesOrEvents: ev, features: c.names.length, cohortFingerprint: c.fingerprint },
          warnings: left ? [`結果が欠けている患者 ${left} 人を除きました`] : [],
        };
      },
    },
    {
      id: "p4-split",
      title: { ja: "テスト集合", en: "Test set" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.p4 as P4State;
        const r = await splitCohort(st.cohort, st.params, ctx.rng, ctx.inputs.lock?.text ?? null, ctx.pluginVersion, ctx.now);
        if ("error" in r) return { summary: {}, warnings: [], errors: [r.error] };
        st.split = r;
        const warnings: string[] = [];
        if (r.lockIsNew) warnings.push("テスト集合の固定ファイルを作りました。保存してください。テスト集合での評価は、保存したファイルを読み込んでから行います");
        if (st.params.lockMode === "none") warnings.push("テスト集合を作りません（交差検証の推定だけです）");
        const used = r.lock?.evaluations.length ?? 0;
        if (!r.lockIsNew && used > 0) warnings.push(`このテスト集合は既に ${used} 回評価されています`);
        return {
          summary: { lockMode: st.params.lockMode, dev: r.dev.length, test: r.test.length, lockSha256: r.lock?.sha256 ?? null, lockSeed: r.lock?.seed ?? null, evaluationsBefore: r.lock ? used : null },
          warnings,
        };
      },
    },
    {
      id: "p4-qc",
      title: { ja: "品質管理と欠損", en: "Quality control and missing values" },
      run(ctx): StepOutcome {
        const st = ctx.state.p4 as P4State;
        const sp = st.split!;
        let exclude: string[] | null = null;
        if (ctx.inputs.exclusion) {
          const ex = exclusionNames(ctx.inputs.exclusion);
          if ("error" in ex) return { summary: {}, warnings: [], errors: [ex.error] };
          exclude = ex;
        }
        const q = qcAndMissing(st.cohort, sp.dev, sp.test, st.params, exclude);
        if ("error" in q) return { summary: {}, warnings: [], errors: [q.error] };
        st.qc = q;
        st.names = q.keep.map((j) => st.cohort.names[j]);
        st.X = subMatrix(st.cohort.X, q.keep);
        const spec = specOf(st.cohort, q.dev, st.params, q.impute);
        const bad = checkSpec(spec, q.dev.length);
        if (bad) return { summary: {}, warnings: [], errors: [bad] };
        if (st.params.pre.unpaired) spec.pdrop = { alpha: st.params.pre.unpaired.alpha, threshold: st.params.pre.unpaired.threshold, names: st.names, log: [] };
        st.spec = spec;
        const warnings: string[] = [];
        if (q.paired) warnings.push(`対応ありの p 値による除外で特徴量を ${q.removed.filter((r) => r.reason.startsWith("対応あり")).length} 個除きました（一覧は結果にあります）`);
        if (q.unknownExcluded.length) warnings.push(`除外リストにあるが表に無い特徴量が ${q.unknownExcluded.length} 個あります: ${q.unknownExcluded.slice(0, 5).join(", ")}`);
        if (q.droppedDev.length) warnings.push(`欠損のある開発集合の症例 ${q.droppedDev.length} 人を除きました（一覧は結果にあります）`);
        if (st.names.length > q.dev.length) warnings.push(`特徴量の数（${st.names.length}）が開発集合の症例数（${q.dev.length}）より多く、性能の見積もりの偏りが大きくなりやすい状況です`);
        return {
          summary: { featuresBefore: st.cohort.names.length, featuresKept: st.names.length, removed: q.removed.length, pairedAlpha: st.params.pre.paired?.alpha ?? null, pairedThreshold: st.params.pre.paired?.threshold ?? null, pairedTests: q.paired ? Object.entries(q.paired.tests).map(([k, v]) => `${k}:${v}`).join(" ") : null, assumptionAlpha: q.assumptions?.alpha ?? null, assumptionTests: q.assumptions ? Object.entries(testCounts(q.assumptions.rows)).map(([k, v]) => `${k}:${v}`).join(" ") : null, excludeList: exclude ? exclude.length : null, missingCells: q.missingCells, missing: st.params.missing, droppedDev: q.droppedDev.length, dev: q.dev.length, ratio: Number((q.dev.length / st.names.length).toFixed(3)) },
          warnings,
        };
      },
    },
    {
      id: "p4-cv",
      title: { ja: "入れ子の交差検証", en: "Nested cross-validation" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.p4 as P4State;
        const trace = new RowTrace();
        st.spec!.trace = trace;
        if (st.qc!.usedRows.length) trace.note("qc", st.qc!.usedRows);
        if (st.qc!.pairedRows.length) trace.note("pdropPaired", st.qc!.pairedRows);
        const cv = await nestedCv(st.X, st.cohort.task, st.qc!.dev, st.spec!, ctx.rng, ctx.tick);
        st.cv = cv;
        const failed = cv.folds.filter((f) => f.score === null).length;
        const pooled = cv.pooled.filter((x): x is number => x !== null);
        return {
          summary: {
            outerK: st.spec!.outerK, innerK: st.spec!.innerK, repeats: st.spec!.repeats, model: st.params.model, kCandidates: st.params.kCandidates.join(","),
            lassoRule: st.params.lassoRule, lassoNLambda: st.params.model === "lasso" ? st.params.lassoNLambda : null,
            grid: st.params.ml ? gridText(st.params.ml) : null, gridSize: st.params.ml ? gridOf(st.params.ml).length : null, fits: fitCount(st.params),
            mean: fmt(cv.summary.mean), min: fmt(cv.summary.min), max: fmt(cv.summary.max), valid: cv.summary.valid, total: cv.summary.total,
            pooledMean: pooled.length ? fmt(pooled.reduce((a, b) => a + b, 0) / pooled.length) : null,
            ...pdropSummary(st.spec!.pdrop?.log.filter((e) => e.scope !== "final" && !e.scope.startsWith("final.")) ?? null),
          },
          ...(st.spec!.pdrop ? { detail: { pdrop: pdropDetail(st.spec!.pdrop.log.filter((e) => /^r\d+f\d+$/.test(e.scope))) } } : {}),
          warnings: failed ? [`${failed} 個の分割は評価できませんでした（理由は結果の表にあります。0 や 0.5 にはしていません）`] : [],
        };
      },
    },
    {
      id: "p4-final",
      title: { ja: "最終モデル", en: "Final model" },
      run(ctx): StepOutcome {
        const st = ctx.state.p4 as P4State;
        const fm = finalModel(st.X, st.cohort.task, st.qc!.dev, st.spec!, ctx.rng);
        st.final = fm;
        const fin = st.spec!.pdrop?.log.filter((e) => e.scope === "final") ?? null;
        const pd = st.spec!.pdrop ? { detail: { pdrop: pdropDetail(fin ?? []) } } : {};
        const droppedFinal = fin?.[0]?.dropped.length ?? null;
        if (fm.error) return { summary: { k: fm.choice.k, lambda: fm.choice.lambda, params: hyperText(fm.choice.params ?? null) || null, pdropDropped: droppedFinal }, warnings: [`最終モデルを作れませんでした: ${fm.error}`], ...pd };
        const m = fm.model!;
        if (st.params.ml) {
          const top = (m.importance ?? []).map((v, a) => ({ n: st.names[m.cols[a]], v })).sort((x, y) => y.v - x.v).slice(0, 5);
          return {
            summary: { k: fm.choice.k, params: hyperText(fm.choice.params ?? null), resolved: hyperText(m.resolved ?? null), output: m.output ?? "probability", importanceTop: top.length ? top.map((x) => `${x.n}:${fmt(x.v)}`).join(" ") : null, pdropDropped: droppedFinal },
            warnings: m.output === "decision" ? ["SVM の出力は決定値で、確率ではありません（較正は該当しません）"] : [],
            ...pd,
          };
        }
        const none = m.cols.length === 0;
        return {
          summary: { k: fm.choice.k, lambda: fm.choice.lambda, features: m.cols.map((j) => st.names[j]).join(" + "), pdropDropped: droppedFinal },
          warnings: none ? ["選んだ λ では係数がすべて 0 です（特徴量を使わないモデル。予測はどの症例でも同じ値になります）"] : [],
          ...pd,
        };
      },
    },
    {
      id: "p4-leak",
      title: { ja: "データリークの検査（分割と前処理）", en: "Data leakage check (splits and preprocessing)" },
      run(ctx): StepOutcome {
        const st = ctx.state.p4 as P4State;
        const ids = st.cohort.ids;
        const split = checkPatientSplit(
          { outerK: st.spec!.outerK, innerK: st.spec!.innerK, outerAssign: st.cv!.outerAssign, innerAssign: st.cv!.innerAssign },
          st.qc!.dev.map((i) => ids[i]), (st.split?.test ?? []).map((i) => ids[i]),
        );
        const rowUse = checkRowUse(st.spec!.trace!);
        st.leak = { split, rowUse };
        const warnings: string[] = [];
        if (!split.ok) warnings.push(`同じ患者が訓練と検証に分かれています: ${split.patients.slice(0, 5).join(", ")}${split.patients.length > 5 ? " …" : ""}`);
        for (const f of rowUse.findings.filter((x) => x.violating > 0)) {
          warnings.push(f.usesOutcome
            ? `目的変数を使う処理が訓練側でない行を使っています: ${f.stage}（${WHERE_JA[f.where]}、${f.violating}/${f.entries} 件）`
            : `教師なしの処理が訓練側でない行の値を読んでいます（結果ラベルは使っていません）: ${f.stage}（${WHERE_JA[f.where]}、${f.violating}/${f.entries} 件）`);
        }
        return {
          summary: {
            splitCheck: split.ok ? "OK" : "NG", splitChecked: split.checked, splitPatients: split.patients.join(", ") || null,
            rowUseCheck: rowUse.level === "ok" ? "OK" : rowUse.level === "notice" ? "NOTICE" : "NG",
            rowUseFindings: rowUse.findings.map((f) => `${f.stage}@${f.where}:${f.violating}/${f.entries}`).join(" ") || null,
          },
          warnings,
        };
      },
    },
    {
      id: "p4-test",
      title: { ja: "テスト集合での評価", en: "Evaluation on the test set" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.p4 as P4State;
        if (!st.params.evaluateTest) return { summary: {}, warnings: [], skipped: true };
        if (st.params.lockMode !== "use" || !st.split?.lock) return { summary: {}, warnings: [], errors: ["テスト集合での評価は、保存した固定ファイルを読み込んで行います"] };
        const fp = await featuresFingerprint(st.names, st.X);
        const r = await evaluateTest(st.cohort, st.split.test, st.final!, st.split.lock, st.params, ctx.rng, fp, ctx.now);
        if ("error" in r) return { summary: {}, warnings: [], errors: [r.error] };
        st.test = r;
        const warnings: string[] = [];
        if (r.evaluation > 1) warnings.push(`このテスト集合での ${r.evaluation} 回目の評価です。報告に回数を書いてください`);
        if (r.excluded.length) warnings.push(`欠損で予測できないテスト症例 ${r.excluded.length} 人を除きました`);
        if (r.output === "decision") warnings.push("SVM の出力は確率ではないので、較正は出していません");
        return { summary: { evaluation: r.evaluation, n: r.n, value: fmt(r.value), lower: fmt(r.lower), upper: fmt(r.upper), output: r.output, lockSha256: r.lock.sha256, excluded: r.excluded.length }, warnings };
      },
    },
  ],
};

export interface R5State {
  /** 前処理の結果（対応ありの除外を有効にしたときだけ）。 */
  pre: PreInfo | null;
  /** 使う特徴量の dataset の列番号（前処理で減らす。何も除かなければ全部）。 */
  cols: number[];
  params: R5Params;
  cohort: SurvivalCohort;
  /** 対象の患者 ID（cohort.patients の並び）。 */
  ids: string[];
  names: string[];
  model: R5Model | null;
  /** LASSO の交差検証の分割の検査（LASSO で絞り込んだときだけ）。ブートストラップは元の標本で評価する方式で、分割ではない。 */
  split: SplitCheck | null;
}

export const R5_MODEL: Skill = {
  id: "r5-model",
  title: { ja: "少数の特徴で予後モデル（R5）", en: "Prognostic model with few features (R5)" },
  question: {
    ja: "少数の特徴量で、生存（または無増悪生存など）を予測するモデルを作れるか。その性能は楽観を補正するとどの程度か。",
    en: "Can a model with a few features predict survival (or progression-free survival, etc.), and how good is it after correcting for optimism?",
  },
  sources: [
    "Salazar P et al. PLoS One 2024 (PMID 39739866)",
    "Harrell FE et al. Stat Med 1996 (PMID 8668867)",
    "Grambsch PM, Therneau TM. Biometrika 1994 (doi 10.1093/biomet/81.3.515)",
  ],
  defaults: { ...DATASET_CHECK.defaults, ...R5_DEFAULTS, ...PRE_PAIRED_DEFAULTS },
  steps: [
    ...DATASET_CHECK.steps,
    {
      id: "r5-cohort",
      title: { ja: "生存解析の対象", en: "Survival cohort" },
      run(ctx): StepOutcome {
        const d = ctx.state.dataset as Dataset | undefined;
        if (!d) return { summary: {}, warnings: [], errors: ["データセットがありません"] };
        const pr = readR5Params(ctx.params);
        if ("error" in pr) return { summary: {}, warnings: [], errors: [pr.error] };
        const cohort = survivalCohort(d);
        if ("error" in cohort) return { summary: {}, warnings: [], errors: [cohort.error] };
        ctx.state.r5 = { pre: null, cols: d.featureNames.map((_, j) => j), params: pr, cohort, ids: cohort.patients.map((i) => d.ids[i]), names: d.featureNames, model: null, split: null } satisfies R5State;
        const events = cohort.surv.event.reduce((a, b) => a + b, 0);
        const warnings: string[] = [];
        if (d.ids.length - cohort.patients.length > 0) warnings.push(`時間かイベントが欠けている患者 ${d.ids.length - cohort.patients.length} 人を除きました`);
        if (pr.screen === "boost") warnings.push("Cox ブースティングは、論文で使われた R の mboost と同じ値になるとは言えません（論文に式が無く、R での照合もしていません）");
        return { summary: { patients: cohort.patients.length, events, features: d.featureNames.length }, warnings };
      },
    },
    {
      id: "pre",
      title: PRE_TITLE,
      run(ctx): StepOutcome {
        const st = ctx.state.r5 as R5State;
        const d = ctx.state.dataset as Dataset;
        const values = st.names.map((_, j) => st.cohort.patients.map((i) => d.features[i][j]));
        // 分散 0・相関は、R5 では手順の中（ブートストラップごと）で行う
        const r = cohortPre(ctx, { qc: false, paired: true, unpaired: false, assumption: false }, st.names, values, st.cohort.patients, null, false);
        if (r === null) return { summary: {}, warnings: [], skipped: true };
        if ("error" in r) return { summary: {}, warnings: [], errors: [r.error] };
        st.pre = r.info;
        st.cols = r.keep;
        st.names = r.keep.map((j) => d.featureNames[j]);
        return { summary: preSummary(r.info), warnings: r.info.removed.length ? [mNote(r.info, false)] : [], detail: preDetail(r.info) };
      },
    },
    {
      id: "r5-model",
      title: { ja: "選択・モデル・補正 C-index・比例ハザードの確認", en: "Selection, model, corrected C-index and proportional hazards" },
      async run(ctx): Promise<StepOutcome> {
        const st = ctx.state.r5 as R5State;
        const d = ctx.state.dataset as Dataset;
        const values = st.cols.map((j) => st.cohort.patients.map((i) => d.features[i][j]));
        const m = await r5Model(st.cohort.surv, values, st.names, st.params, ctx.rng, ctx.tick);
        if ("error" in m) return { summary: {}, warnings: [], errors: [m.error] };
        st.model = m;
        const r = m.result;
        const foldOf = r.lasso?.foldOf;
        if (foldOf) st.split = checkPatientSplit({ outerK: st.params.lassoFolds, innerK: 0, outerAssign: [foldOf], innerAssign: [] }, st.ids);
        const splitWarn = st.split && !st.split.ok ? [`同じ患者が LASSO の訓練と検証に分かれています: ${st.split.patients.slice(0, 5).join(", ")}`] : [];
        const nm = (cols: number[]) => cols.map((j) => st.names[j]).join(", ");
        return {
          summary: {
            featuresUsed: m.used.length, droppedMissing: m.droppedMissing.length, corrRemoved: r.corrRemoved.length,
            screen: st.params.screen, candidates: r.candidates.length, selected: nm(r.selected), nSelected: r.selected.length,
            capped: r.steps.filter((x) => x.action === "cap").length,
            lambda: r.lasso ? r.lasso.lambda : null,
            cIndexApparent: m.cApparent, cIndexCorrected: m.corrected?.corrected ?? null, optimism: m.corrected?.optimism ?? null,
            bootstraps: m.corrected?.bootstraps ?? null, bootstrapFailed: m.corrected?.failed ?? null,
            phGlobalP: m.schoenfeld?.global.p ?? null,
            splitCheck: st.split ? (st.split.ok ? "OK" : "NG") : null, splitPatients: st.split?.patients.join(", ") || null,
          },
          warnings: [...m.notes, ...splitWarn],
        };
      },
    },
  ],
};

export const SKILLS: Skill[] = [DATASET_CHECK, R1_PROGNOSIS, R2_DISCRIMINATION, R3_REPRODUCIBILITY, P4_PIPELINE, R5_MODEL];
