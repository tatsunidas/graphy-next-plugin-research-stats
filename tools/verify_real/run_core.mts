/**
 * 実データ検証: src/core の runSkill と各 Skill を Node から直接呼び、結果を JSON に出す（UI は通さない）。
 * 使い方（リポジトリ直下で）:
 *   RS_VERIFY_DATA=<LGN-NSCLC-EGFR/runs> RS_VERIFY_OUT=<出力先> bash tools/verify_real/run_core.sh [run-id の正規表現]
 * 乱数シード・分割数・しきい値などの設定は、下の RUNS にすべて書いてあり、出力 JSON の params にも残る。
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { runSkill } from "../../src/core/engine";
import { Rng } from "../../src/core/rng";
import { parseCsv, toCsv } from "../../src/core/csv";
import type { TableSource } from "../../src/core/dataset";
import type { ParamValue, Skill } from "../../src/core/skill";
import { DATASET_CHECK, R1_PROGNOSIS, R2_DISCRIMINATION, R3_REPRODUCIBILITY, P4_PIPELINE, R5_MODEL, type P4State, type R1State, type R2State, type R3State, type R5State } from "../../src/core/skills";
import { r1ResultsCsv, kmPointsCsv } from "../../src/core/exportR1";
import { r2ResultsCsv } from "../../src/core/exportR2";
import { r3ResultsCsv } from "../../src/core/exportR3";
import { p4FoldsCsv, p4PredictionsCsv, p4QcCsv, p4SelectionCsv } from "../../src/core/exportP4";
import { r5ModelCsv, r5StepsCsv } from "../../src/core/exportR5";
import { datasetCsv } from "../../src/core/exportDataset";
import { blandAltmanFor } from "../../src/core/r3";

const DATA = process.env.RS_VERIFY_DATA, OUT = process.env.RS_VERIFY_OUT;
if (!DATA || !OUT) { console.error("RS_VERIFY_DATA と RS_VERIFY_OUT を設定してください"); process.exit(2); }
mkdirSync(OUT, { recursive: true });

const read = (rel: string) => readFileSync(join(DATA, rel), "utf8");
const src = (name: string, rel: string): TableSource => ({ name, sha256: null, table: parseCsv(read(rel)) });
const FEAT_A = () => src("A_default56.csv", "default56/nsclc.csv");
const FEAT_B = () => src("B_iso1mm_bw25.csv", "iso1mm_bw25/nsclc.csv");
const CLIN = "skill_verify/clinical_surv56_subject.csv";

/** 負例: 臨床の表から先頭 3 行を除き、4 行目の患者 ID を小文字にする（結合の人数・片方だけの患者が変わる）。 */
type Dmg = boolean;
type Perm = null | { kind: "outcome"; seed: number } | { kind: "survival"; seed: number };

/** 臨床の表の列を症例間で入れ替える（outcome: EGFR の列。survival: OS_days と OS_event の組をまとめて）。欠損の位置も一緒に動く。 */
function permutedClinical(perm: Perm, damage = false): TableSource {
  const t = parseCsv(read(CLIN));
  if (damage) { t.rows.splice(0, 3); t.rows[0][t.header.indexOf("SubjectCode")] = t.rows[0][t.header.indexOf("SubjectCode")].toLowerCase(); }
  if (perm) {
    const cols = perm.kind === "outcome" ? ["EGFR"] : ["OS_days", "OS_event"];
    const js = cols.map((c) => t.header.indexOf(c));
    const rng = new Rng(perm.seed);
    const idx = t.rows.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = rng.int(i + 1); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const orig = t.rows.map((r) => [...r]);
    t.rows.forEach((r, i) => js.forEach((j) => { r[j] = orig[idx[i]][j]; }));
  }
  return { name: "clinical_surv56_subject.csv", sha256: null, table: t };
}

const JOIN = { featureIdColumn: "CaseLabel", patientColumn: "SubjectCode", clinicalIdColumn: "SubjectCode" };
const SURV = { timeColumn: "OS_days", eventColumn: "OS_event" };
const SEED = 20261125;

interface Run { damage?: Dmg; id: string; skill: Skill; params: Record<string, ParamValue>; seed?: number; perm?: Perm; features?: "A" | "AB"; }
const runs: Run[] = [];
const add = (r: Run) => runs.push(r);

add({ id: "dc", skill: DATASET_CHECK, params: { ...JOIN, ...SURV, outcomeColumn: "EGFR", covariates: ["Age", "Female"] } });

add({ id: "dc_neg", skill: DATASET_CHECK, params: { ...JOIN, ...SURV, outcomeColumn: "EGFR", covariates: ["Age", "Female"] }, damage: true });

// R1
const R1 = { ...JOIN, ...SURV, covariates: ["Age", "Female"], bootstrapB: 500, permutations: 1000 };
add({ id: "r1_median", skill: R1_PROGNOSIS, params: { ...R1, groupMode: "median" } });
add({ id: "r1_tertile", skill: R1_PROGNOSIS, params: { ...R1, groupMode: "tertile", bootstrapB: 0 } });
add({ id: "r1_optimal", skill: R1_PROGNOSIS, params: { ...R1, groupMode: "optimal", bootstrapB: 0 } });
// 既定の alpha=0.05 では q<alpha が 34 個になり、変数の上限（⌊15 ÷ 10⌋ = 1。共変量 2 個で埋まる）のため特徴量は多変量に入らない。
// 共変量なし・alpha=0.03 では、上限の 1 変数に先頭の特徴量が入り、多変量 Cox と補正 C-index（上限つきの選択のやり直し）を参照と突き合わせられる。
add({ id: "r1_median_a03", skill: R1_PROGNOSIS, params: { ...R1, covariates: [], groupMode: "median", alpha: 0.03 } });

// R2
const R2 = { ...JOIN, outcomeColumn: "EGFR", positiveLabel: "Mutant", covariates: [] as string[] };
add({ id: "r2_acc", skill: R2_DISCRIMINATION, params: { ...R2, criterion: "accuracy" } });
add({ id: "r2_youden", skill: R2_DISCRIMINATION, params: { ...R2, criterion: "youden" } });
// BH のままでは q<0.05 が 0 個で組み合わせが作られない。BH 補正の LOO（各折で q<alpha を再計算）の経路を通すため、alpha=0.45 に上げた設定（統計的な意味は無い。経路の検査用）
add({ id: "r2_acc_a45", skill: R2_DISCRIMINATION, params: { ...R2, criterion: "accuracy", alpha: 0.45, topK: 2 } });
add({ id: "r2_acc_none", skill: R2_DISCRIMINATION, params: { ...R2, criterion: "accuracy", multiplicity: "none", alpha: 0.01, topK: 3 } });

// R3
add({ id: "r3", skill: R3_REPRODUCIBILITY, params: {}, features: "AB" });

// P4
const P4B = { ...JOIN, outcomeColumn: "EGFR", outcomeKind: "binary", positiveLabel: "Mutant", lockMode: "none", outerK: 5, innerK: 3, repeats: 2, kCandidates: "1, 2, 4", missing: "impute", correlationThreshold: null };
const P4S = { ...JOIN, ...SURV, outcomeKind: "survival", stratifySurvival: true, lockMode: "none", outerK: 5, innerK: 3, repeats: 2, kCandidates: "1, 2, 4", missing: "impute", correlationThreshold: null };
const PRE = { dropConstant: true, correlationThreshold: 0.9 };
const PDROP = { preUnpairedOn: true, preUnpairedAlpha: 0.05, preUnpairedThr: 0.2 };
for (const [tag, m] of [["topk", { model: "topk" }], ["lasso", { model: "lasso", lassoRule: "min", lassoNLambda: 50 }]] as const) {
  add({ id: `p4b_${tag}`, skill: P4_PIPELINE, params: { ...P4B, ...m }, seed: SEED });
  add({ id: `p4b_${tag}_again`, skill: P4_PIPELINE, params: { ...P4B, ...m }, seed: SEED });
  add({ id: `p4b_${tag}_pre`, skill: P4_PIPELINE, params: { ...P4B, ...m, ...PRE, ...PDROP }, seed: SEED });
  add({ id: `p4s_${tag}`, skill: P4_PIPELINE, params: { ...P4S, ...m }, seed: SEED });
  add({ id: `p4s_${tag}_again`, skill: P4_PIPELINE, params: { ...P4S, ...m }, seed: SEED });
  add({ id: `p4s_${tag}_pre`, skill: P4_PIPELINE, params: { ...P4S, ...m, ...PRE }, seed: SEED });
  for (let k = 1; k <= 20; k++) {
    add({ id: `p4b_${tag}_permB${k}`, skill: P4_PIPELINE, params: { ...P4B, ...m }, seed: SEED + k, perm: { kind: "outcome", seed: 1000 + k } });
    add({ id: `p4b_${tag}_pre_permB${k}`, skill: P4_PIPELINE, params: { ...P4B, ...m, ...PRE, ...PDROP }, seed: SEED + k, perm: { kind: "outcome", seed: 1000 + k } });
    add({ id: `p4s_${tag}_permS${k}`, skill: P4_PIPELINE, params: { ...P4S, ...m }, seed: SEED + k, perm: { kind: "survival", seed: 2000 + k } });
    add({ id: `p4s_${tag}_pre_permS${k}`, skill: P4_PIPELINE, params: { ...P4S, ...m, ...PRE }, seed: SEED + k, perm: { kind: "survival", seed: 2000 + k } });
  }
}

// R5（ブートストラップ回数は実行時間のため 2000 から絞る。出力 JSON の params に残る）
const R5 = { ...JOIN, ...SURV, r5Corr: 0.9, r5BootstrapB: 200 };
add({ id: "r5_lasso", skill: R5_MODEL, params: { ...R5, r5Screen: "lasso", r5LassoRule: "min" }, seed: SEED });
add({ id: "r5_lasso_again", skill: R5_MODEL, params: { ...R5, r5Screen: "lasso", r5LassoRule: "min" }, seed: SEED });
add({ id: "r5_boost", skill: R5_MODEL, params: { ...R5, r5Screen: "boost", r5BoostNu: 0.1 }, seed: SEED });
add({ id: "r5_boost_again", skill: R5_MODEL, params: { ...R5, r5Screen: "boost", r5BoostNu: 0.1 }, seed: SEED });
for (let k = 1; k <= 10; k++) {
  add({ id: `r5_boost_permS${k}`, skill: R5_MODEL, params: { ...R5, r5Screen: "boost", r5BoostNu: 0.1 }, seed: SEED + k, perm: { kind: "survival", seed: 2000 + k } });
  if (k > 5) continue;
  add({ id: `r5_lasso_permS${k}`, skill: R5_MODEL, params: { ...R5, r5Screen: "lasso", r5LassoRule: "min" }, seed: SEED + k, perm: { kind: "survival", seed: 2000 + k } });
}

const filter = process.argv[2] ? new RegExp(process.argv[2]) : null;
const num = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : null);

for (const r of runs) {
  if (filter && !filter.test(r.id)) continue;
  const t0 = Date.now();
  const inputs = { features: r.features === "AB" ? [FEAT_A(), FEAT_B()] : [FEAT_A()], ...(r.skill === R3_REPRODUCIBILITY ? {} : { clinical: permutedClinical(r.perm ?? null, r.damage ?? false) }) };
  const seed = r.seed ?? SEED;
  const { record, state } = await runSkill(r.skill, inputs, r.params, { seed, pluginVersion: "verify", now: () => "2000-01-01T00:00:00Z" });
  const tables: Record<string, string> = {};
  const extra: Record<string, unknown> = {};
  if (r.skill === DATASET_CHECK || state.dataset) {
    if (state.dataset) tables.dataset = datasetCsv(state.dataset as never, "SubjectCode");
    extra.report = state.report ?? null;
  }
  if (record.status === "ok" || state.r1 || state.r2 || state.r3 || state.p4 || state.r5) {
    if (state.r1) {
      const s = state.r1 as R1State;
      tables.results = r1ResultsCsv(s.rows);
      tables.km = kmPointsCsv(s.context, s.rows);
      extra.r1 = { patients: s.cohort.patients, model: s.model ? { selected: s.model.selected, screened: s.model.screened, notEntered: s.model.notEntered, maxVars: s.model.maxVars, covariates: s.model.covariates, nUsed: s.model.nUsed, eventsUsed: s.model.eventsUsed, terms: s.model.terms, addedValue: s.model.addedValue, cIndexApparent: s.model.cIndexApparent, corrected: s.model.corrected, notes: s.model.notes, error: s.model.error, loglik: s.model.fit?.loglik ?? null } : null, comparisons: s.comparisons };
    }
    if (state.r2) {
      const s = state.r2 as R2State;
      tables.results = r2ResultsCsv(s.rows);
      const c = s.combo;
      extra.r2 = { patients: s.cohort.patients, y: s.cohort.y, comparisons: s.comparisons, combo: c ? { features: c.features, nUsed: c.nUsed, positives: c.positives, terms: c.terms, apparent: c.apparent, loo: c.loo, looCohort: c.looCohort, looFailed: c.looFailed, candidates: c.candidates, notes: c.notes, error: c.error, looPerformance: c.looPerformance } : null };
    }
    if (state.r3) {
      const s = state.r3 as R3State;
      tables.results = r3ResultsCsv(s.rows);
      const ba: Record<string, unknown> = {};
      for (const f of s.data.features) ba[f] = blandAltmanFor(s.data, f, 0, 1);
      for (const k of Object.keys(ba)) { const b = ba[k] as { points?: unknown }; if (b && typeof b === "object" && "points" in b) delete b.points; }
      extra.r3 = { conditions: s.data.conditions, matched: s.data.report.matched, excludedColumns: s.data.report.excludedColumns, ba };
    }
    if (state.p4) {
      const s = state.p4 as P4State;
      tables.folds = p4FoldsCsv(s); tables.pred = p4PredictionsCsv(s); tables.selection = p4SelectionCsv(s); tables.qc = p4QcCsv(s);
      extra.p4 = { ids: s.cohort.ids, dev: s.qc?.dev ?? null, outerAssign: s.cv?.outerAssign ?? null, pooled: s.cv?.pooled ?? null, summary: s.cv?.summary ?? null, finalFeatures: s.final?.model ? s.final.model.cols.map((j) => s.names[j]) : null, namesKept: s.names };
    }
    if (state.r5) {
      const s = state.r5 as R5State;
      tables.model = r5ModelCsv(s); tables.steps = r5StepsCsv(s);
      const m = s.model;
      extra.r5 = m ? { patients: s.cohort.patients, ids: s.ids, selected: m.result.selected.map((j) => m.names[j]), candidates: m.result.candidates.map((j) => m.names[j]), terms: m.terms, cApparent: num(m.cApparent), corrected: m.corrected, droppedMissing: m.droppedMissing, nUsedFeatures: m.used.length, steps: m.result.steps.map((x) => ({ feature: m.names[x.col], action: x.action, p: x.p })), maxVars: s.params.maxVars, pIn: s.params.pIn, pOut: s.params.pOut, notes: m.notes } : null;
    }
  }
  const out = { id: r.id, skillId: r.skill.id, seed, perm: r.perm ?? null, params: record.params, changedFromDefault: record.changedFromDefault, status: record.status, steps: record.steps, extra, tables, elapsedMs: Date.now() - t0 };
  writeFileSync(join(OUT, `${r.id}.json`), JSON.stringify(out));
  const errs = record.steps.flatMap((x) => x.errors);
  console.log(`${r.id}\t${record.status}\t${((Date.now() - t0) / 1000).toFixed(1)}s${errs.length ? "\t" + errs.join(" / ") : ""}`);
}
void toCsv;
