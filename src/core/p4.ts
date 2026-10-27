/**
 * スキル P4「予測モデルの性能（入れ子の交差検証とテスト集合）」の計算本体。画面・skills.ts に依存しない。
 *
 * 流れ: 対象（結果の型）→ テスト集合（固定ファイルを作る／読む／作らない）→ 品質管理（開発集合だけ）
 * → 欠損の扱い → 入れ子の交差検証（開発集合）→ 最終モデル → [テスト集合で評価（固定ファイルの回数を 1 増やす）]。
 *
 * 文献に推奨値の無い設定（分割数・反復・特徴量の数の候補・テストの割合・相関のしきい値）は既定値を置かず、
 * 利用者が入れるまで実行しない。
 */
import type { LambdaRule } from "./model/lasso";
import type { Dataset, TableSource } from "./dataset";
import type { ParamValue } from "./skill";
import { Rng } from "./rng";
import { binaryCohort } from "./r2";
import { survivalCohort } from "./r1";
import { aucCi, rocPoints, type RocPoint } from "./discrim/roc";
import { calibrationBins, type CalibrationBin } from "./discrim/calibration";
import { cIndex } from "./survival/cindex";
import { qcFeatures, type QcRemoval } from "./pipeline/qc";
import { finalModel, gridOf, hyperText, nestedCv, validateMl, validateSpec, type FinalModel, type MlSpec, type NestedCvResult, type NestedCvSpec, type Task } from "./pipeline/nestedCv";
import { stratifiedHoldout } from "./pipeline/split";
import { addEvaluation, checkLock, fingerprint, parseLock, sealLock, canonicalJson, LOCK_FORMAT, LOCK_VERSION, type LockFile } from "./pipeline/lockFile";
import { sha256Hex } from "./digest";

export type OutcomeKind = "binary" | "survival";
export type LockMode = "create" | "use" | "none";
export type MissingMode = "complete" | "impute";
export type ModelKind = "topk" | "lasso" | "rf" | "svm" | "gbm";
const MODELS: ModelKind[] = ["topk", "lasso", "rf", "svm", "gbm"];

export interface P4Params {
  outcomeKind: OutcomeKind;
  positiveLabel: string | null;
  lockMode: LockMode;
  testFraction: number | null;
  stratifySurvival: boolean;
  outerK: number;
  innerK: number;
  repeats: number;
  kCandidates: number[];
  /** 上位 k 個（罰則なし）、LASSO、ランダムフォレスト・SVM・勾配ブースティング（二値だけ）。 */
  model: ModelKind;
  lassoRule: LambdaRule | null;
  lassoNLambda: number;
  /** RF・SVM・GBM の候補の格子（それ以外は null）。 */
  ml: MlSpec | null;
  dropConstant: boolean;
  correlationThreshold: number | null;
  missing: MissingMode | null;
  evaluateTest: boolean;
  calibrationBins: number;
  testBootstrapB: number;
}

export const P4_DEFAULTS: Record<string, ParamValue> = {
  outcomeKind: null,
  positiveLabel: null,
  lockMode: null,
  testFraction: null,
  stratifySurvival: null,
  outerK: null,
  innerK: null,
  repeats: null,
  kCandidates: null,
  model: "topk",
  lassoRule: null,
  lassoNLambda: 100,
  // RF・SVM・GBM: 読んだ論文に設定が無いので既定は置かない（画面に scikit-learn の既定を使用例として出す）
  rfTrees: null,
  rfMtry: null,
  svmC: null,
  svmGamma: null,
  gbmStages: null,
  gbmDepth: null,
  gbmRate: null,
  gbmSubsample: null,
  dropConstant: false,
  correlationThreshold: null,
  missing: null,
  evaluateTest: false,
  calibrationBins: 10,
  testBootstrapB: 500,
};

/** "1, 2, 4" → [1, 2, 4]。 */
export function parseKList(v: ParamValue): number[] | { error: string } {
  if (typeof v !== "string" || v.trim() === "") return { error: "特徴量の数の候補を入れてください（例: 1, 2, 4。既定値は決めていません）" };
  const xs = v.split(/[,\s、]+/).filter(Boolean).map(Number);
  if (xs.some((x) => !Number.isInteger(x) || x < 1)) return { error: "特徴量の数の候補は 1 以上の整数をカンマで区切って入れてください" };
  if (new Set(xs).size !== xs.length) return { error: "特徴量の数の候補が重複しています" };
  return [...xs].sort((a, b) => a - b);
}

/** "0.1, 1, scale" → [0.1, 1, "scale"]。words は数のかわりに書ける語。 */
export function parseList(v: ParamValue, what: string, words: string[] = []): (number | string)[] | { error: string } {
  if (typeof v === "number") return [v];
  if (typeof v !== "string" || v.trim() === "") return { error: `${what}を入れてください（既定値は決めていません）` };
  const out: (number | string)[] = [];
  for (const t of v.split(/[,\s、]+/).filter(Boolean)) {
    if (words.includes(t)) { out.push(t); continue; }
    const x = Number(t);
    if (!Number.isFinite(x)) return { error: `${what}に読めない値があります: ${t}` };
    out.push(x);
  }
  return out;
}

function readMl(p: Record<string, ParamValue>, kind: "rf" | "svm" | "gbm"): MlSpec | { error: string } {
  const num = (k: string, what: string): number | { error: string } => (typeof p[k] === "number" ? (p[k] as number) : { error: `${what}を入れてください（既定値は決めていません）` });
  let m: MlSpec;
  if (kind === "rf") {
    const trees = num("rfTrees", "木の数"), mtry = parseList(p.rfMtry, "mtry の候補", ["sqrt"]);
    for (const v of [trees, mtry]) if (typeof v === "object" && "error" in v) return v;
    m = { kind, trees: trees as number, mtry: mtry as (number | "sqrt")[] };
  } else if (kind === "svm") {
    const C = parseList(p.svmC, "C の候補"), gamma = parseList(p.svmGamma, "γ の候補", ["scale"]);
    for (const v of [C, gamma]) if ("error" in v) return v;
    m = { kind, C: C as number[], gamma: gamma as (number | "scale")[] };
  } else {
    const stages = parseList(p.gbmStages, "段数の候補"), depth = parseList(p.gbmDepth, "木の深さの候補");
    const lr = num("gbmRate", "学習率"), sub = num("gbmSubsample", "subsample");
    for (const v of [stages, depth, lr, sub]) if (typeof v === "object" && "error" in v) return v;
    m = { kind, stages: stages as number[], depth: depth as number[], learningRate: lr as number, subsample: sub as number };
  }
  const bad = validateMl(m);
  return bad ? { error: bad } : m;
}

/** 入れ子の交差検証の当てはめの回数（格子 × 内側 ＋ 1 を外側 × 反復、最終モデルの分を足す）。 */
export function fitCount(p: P4Params): number | null {
  if (!p.ml) return null;
  const g = gridOf(p.ml).length;
  return p.repeats * p.outerK * (g * p.innerK + 1) + g * p.innerK + 1;
}

export function readP4Params(p: Record<string, ParamValue>): P4Params | { error: string } {
  const req = (what: string) => ({ error: `${what}を選んでください（既定値は決めていません）` });
  if (p.outcomeKind !== "binary" && p.outcomeKind !== "survival") return req("結果の型（二値・生存）");
  if (p.lockMode !== "create" && p.lockMode !== "use" && p.lockMode !== "none") return req("テスト集合の扱い");
  const int = (k: string, lo: number, hi: number, what: string): number | { error: string } => {
    const v = p[k];
    if (v === null || v === undefined) return { error: `${what}を入れてください（既定値は決めていません）` };
    if (typeof v !== "number" || !Number.isInteger(v) || v < lo || v > hi) return { error: `${what}は ${lo} 以上 ${hi} 以下の整数にしてください` };
    return v;
  };
  const outerK = int("outerK", 2, 100000, "外側の分割数"), innerK = int("innerK", 2, 100000, "内側の分割数"), repeats = int("repeats", 1, 100, "反復回数");
  for (const v of [outerK, innerK, repeats]) if (typeof v === "object") return v;
  if (!MODELS.includes(p.model as ModelKind)) return { error: `モデルは ${MODELS.join("・")} のどれかです` };
  const model = p.model as ModelKind;
  let ks: number[] = [], lassoRule: LambdaRule | null = null, lassoNLambda = 100, ml: MlSpec | null = null;
  if (model === "rf" || model === "svm" || model === "gbm") {
    if (p.outcomeKind !== "binary") return { error: "ランダムフォレスト・SVM・勾配ブースティングは二値の結果だけです（生存のランダムサバイバルフォレストなどは、文献での使われ方を確かめてから決めるため入れていません）" };
    const r = readMl(p, model);
    if ("error" in r) return r;
    ml = r;
  } else if (model === "topk") {
    const r = parseKList(p.kCandidates);
    if ("error" in r) return r;
    ks = r;
  } else {
    if (p.lassoRule !== "min" && p.lassoRule !== "1se") return { error: "LASSO の λ の選び方（最小か 1SE）を選んでください（既定値は決めていません）" };
    lassoRule = p.lassoRule;
    const nl = p.lassoNLambda;
    if (typeof nl !== "number" || !Number.isInteger(nl) || nl < 2 || nl > 1000) return { error: "λ の点の数は 2 以上 1000 以下の整数にしてください" };
    lassoNLambda = nl;
  }
  let testFraction: number | null = null;
  if (p.lockMode === "create") {
    if (typeof p.testFraction !== "number" || !(p.testFraction > 0 && p.testFraction < 1)) return { error: "テスト集合の割合を 0 より大きく 1 より小さい値で入れてください（既定値は決めていません）" };
    testFraction = p.testFraction;
  }
  if (p.outcomeKind === "binary" && (typeof p.positiveLabel !== "string" || p.positiveLabel === "")) return { error: "陽性とするラベルを選んでください（既定は決めていません）" };
  if (p.outcomeKind === "survival" && typeof p.stratifySurvival !== "boolean") return { error: "生存では、外側の分割をイベントの有無で層化するかを選んでください（文献に決まった方法が無いため既定は決めていません）" };
  const ct = p.correlationThreshold;
  if (ct !== null && (typeof ct !== "number" || !(ct > 0 && ct <= 1))) return { error: "相関のしきい値は 0 より大きく 1 以下にしてください" };
  if (p.missing !== null && p.missing !== "complete" && p.missing !== "impute") return { error: "欠損の扱いは complete か impute です" };
  const cb = p.calibrationBins, bb = p.testBootstrapB;
  if (typeof cb !== "number" || !Number.isInteger(cb) || cb < 2 || cb > 50) return { error: "較正の区間の数は 2 以上 50 以下の整数にしてください" };
  if (typeof bb !== "number" || !Number.isInteger(bb) || bb < 100 || bb > 100000) return { error: "テストのブートストラップ回数は 100 以上 100000 以下の整数にしてください" };
  return {
    outcomeKind: p.outcomeKind, positiveLabel: p.outcomeKind === "binary" ? (p.positiveLabel as string) : null, lockMode: p.lockMode, testFraction,
    stratifySurvival: p.stratifySurvival === true, outerK: outerK as number, innerK: innerK as number, repeats: repeats as number, kCandidates: ks, model, lassoRule, lassoNLambda, ml,
    dropConstant: p.dropConstant === true, correlationThreshold: ct as number | null, missing: (p.missing as MissingMode | null) ?? null,
    evaluateTest: p.evaluateTest === true, calibrationBins: cb, testBootstrapB: bb,
  };
}

export interface P4Cohort {
  ids: string[];
  /** 結果（二値はラベルの文字列、生存は [時間, イベント]）。指紋に使う。 */
  outcome: unknown[];
  task: Task;
  names: string[];
  /** [症例][特徴量]。 */
  X: (number | null)[][];
  labels: [string, string] | null;
  fingerprint: string;
}

export async function buildCohort(d: Dataset, p: P4Params): Promise<P4Cohort | { error: string }> {
  if (p.outcomeKind === "binary") {
    const c = binaryCohort(d, p.positiveLabel as string);
    if ("error" in c) return c;
    const ids = c.patients.map((i) => d.ids[i]);
    const outcome = c.y.map((v) => c.labels[v]);
    return { ids, outcome, task: { kind: "binary", y: c.y }, names: d.featureNames, X: c.patients.map((i) => d.features[i]), labels: c.labels, fingerprint: await fingerprint(ids, outcome) };
  }
  const c = survivalCohort(d);
  if ("error" in c) return c;
  const ids = c.patients.map((i) => d.ids[i]);
  const outcome = c.surv.time.map((t, q) => [t, c.surv.event[q]]);
  return { ids, outcome, task: { kind: "survival", surv: c.surv }, names: d.featureNames, X: c.patients.map((i) => d.features[i]), labels: null, fingerprint: await fingerprint(ids, outcome) };
}

export const strataOf = (task: Task): number[] => (task.kind === "binary" ? task.y : task.surv.event);

export interface SplitResult {
  lock: LockFile | null;
  /** このデータで新しく作った固定ファイルなら true（保存してから使う）。 */
  lockIsNew: boolean;
  dev: number[];
  test: number[];
}

/**
 * テスト集合を決める。乱数はシードから 1 つ取り、ホールドアウトはそれをシードにした別の乱数で行う
 * （「作る」「読む」「作らない」のどれでも、交差検証が同じシードで同じ分割になるように）。
 */
export async function splitCohort(c: P4Cohort, p: P4Params, rng: Rng, lockText: string | null, pluginVersion: string, now: string): Promise<SplitResult | { error: string }> {
  const splitSeed = rng.nextU32();
  const all = c.ids.map((_, i) => i);
  if (p.lockMode === "none") return { lock: null, lockIsNew: false, dev: all, test: [] };
  if (p.lockMode === "create") {
    const strat = p.outcomeKind === "binary" || p.stratifySurvival ? strataOf(c.task) : null;
    const isTest = stratifiedHoldout(c.ids.length, strat, p.testFraction as number, new Rng(splitSeed));
    const test = all.filter((i) => isTest[i]), dev = all.filter((i) => !isTest[i]);
    if (!test.length || !dev.length) return { error: "テスト集合か開発集合が空になります。割合を変えてください" };
    const lock = await sealLock({
      format: LOCK_FORMAT, version: LOCK_VERSION, createdAt: now, pluginVersion, outcomeKind: p.outcomeKind, seed: splitSeed,
      testFraction: p.testFraction as number, stratifiedBy: strat ? (p.outcomeKind === "binary" ? "outcome" : "event") : null,
      cohortFingerprint: c.fingerprint, cohortSize: c.ids.length, testIds: test.map((i) => c.ids[i]).sort(), evaluations: [],
    });
    return { lock, lockIsNew: true, dev, test };
  }
  if (lockText === null) return { error: "テスト集合の固定ファイルを読み込んでください" };
  const lock = await parseLock(lockText);
  if ("error" in lock) return lock;
  const bad = checkLock(lock, p.outcomeKind, c.fingerprint, c.ids);
  if (bad) return { error: bad };
  const ts = new Set(lock.testIds);
  return { lock, lockIsNew: false, dev: all.filter((i) => !ts.has(c.ids[i])), test: all.filter((i) => ts.has(c.ids[i])) };
}

/** R3 の除外候補 CSV などから、"feature" 列の名前を読む。 */
export function exclusionNames(src: TableSource): string[] | { error: string } {
  const j = src.table.header.indexOf("feature");
  if (j < 0) return { error: `${src.name}: 「feature」列がありません（再現性のスキルの除外候補 CSV を使ってください）` };
  return src.table.rows.map((r) => String(r[j] ?? "").trim()).filter((x) => x !== "");
}

export interface QcStepResult {
  keep: number[];
  removed: QcRemoval[];
  unknownExcluded: string[];
  /** 欠損で外した開発集合の症例（complete のとき）。 */
  droppedDev: string[];
  dev: number[];
  impute: boolean;
  missingCells: number;
}

/** 品質管理（開発集合だけ）と欠損の扱い。 */
export function qcAndMissing(c: P4Cohort, dev: number[], test: number[], p: P4Params, exclude: string[] | null): QcStepResult | { error: string } {
  const q = qcFeatures(c.X, dev, c.names, { dropConstant: p.dropConstant, correlationThreshold: p.correlationThreshold, exclude });
  if (!q.keep.length) return { error: "品質管理の後に残った特徴量がありません" };
  const rows = [...dev, ...test];
  const missingCells = rows.reduce((a, i) => a + q.keep.filter((j) => c.X[i][j] === null).length, 0);
  let devOut = dev, droppedDev: string[] = [];
  if (missingCells > 0) {
    if (p.missing === null) return { error: `使う特徴量に欠損が ${missingCells} 個あります。扱い（欠損のある症例を除く／訓練の分割の中央値で補う）を選んでください。0 では補いません` };
    if (p.missing === "complete") {
      devOut = dev.filter((i) => q.keep.every((j) => c.X[i][j] !== null));
      droppedDev = dev.filter((i) => !devOut.includes(i)).map((i) => c.ids[i]);
    }
  }
  return { keep: q.keep, removed: q.removed, unknownExcluded: q.unknownExcluded, droppedDev, dev: devOut, impute: p.missing === "impute", missingCells };
}

export function specOf(c: P4Cohort, dev: number[], p: P4Params, impute: boolean): NestedCvSpec {
  const strat = p.outcomeKind === "binary" || p.stratifySurvival ? strataOf(c.task) : null;
  return {
    outerK: p.outerK, innerK: p.innerK, repeats: p.repeats, kCandidates: p.kCandidates, strata: strat ? dev.map((i) => strat[i]) : null, impute,
    model: p.ml ?? (p.model === "lasso" ? { kind: "lasso", nLambda: p.lassoNLambda, rule: p.lassoRule as LambdaRule } : { kind: "topk" }),
  };
}

/** 品質管理で残した列だけの行列。 */
export const subMatrix = (X: (number | null)[][], keep: number[]) => X.map((r) => keep.map((j) => r[j]));

export function checkSpec(spec: NestedCvSpec, nDev: number): string | null {
  return validateSpec(spec, nDev);
}

export { nestedCv, finalModel };
export type { NestedCvResult, FinalModel };

export interface TestResult {
  evaluation: number;
  n: number;
  /** 欠損で予測できず外したテスト症例。 */
  excluded: string[];
  /** 二値: AUC と DeLong の CI。生存: Harrell の C とブートストラップ（パーセンタイル）の CI。 */
  value: number | null;
  lower: number | null;
  upper: number | null;
  roc: RocPoint[];
  calibration: CalibrationBin[];
  /** "decision"（SVM）なら確率ではないので、較正は出さない。 */
  output: "probability" | "decision" | "risk";
  bootstrapFailed: number;
  lock: LockFile;
}

/** 特徴量（品質管理の後）の指紋。評価の履歴に残す。 */
export async function featuresFingerprint(names: string[], X: (number | null)[][]): Promise<string> {
  return (await sha256Hex(new TextEncoder().encode(canonicalJson({ names, X })))) ?? "未取得";
}

const percentile = (s: number[], q: number) => {
  const h = (s.length - 1) * q, lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
};

/** テスト集合で 1 回評価し、回数を 1 増やした固定ファイルを返す。 */
export async function evaluateTest(c: P4Cohort, test: number[], fm: FinalModel, lock: LockFile, p: P4Params, rng: Rng, featFp: string, now: string): Promise<TestResult | { error: string }> {
  if (!fm.model) return { error: fm.error ?? "最終モデルがありません" };
  const pred = test.map((i) => fm.model!.predict(i));
  const ok = test.map((i, q) => ({ i, v: pred[q] })).filter((x): x is { i: number; v: number } => x.v !== null);
  const excluded = test.filter((_, q) => pred[q] === null).map((i) => c.ids[i]);
  let value: number | null = null, lower: number | null = null, upper: number | null = null, roc: RocPoint[] = [], cal: CalibrationBin[] = [], bf = 0;
  if (c.task.kind === "binary") {
    const y = c.task.y;
    const l = { score: ok.map((x) => x.v), label: ok.map((x) => y[x.i]) };
    const ci = aucCi(l);
    if (ci) { value = ci.auc; lower = ci.lower; upper = ci.upper; }
    roc = rocPoints(l);
    if (fm.model.output !== "decision") cal = calibrationBins(l.score, l.label, p.calibrationBins);
  } else {
    const s = c.task.surv;
    const sub = (ix: number[]) => ({ time: ix.map((i) => s.time[i]), event: ix.map((i) => s.event[i]) });
    value = cIndex(sub(ok.map((x) => x.i)), ok.map((x) => x.v));
    const bs: number[] = [];
    for (let b = 0; b < p.testBootstrapB; b++) {
      const draw = ok.map(() => ok[rng.int(ok.length)]);
      const v = cIndex(sub(draw.map((x) => x.i)), draw.map((x) => x.v));
      if (v === null) bf++; else bs.push(v);
    }
    if (bs.length) { bs.sort((a, b) => a - b); lower = percentile(bs, 0.025); upper = percentile(bs, 0.975); }
  }
  const newLock = await addEvaluation(lock, { at: now, featuresFingerprint: featFp, summary: { metric: c.task.kind === "binary" ? "AUC" : "C-index", value, lower, upper, n: ok.length, k: fm.choice.k, lambda: fm.choice.lambda, ...(fm.choice.params ? { model: p.model, params: hyperText(fm.choice.params) } : {}) } });
  const output = c.task.kind === "survival" ? "risk" : fm.model.output === "decision" ? "decision" : "probability";
  return { evaluation: newLock.evaluations.length, n: ok.length, excluded, value, lower, upper, roc, calibration: cal, output, bootstrapFailed: bf, lock: newLock };
}
