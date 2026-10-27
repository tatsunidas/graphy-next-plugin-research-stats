/**
 * 入れ子の交差検証。外側の各分割の訓練の行の中で、内側の交差検証により「使う特徴量の数 k」を候補から選び、
 * その k で外側の訓練の行から作り直して、外側の検証の行を予測する。
 *
 * 外側の訓練の行（内側では内側の訓練の行）だけで行うもの: 欠損の補完（中央値）、特徴量の順位付け、標準化、当てはめ。
 * 順位付けは、二値は Mann–Whitney の p、生存は単変量 Cox の Wald の p の小さい順（同じなら列の順）。
 * モデルは、二値がロジスティック回帰（罰則なし）、生存が Cox（Efron）。
 *
 * 外側は層化できる（層は呼び出し側が渡す）。内側は層化しない。
 * k の選び方: 二値は内側の予測を束ねた AUC、生存は内側の分割ごとの C-index の平均が最大の k。同じ値なら小さい k。
 * 乱数の使い方: 反復ごとに外側の分割 → 各外側の分割の内側の分割、の順。最終モデルの内側の分割はその後。
 */
import type { Rng } from "../rng";
import { auc } from "../discrim/roc";
import { logisticFit } from "../discrim/logistic";
import { coxFit } from "../survival/cox";
import { cIndex } from "../survival/cindex";
import { mannWhitney } from "../survival/mannwhitney";
import type { Surv } from "../survival/types";
import { fitStandardizer, imputeByTrain, standardizeRow, type Imputed } from "./preprocess";
import { stratifiedFolds } from "./split";

/** X の行に揃えた結果。 */
export type Task = { kind: "binary"; y: number[] } | { kind: "survival"; surv: Surv };

export interface NestedCvSpec {
  outerK: number;
  innerK: number;
  repeats: number;
  /** 使う特徴量の数の候補（1 以上の整数）。 */
  kCandidates: number[];
  /** 外側の層（dev の並び）。null なら層化しない。 */
  strata: number[] | null;
  /** true: 訓練の行の中央値で補う。false: 欠損がある列は、その訓練の行では使わない。 */
  impute: boolean;
}

/** **試験専用**の漏れ（画面からは選べない）。負例として、偏りが出ることを確かめるために使う。 */
export interface Leak {
  /** 特徴量の順位付けを、分割の前に開発集合全体で 1 回だけ行う。 */
  rankOnAll?: boolean;
}

export type Tick = (label: string, fraction: number) => Promise<void>;

export interface Trained {
  /** 実際に使った列（候補が足りなければ k より少ない）。 */
  cols: number[];
  /** 行 i の予測（二値は確率、生存は線形予測子＝リスク）。欠けた値があれば null。 */
  predict(i: number): number | null;
  /** 当てはめに使った、訓練の行の中央値（補完しないなら null）・平均・SD。 */
  medians: (number | null)[];
  means: number[];
  sds: number[];
}

const subSurv = (s: Surv, ix: number[]): Surv => ({ time: ix.map((i) => s.time[i]), event: ix.map((i) => s.event[i]) });

/** 訓練の行で特徴量を順位付けする。検定できない列は外す。 */
export function rankFeatures(imp: Imputed, task: Task, rows: number[]): number[] {
  const m = imp.usable.length;
  const ps: { j: number; p: number }[] = [];
  for (let j = 0; j < m; j++) {
    if (!imp.usable[j]) continue;
    const okRows = rows.filter((i) => Number.isFinite(imp.X[i][j]));
    if (okRows.length !== rows.length) continue;
    let p: number | null = null;
    if (task.kind === "binary") {
      const r = mannWhitney(okRows.filter((i) => task.y[i] === 1).map((i) => imp.X[i][j]), okRows.filter((i) => task.y[i] === 0).map((i) => imp.X[i][j]));
      p = "error" in r ? null : r.p;
    } else {
      const r = coxFit(subSurv(task.surv, okRows), okRows.map((i) => [imp.X[i][j]]));
      p = r.ok ? r.fit.p[0] : null;
    }
    if (p !== null && Number.isFinite(p)) ps.push({ j, p });
  }
  return ps.sort((a, b) => a.p - b.p || a.j - b.j).map((x) => x.j);
}

/** 上位 k 列で、訓練の行だけで標準化して当てはめる。 */
export function fitTopK(imp: Imputed, task: Task, train: number[], ranking: number[], k: number): Trained | { error: string } {
  const cols = ranking.slice(0, k);
  if (!cols.length) return { error: "選べる特徴量がありません" };
  const st = fitStandardizer(imp.X, train, cols);
  if ("error" in st) return st;
  const Z = train.map((i) => standardizeRow(st, imp.X[i]) as number[]);
  const base = { cols, medians: imp.medians, means: st.means, sds: st.sds };
  if (task.kind === "binary") {
    const f = logisticFit(Z, train.map((i) => task.y[i]));
    if (!f.ok) return { error: f.error };
    return { ...base, predict: (i) => { const z = standardizeRow(st, imp.X[i]); return z ? f.fit.predict(z) : null; } };
  }
  const f = coxFit(subSurv(task.surv, train), Z);
  if (!f.ok) return { error: f.error };
  const beta = f.fit.beta;
  return { ...base, predict: (i) => { const z = standardizeRow(st, imp.X[i]); return z ? z.reduce((a, v, q) => a + v * beta[q], 0) : null; } };
}

/** 行 rows の予測の評価（二値は AUC、生存は Harrell の C）。予測が欠けた行は除く。 */
export function scoreRows(task: Task, rows: number[], pred: (number | null)[]): number | null {
  const ok = rows.map((i, q) => ({ i, v: pred[q] })).filter((x): x is { i: number; v: number } => x.v !== null);
  if (task.kind === "binary") return auc({ score: ok.map((x) => x.v), label: ok.map((x) => task.y[x.i]) });
  return cIndex(subSurv(task.surv, ok.map((x) => x.i)), ok.map((x) => x.v));
}

export interface InnerChoice {
  /** 選んだ k。どの候補も評価できなければ null。 */
  k: number | null;
  /** 候補ごとの内側の評価（kCandidates の並び）。 */
  scores: (number | null)[];
  /** 訓練の行（渡した並び）ごとの内側の分割番号。 */
  folds: number[];
}

/** 訓練の行 T の中で、内側の交差検証（層化しない）により k を選ぶ。 */
export function chooseK(X: (number | null)[][], task: Task, T: number[], spec: NestedCvSpec, rng: Rng, fixedRanking: number[] | null): InnerChoice {
  const folds = stratifiedFolds(T.length, null, spec.innerK, rng);
  const K = spec.kCandidates;
  const pooled: (number | null)[][] = K.map(() => new Array<number | null>(T.length).fill(null));
  const perFold: (number | null)[][] = K.map(() => []);
  for (let f = 0; f < spec.innerK; f++) {
    const tin = T.filter((_, p) => folds[p] !== f), pos = T.map((_, p) => p).filter((p) => folds[p] === f);
    const tout = pos.map((p) => T[p]);
    const imp = imputeByTrain(X, tin, spec.impute);
    const ranking = fixedRanking ?? rankFeatures(imp, task, tin);
    K.forEach((k, a) => {
      const m = fitTopK(imp, task, tin, ranking, k);
      const pr = tout.map((i) => ("error" in m ? null : m.predict(i)));
      pos.forEach((p, q) => { pooled[a][p] = pr[q]; });
      perFold[a].push("error" in m ? null : scoreRows(task, tout, pr));
    });
  }
  const scores = K.map((_, a) => {
    if (task.kind === "binary") return scoreRows(task, T, pooled[a]);
    const v = perFold[a].filter((x): x is number => x !== null);
    return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
  });
  let best: number | null = null;
  scores.forEach((s, a) => {
    if (s === null) return;
    if (best === null || s > (scores[best] as number) + 1e-12 || (Math.abs(s - (scores[best] as number)) <= 1e-12 && K[a] < K[best])) best = a;
  });
  return { k: best === null ? null : K[best], scores, folds };
}

export interface OuterFold {
  repeat: number;
  fold: number;
  nTrain: number;
  nTest: number;
  k: number | null;
  innerScores: (number | null)[];
  /** 実際に使った列。 */
  cols: number[];
  /** 外側の検証の行での AUC / C。 */
  score: number | null;
  error: string | null;
}

export interface NestedCvResult {
  folds: OuterFold[];
  /** 反復ごとの、dev の並びでの外側の分割番号。 */
  outerAssign: number[][];
  /** 反復ごと・外側の分割ごとの、内側の分割番号（その外側の訓練の行の並び）。 */
  innerAssign: number[][][];
  /** 反復ごとの、dev の並びでの予測（外側の検証の行として予測した値）。 */
  pred: (number | null)[][];
  /** 二値のとき、反復ごとの束ねた AUC。生存は null。 */
  pooled: (number | null)[];
  /** 外側の分割の評価の平均・最小・最大（評価できた分割だけ）。 */
  summary: { mean: number | null; min: number | null; max: number | null; valid: number; total: number };
  /** 列ごとの、外側の分割で選ばれた回数。 */
  selectionCounts: number[];
}

export function validateSpec(spec: NestedCvSpec, nDev: number): string | null {
  const int = (v: number) => Number.isInteger(v);
  if (!int(spec.outerK) || spec.outerK < 2 || spec.outerK > nDev) return `外側の分割数は 2 以上 ${nDev} 以下の整数にしてください`;
  const minTrain = nDev - Math.ceil(nDev / spec.outerK);
  if (!int(spec.innerK) || spec.innerK < 2 || spec.innerK > minTrain) return `内側の分割数は 2 以上 ${minTrain} 以下の整数にしてください（外側の訓練の行の最小の人数）`;
  if (!int(spec.repeats) || spec.repeats < 1 || spec.repeats > 100) return "反復回数は 1 以上 100 以下の整数にしてください";
  if (!spec.kCandidates.length || spec.kCandidates.some((k) => !int(k) || k < 1)) return "特徴量の数の候補は 1 以上の整数で 1 つ以上入れてください";
  if (new Set(spec.kCandidates).size !== spec.kCandidates.length) return "特徴量の数の候補が重複しています";
  if (spec.strata && spec.strata.length !== nDev) return "層の長さが開発集合の人数と違います";
  return null;
}

/** dev は X の行番号（開発集合）。テスト集合の行は渡さないこと（ここでは一切見ない）。 */
export async function nestedCv(X: (number | null)[][], task: Task, dev: number[], spec: NestedCvSpec, rng: Rng, tick?: Tick, leak: Leak = {}): Promise<NestedCvResult> {
  const bad = validateSpec(spec, dev.length);
  if (bad) throw new RangeError(bad);
  const m = X[0]?.length ?? 0;
  const counts = new Array<number>(m).fill(0);
  const fixed = leak.rankOnAll ? rankFeatures(imputeByTrain(X, dev, spec.impute), task, dev) : null;
  const folds: OuterFold[] = [], outerAssign: number[][] = [], innerAssign: number[][][] = [], pred: (number | null)[][] = [], pooled: (number | null)[] = [];
  const total = spec.repeats * spec.outerK;
  for (let r = 0; r < spec.repeats; r++) {
    const assign = stratifiedFolds(dev.length, spec.strata, spec.outerK, rng);
    outerAssign.push(assign);
    const inner: number[][] = [];
    const pr = new Array<number | null>(dev.length).fill(null);
    for (let f = 0; f < spec.outerK; f++) {
      await tick?.("nested-cv", (r * spec.outerK + f) / total);
      const T = dev.filter((_, p) => assign[p] !== f), pos = dev.map((_, p) => p).filter((p) => assign[p] === f);
      const V = pos.map((p) => dev[p]);
      const ch = chooseK(X, task, T, spec, rng, fixed);
      inner.push(ch.folds);
      const row: OuterFold = { repeat: r, fold: f, nTrain: T.length, nTest: V.length, k: ch.k, innerScores: ch.scores, cols: [], score: null, error: null };
      if (ch.k === null) row.error = "内側の交差検証で、どの候補も評価できませんでした";
      else {
        const imp = imputeByTrain(X, T, spec.impute);
        const mdl = fitTopK(imp, task, T, fixed ?? rankFeatures(imp, task, T), ch.k);
        if ("error" in mdl) row.error = mdl.error;
        else {
          row.cols = mdl.cols;
          for (const j of mdl.cols) counts[j]++;
          const v = V.map((i) => mdl.predict(i));
          pos.forEach((p, q) => { pr[p] = v[q]; });
          row.score = scoreRows(task, V, v);
        }
      }
      folds.push(row);
    }
    innerAssign.push(inner);
    pred.push(pr);
    pooled.push(task.kind === "binary" ? scoreRows(task, dev, pr) : null);
  }
  await tick?.("nested-cv", 1);
  const sc = folds.map((f) => f.score).filter((x): x is number => x !== null);
  return {
    folds, outerAssign, innerAssign, pred, pooled, selectionCounts: counts,
    summary: {
      mean: sc.length ? sc.reduce((a, b) => a + b, 0) / sc.length : null,
      min: sc.length ? Math.min(...sc) : null, max: sc.length ? Math.max(...sc) : null,
      valid: sc.length, total: folds.length,
    },
  };
}

export interface FinalModel {
  choice: InnerChoice;
  model: Trained | null;
  error: string | null;
}

/**
 * 最終モデル: 開発集合全体での（入れ子でない）交差検証で k を選び、開発集合全体で作り直す。
 * 入れ子の交差検証の推定値は「この手順全体」の性能で、このモデルそのものの性能ではない。
 */
export function finalModel(X: (number | null)[][], task: Task, dev: number[], spec: NestedCvSpec, rng: Rng): FinalModel {
  const choice = chooseK(X, task, dev, spec, rng, null);
  if (choice.k === null) return { choice, model: null, error: "交差検証で、どの候補も評価できませんでした" };
  const imp = imputeByTrain(X, dev, spec.impute);
  const mdl = fitTopK(imp, task, dev, rankFeatures(imp, task, dev), choice.k);
  return "error" in mdl ? { choice, model: null, error: mdl.error } : { choice, model: mdl, error: null };
}
