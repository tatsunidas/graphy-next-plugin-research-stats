/**
 * ROC 曲線・AUC・DeLong の信頼区間・しきい値の選び方。
 * 向き: "higher" は値が大きいほど陽性らしい（陽性と予測 = 値 > しきい値）、"lower" は小さいほど陽性らしい
 * （陽性と予測 = 値 < しきい値）。しきい値は隣り合う異なる値の中点で、値そのものとは一致しない。
 */
import { normalQuantile } from "../survival/dist";
import { proportion, type Proportion } from "./binom";

export type Direction = "higher" | "lower";
export type Criterion = "accuracy" | "youden";

export interface RocPoint {
  fpr: number;
  tpr: number;
  /** この点の分類規則「値 >= threshold なら陽性」（向きが lower のときは符号を反転した値で計算している）。 */
  threshold: number;
}

const oriented = (x: number[], dir: Direction) => (dir === "higher" ? x : x.map((v) => -v));

export interface Labelled {
  score: number[];
  /** 1 が陽性、0 が陰性。 */
  label: number[];
}

export function counts(l: Labelled): { pos: number; neg: number } {
  const pos = l.label.reduce((a, b) => a + b, 0);
  return { pos, neg: l.label.length - pos };
}

/** AUC = P(陽性の値 > 陰性の値) + 0.5·P(同じ値)。Mann–Whitney の U から求める。 */
export function auc(l: Labelled, dir: Direction = "higher"): number | null {
  const s = oriented(l.score, dir), { pos, neg } = counts(l);
  if (!pos || !neg) return null;
  const order = s.map((_, i) => i).sort((a, b) => s[a] - s[b]);
  let rankSumPos = 0;
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j < order.length && s[order[j]] === s[order[i]]) j++;
    const r = (i + 1 + j) / 2;
    for (let q = i; q < j; q++) if (l.label[order[q]] === 1) rankSumPos += r;
    i = j;
  }
  return (rankSumPos - (pos * (pos + 1)) / 2) / (pos * neg);
}

/** AUC が 0.5 以上になる向きを選ぶ。 */
export function bestDirection(l: Labelled): Direction {
  const a = auc(l, "higher");
  return a !== null && a < 0.5 ? "lower" : "higher";
}

/** DeLong の AUC の信頼区間（正規近似。AUC が 0 か 1 で分散が 0 なら幅 0）。 */
export function aucCi(l: Labelled, dir: Direction = "higher", level = 0.95): { auc: number; se: number; lower: number; upper: number } | null {
  const a = auc(l, dir);
  if (a === null) return null;
  const s = oriented(l.score, dir);
  const P = s.filter((_, i) => l.label[i] === 1), N = s.filter((_, i) => l.label[i] === 0);
  const sortedN = [...N].sort((x, y) => x - y), sortedP = [...P].sort((x, y) => x - y);
  // 配置値 V10(陽性 i) = P(陰性 < x_i) + 0.5 P(陰性 = x_i)、V01(陰性 j) は逆
  const lowerCount = (arr: number[], v: number) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } return lo; };
  const upperCount = (arr: number[], v: number) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; } return lo; };
  const v10 = P.map((x) => (lowerCount(sortedN, x) + 0.5 * (upperCount(sortedN, x) - lowerCount(sortedN, x))) / N.length);
  const v01 = N.map((x) => (sortedP.length - upperCount(sortedP, x) + 0.5 * (upperCount(sortedP, x) - lowerCount(sortedP, x))) / P.length);
  const variance = (v: number[]) => { const m = v.reduce((p, q) => p + q, 0) / v.length; return v.length > 1 ? v.reduce((p, q) => p + (q - m) ** 2, 0) / (v.length - 1) : 0; };
  const se = Math.sqrt(variance(v10) / P.length + variance(v01) / N.length);
  const z = normalQuantile(1 - (1 - level) / 2);
  return { auc: a, se, lower: Math.max(0, a - z * se), upper: Math.min(1, a + z * se) };
}

/** ROC 点（同じ値は 1 点にまとめる）。(0,0) から (1,1) まで。 */
export function rocPoints(l: Labelled, dir: Direction = "higher"): RocPoint[] {
  const s = oriented(l.score, dir), { pos, neg } = counts(l);
  const order = s.map((_, i) => i).sort((a, b) => s[b] - s[a]);
  const pts: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: Infinity }];
  let tp = 0, fp = 0;
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j < order.length && s[order[j]] === s[order[i]]) { if (l.label[order[j]] === 1) tp++; else fp++; j++; }
    pts.push({ fpr: fp / neg, tpr: tp / pos, threshold: s[order[i]] });
    i = j;
  }
  return pts;
}

export interface Confusion {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export function confusion(label: number[], predicted: number[]): Confusion {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  label.forEach((y, i) => {
    if (y === 1) { if (predicted[i] === 1) c.tp++; else c.fn++; }
    else if (predicted[i] === 1) c.fp++; else c.tn++;
  });
  return c;
}

export interface Performance {
  sensitivity: Proportion;
  specificity: Proportion;
  ppv: Proportion;
  npv: Proportion;
  accuracy: Proportion;
  confusion: Confusion;
}

export function performance(c: Confusion, level = 0.95): Performance {
  return {
    sensitivity: proportion(c.tp, c.tp + c.fn, level), specificity: proportion(c.tn, c.tn + c.fp, level),
    ppv: proportion(c.tp, c.tp + c.fp, level), npv: proportion(c.tn, c.tn + c.fn, level),
    accuracy: proportion(c.tp + c.tn, c.tp + c.fp + c.tn + c.fn, level), confusion: c,
  };
}

export const predictAt = (score: number[], cut: number, dir: Direction): number[] => score.map((v) => ((dir === "higher" ? v > cut : v < cut) ? 1 : 0));

export interface Cut {
  cut: number;
  direction: Direction;
  confusion: Confusion;
}

/**
 * 基準（精度最大 または Youden 指数最大）で最良のしきい値を選ぶ。同点なら値の小さいしきい値。
 * 候補は隣り合う異なる値の中点。向きは固定（呼び出し側が bestDirection で決める）。
 */
export function bestCut(l: Labelled, dir: Direction, criterion: Criterion): Cut | null {
  const { pos, neg } = counts(l);
  if (!pos || !neg) return null;
  const idx = l.score.map((_, i) => i).sort((a, b) => l.score[a] - l.score[b]);
  const n = idx.length;
  let best: Cut | null = null, bestVal = -Infinity;
  let posBelow = 0, negBelow = 0;
  for (let k = 0; k < n - 1; k++) {
    if (l.label[idx[k]] === 1) posBelow++; else negBelow++;
    if (l.score[idx[k]] === l.score[idx[k + 1]]) continue;
    const cut = (l.score[idx[k]] + l.score[idx[k + 1]]) / 2;
    // 「高い側 = 陽性」なら TP = 上側の陽性、「低い側 = 陽性」なら TP = 下側の陽性
    const c: Confusion = dir === "higher"
      ? { tp: pos - posBelow, fp: neg - negBelow, tn: negBelow, fn: posBelow }
      : { tp: posBelow, fp: negBelow, tn: neg - negBelow, fn: pos - posBelow };
    const val = criterion === "accuracy" ? (c.tp + c.tn) / n : c.tp / pos + c.tn / neg - 1;
    if (val > bestVal + 1e-12) { bestVal = val; best = { cut, direction: dir, confusion: c }; }
  }
  return best;
}
