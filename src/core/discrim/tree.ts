/**
 * 再帰的な決定木（肺癌の論文の定義）。各節点で、その節点に残った症例について、診断精度が最大になる
 * 特徴量・向き・しきい値を ROC 的に選ぶ（値 > しきい値 を陽性とする向き、または 値 < しきい値）。
 * 陽性側の枝は陽性、陰性側の枝は陰性と判定し、枝の誤分類率が stopRate（既定 15%）未満なら、その枝に節点を足さない。
 * 一般の CART ではない（不純度は使わない）。
 *
 * 論文に書かれていない仮定（実装記録に明記）: 候補の特徴量は呼び出し側が指定／節点の最小症例数と最大の深さ
 * （暴走防止）／分けても精度が上がらない枝はそれ以上分けない／同点は特徴量番号・小さいしきい値・higher 優先の順。
 */
import type { Rng } from "../rng";
import { betaQuantile } from "../survival/dist";
import { performance, type Confusion, type Direction, type Performance } from "./roc";

export interface TreeOptions {
  /** 枝の誤分類率がこれ未満なら止める（論文は 15%）。 */
  stopRate: number;
  minNode: number;
  maxDepth: number;
  candidates: number[];
}

export interface TreeNode {
  depth: number;
  n: number;
  pos: number;
  /** 親の判定でこの枝に割り当てられたラベル（根は null）。 */
  branchLabel: 0 | 1 | null;
  /** 葉のときの判定。 */
  label: 0 | 1;
  split: { feature: number; direction: Direction; cut: number; confusion: Confusion } | null;
  positive: TreeNode | null;
  negative: TreeNode | null;
}

const classify = (v: number, direction: Direction, cut: number) => (direction === "higher" ? v > cut : v < cut);

/** 節点の症例 idx で最良の分割を探す。最良の精度（正解数）と分割を返す。 */
function bestSplit(X: number[][], y: number[], idx: number[], cands: number[]): { feature: number; direction: Direction; cut: number; confusion: Confusion; correct: number } | null {
  const n = idx.length, pos = idx.reduce((a, i) => a + y[i], 0), neg = n - pos;
  let best: { feature: number; direction: Direction; cut: number; confusion: Confusion; correct: number } | null = null;
  for (const j of cands) {
    const order = [...idx].sort((a, b) => X[a][j] - X[b][j]);
    let pb = 0, nb = 0;
    for (let k = 0; k < n - 1; k++) {
      if (y[order[k]] === 1) pb++; else nb++;
      if (X[order[k]][j] === X[order[k + 1]][j]) continue;
      const cut = (X[order[k]][j] + X[order[k + 1]][j]) / 2;
      for (const direction of ["higher", "lower"] as const) {
        const c: Confusion = direction === "higher"
          ? { tp: pos - pb, fp: neg - nb, tn: nb, fn: pb }
          : { tp: pb, fp: nb, tn: neg - nb, fn: pos - pb };
        const correct = c.tp + c.tn;
        if (!best || correct > best.correct) best = { feature: j, direction, cut, confusion: c, correct };
      }
    }
  }
  return best;
}

export function buildTree(X: number[][], y: number[], opts: TreeOptions): TreeNode {
  const grow = (idx: number[], depth: number, branchLabel: 0 | 1 | null): TreeNode => {
    const n = idx.length, pos = idx.reduce((a, i) => a + y[i], 0);
    const node: TreeNode = { depth, n, pos, branchLabel, label: branchLabel ?? (pos * 2 >= n ? 1 : 0), split: null, positive: null, negative: null };
    if (n === 0) return node;
    if (branchLabel !== null) {
      const wrong = branchLabel === 1 ? n - pos : pos;
      if (wrong / n < opts.stopRate || n < opts.minNode || depth >= opts.maxDepth) return node;
    }
    const s = bestSplit(X, y, idx, opts.candidates);
    if (!s) return node;
    const baseline = branchLabel === null ? Math.max(pos, n - pos) : (branchLabel === 1 ? pos : n - pos);
    if (s.correct <= baseline) return node;
    node.split = { feature: s.feature, direction: s.direction, cut: s.cut, confusion: s.confusion };
    const up = idx.filter((i) => classify(X[i][s.feature], s.direction, s.cut));
    const down = idx.filter((i) => !classify(X[i][s.feature], s.direction, s.cut));
    node.positive = grow(up, depth + 1, 1);
    node.negative = grow(down, depth + 1, 0);
    return node;
  };
  return grow(Array.from({ length: X.length }, (_, i) => i), 0, null);
}

export function predictTree(t: TreeNode, row: number[]): 0 | 1 {
  let node = t;
  while (node.split && node.positive && node.negative) {
    node = classify(row[node.split.feature], node.split.direction, node.split.cut) ? node.positive : node.negative;
  }
  return node.label;
}

export function treePerformance(t: TreeNode, X: number[][], y: number[]): Performance {
  const pred = X.map((r) => predictTree(t, r));
  const c: Confusion = { tp: 0, fp: 0, tn: 0, fn: 0 };
  y.forEach((v, i) => { if (v === 1) { if (pred[i] === 1) c.tp++; else c.fn++; } else if (pred[i] === 1) c.fp++; else c.tn++; });
  return performance(c);
}

export function treeNodes(t: TreeNode): TreeNode[] {
  const out: TreeNode[] = [];
  const walk = (n: TreeNode) => { out.push(n); if (n.positive) walk(n.positive); if (n.negative) walk(n.negative); };
  walk(t);
  return out;
}

export interface MonteCarloCi {
  iterations: number;
  sensitivity: { lower: number; upper: number };
  specificity: { lower: number; upper: number };
  accuracy: { lower: number; upper: number };
  /** 抽出した各回の値の平均（点推定との比較や、乱数の再現の確認に使う）。 */
  means: { sensitivity: number; specificity: number; accuracy: number };
}

const quantile7 = (sorted: number[], q: number) => {
  const h = (sorted.length - 1) * q, lo = Math.floor(h), hi = Math.ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
};

/**
 * モンテカルロ: 各節点の感度・特異度を、その節点の二項データ由来の Beta(TP+1, FN+1)・Beta(TN+1, FP+1) から抽出し、
 * 各症例の判定をその確率で置き換えて木を通し、全体の感度・特異度・精度を求める。これを iterations 回繰り返し、
 * 2.5・97.5 パーセンタイルを CI とする（論文は「各節点の感度・特異度の 95% CI の不確かさを木に伝える」と述べるのみで、
 * 抽出の分布は書かれていない。ここは本実装の仮定）。
 */
export function monteCarloCi(t: TreeNode, X: number[][], y: number[], iterations: number, rng: Rng): MonteCarloCi {
  const nodes = treeNodes(t).filter((n) => n.split);
  const sens: number[] = [], spec: number[] = [], acc: number[] = [];
  for (let it = 0; it < iterations; it++) {
    const p = new Map<TreeNode, { se: number; sp: number }>();
    for (const n of nodes) {
      const c = n.split!.confusion;
      p.set(n, { se: betaQuantile(rng.next(), c.tp + 1, c.fn + 1), sp: betaQuantile(rng.next(), c.tn + 1, c.fp + 1) });
    }
    let tp = 0, fn = 0, tn = 0, fp = 0;
    for (let i = 0; i < X.length; i++) {
      let node = t;
      while (node.split && node.positive && node.negative) {
        const q = p.get(node)!;
        const goesPositive = y[i] === 1 ? rng.next() < q.se : rng.next() >= q.sp;
        node = goesPositive ? node.positive : node.negative;
      }
      if (y[i] === 1) { if (node.label === 1) tp++; else fn++; } else if (node.label === 1) fp++; else tn++;
    }
    sens.push(tp / (tp + fn)); spec.push(tn / (tn + fp)); acc.push((tp + tn) / X.length);
  }
  const ci = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return { lower: quantile7(s, 0.025), upper: quantile7(s, 0.975) }; };
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  return { iterations, sensitivity: ci(sens), specificity: ci(spec), accuracy: ci(acc), means: { sensitivity: mean(sens), specificity: mean(spec), accuracy: mean(acc) } };
}
