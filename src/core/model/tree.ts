/**
 * CART（二分木）。ランダムフォレストと勾配ブースティングの共通部品。scikit-learn の決定木と同じ定義にしている:
 * - 分岐の候補は、ノードの行を特徴量で並べたときの隣り合う異なる値の中点（a/2 + b/2）。x ≤ しきい値が左。
 *   値の差が 1e-7 以下なら同じ値とみなす（sklearn の FEATURE_THRESHOLD）。
 * - 基準は、分類が Gini、回帰が二乗誤差。重み（ブートストラップの回数）を扱う。重み 0 の行は使わない。
 * - 葉にする条件: 深さが上限、行が 2 未満、不純度が 2.2e-16 以下、または有効な分岐が無い。有効な分岐があれば
 *   改善が 0 でも分ける（sklearn と同じ）。葉の最小の行数は 1。
 * - mtry を指定すると、ノードごとに特徴量を無作為な順で引き、ノード内で値が一定のものは数えずに、
 *   値が一定でない特徴量を mtry 個見るまで探す。
 */
import type { Rng } from "../rng";

const FEATURE_THRESHOLD = 1e-7;
const EPSILON = 2.220446049250313e-16;

export type Criterion = "gini" | "mse";

export interface TreeNode {
  /** 葉は -1。 */
  feature: number;
  threshold: number;
  left: number;
  right: number;
  /** 葉の値（分類はクラス 1 の重みの割合、回帰は重み付き平均）。 */
  value: number;
  /** ノードの重みの合計と不純度（重要度の計算用）。 */
  weight: number;
  impurity: number;
}

export interface Tree {
  nodes: TreeNode[];
}

export interface TreeOptions {
  criterion: Criterion;
  /** null なら上限なし。 */
  maxDepth: number | null;
  /** null なら毎回すべての特徴量を見る。 */
  mtry: number | null;
}

/**
 * X の行 rows（重み w[行] > 0 のものだけを渡す）と目的 target[行] から木を作る。cols は使う列。
 * 乱数は mtry を指定したときだけ使う。
 */
export function buildTree(X: number[][], rows: number[], w: ArrayLike<number>, target: ArrayLike<number>, cols: number[], opts: TreeOptions, rng: Rng | null): Tree {
  const nodes: TreeNode[] = [];
  const stats = (S: number[]) => {
    let W = 0, s = 0, s2 = 0;
    for (const i of S) { W += w[i]; s += w[i] * target[i]; s2 += w[i] * target[i] * target[i]; }
    const value = s / W;
    // 分類の target は 0/1 なので、Gini = 1 − p² − (1−p)²。回帰は重み付きの分散。
    const impurity = opts.criterion === "gini" ? 1 - value * value - (1 - value) * (1 - value) : s2 / W - value * value;
    return { W, s, value, impurity };
  };
  const grow = (S: number[], depth: number): number => {
    const id = nodes.length;
    const st = stats(S);
    nodes.push({ feature: -1, threshold: 0, left: -1, right: -1, value: st.value, weight: st.W, impurity: st.impurity });
    if ((opts.maxDepth !== null && depth >= opts.maxDepth) || S.length < 2 || st.impurity <= EPSILON) return id;
    const sp = bestSplit(X, S, w, target, cols, opts, st.W, st.s, rng);
    if (!sp) return id;
    const L = S.filter((i) => X[i][sp.feature] <= sp.threshold), R = S.filter((i) => X[i][sp.feature] > sp.threshold);
    const node = nodes[id];
    node.feature = sp.feature;
    node.threshold = sp.threshold;
    node.left = grow(L, depth + 1);
    node.right = grow(R, depth + 1);
    return id;
  };
  grow(rows, 0);
  return { nodes };
}

function bestSplit(X: number[][], S: number[], w: ArrayLike<number>, target: ArrayLike<number>, cols: number[], opts: TreeOptions, W: number, sum: number, rng: Rng | null): { feature: number; threshold: number } | null {
  const order = cols.slice();
  const limit = opts.mtry === null ? cols.length : opts.mtry;
  let best = -Infinity, bestF = -1, bestT = 0, visited = 0;
  const idx = S.slice();
  for (let k = 0; k < order.length && visited < limit; k++) {
    if (rng) { const r = k + rng.int(order.length - k); [order[k], order[r]] = [order[r], order[k]]; }
    const f = order[k];
    idx.sort((a, b) => X[a][f] - X[b][f]);
    const n = idx.length;
    if (X[idx[n - 1]][f] <= X[idx[0]][f] + FEATURE_THRESHOLD) continue; // ノード内で一定: 数えない
    visited++;
    let wL = 0, sL = 0;
    for (let p = 1; p < n; p++) {
      const i = idx[p - 1];
      wL += w[i];
      sL += w[i] * target[i];
      const a = X[i][f], b = X[idx[p]][f];
      if (b <= a + FEATURE_THRESHOLD) continue;
      const wR = W - wL, sR = sum - sL;
      // 代理の改善量（sklearn の proxy_impurity_improvement と同じ順位になる式）。
      // Gini: Σ_k cL_k²/wL + Σ_k cR_k²/wR、二乗誤差: sL²/wL + sR²/wR。
      const proxy = opts.criterion === "gini"
        ? (sL * sL + (wL - sL) * (wL - sL)) / wL + (sR * sR + (wR - sR) * (wR - sR)) / wR
        : (sL * sL) / wL + (sR * sR) / wR;
      if (proxy > best) {
        best = proxy;
        bestF = f;
        let t = a / 2 + b / 2;
        if (t === b || !Number.isFinite(t)) t = a;
        bestT = t;
      }
    }
  }
  return bestF < 0 ? null : { feature: bestF, threshold: bestT };
}

/** x（X と同じ列の並びの 1 行）が落ちる葉の番号。 */
export function leafOf(tree: Tree, x: ArrayLike<number>): number {
  let k = 0;
  const nd = tree.nodes;
  while (nd[k].feature >= 0) k = x[nd[k].feature] <= nd[k].threshold ? nd[k].left : nd[k].right;
  return k;
}

/**
 * 不純度の減少による重要度（正規化しない。sklearn の compute_feature_importances(normalize=False) と同じ）。
 * m は X の列の数。
 */
export function rawImportance(tree: Tree, m: number): number[] {
  const imp = new Array<number>(m).fill(0);
  const nd = tree.nodes;
  for (const n of nd) {
    if (n.feature < 0) continue;
    const L = nd[n.left], R = nd[n.right];
    imp[n.feature] += n.weight * n.impurity - L.weight * L.impurity - R.weight * R.impurity;
  }
  return imp.map((v) => v / nd[0].weight);
}

export function normalize(v: number[]): number[] {
  const s = v.reduce((a, b) => a + b, 0);
  return s > 0 ? v.map((x) => x / s) : v.map(() => 0);
}
