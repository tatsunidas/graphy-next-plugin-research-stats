/**
 * 勾配ブースティング（二値、対数損失）。scikit-learn 1.9 の GradientBoostingClassifier と同じ定義:
 * - 初期値は事前の対数オッズ log(p̄/(1−p̄))。
 * - 各段で、負の勾配 r = y − p に回帰木（二乗誤差、深さの上限 maxDepth、葉の最小の行数 1）を当てる。
 *   木の基準は sklearn 1.9 で friedman_mse から squared_error に変わった。ここは 1.9 に合わせている。
 * - 葉の値は Newton の一歩 Σr / Σp(1−p)（分母が 1e-150 未満なら 0）。予測に学習率 × 葉の値を足す。
 * - subsample < 1 なら、各段で ⌊subsample·n⌋ 行（最小 1）を非復元で選び、木と葉の値はその行だけで求める。
 * - 重要度は、木ごとの正規化しない不純度の減少の平均を正規化したもの（葉 1 枚の木は除く）。
 */
import type { Rng } from "../rng";
import { buildTree, leafOf, normalize, rawImportance, type Tree } from "./tree";

export interface GbmOptions {
  stages: number;
  maxDepth: number;
  learningRate: number;
  /** (0, 1]。1 なら乱数を使わない。 */
  subsample: number;
  /** **試験専用**の誤り: 葉の値を残差の平均にする（Newton の一歩をやめる）。 */
  meanLeaf?: boolean;
}

export interface Gbm {
  cols: number[];
  init: number;
  trees: Tree[];
  /** 木ごとの葉の値（学習率を掛ける前）。 */
  leafValues: number[][];
  learningRate: number;
  importance: number[];
  /** 対数オッズ。 */
  raw(x: ArrayLike<number>): number;
  /** クラス 1 の確率。 */
  predict(x: ArrayLike<number>): number;
}

const expit = (v: number) => 1 / (1 + Math.exp(-v));

export function gbm(X: number[][], y: number[], rows: number[], cols: number[], o: GbmOptions, rng: Rng): Gbm {
  if (!Number.isInteger(o.stages) || o.stages < 1) throw new RangeError("段数は 1 以上の整数です");
  if (!Number.isInteger(o.maxDepth) || o.maxDepth < 1) throw new RangeError("木の深さは 1 以上の整数です");
  if (!(o.learningRate > 0)) throw new RangeError("学習率は正の数です");
  if (!(o.subsample > 0 && o.subsample <= 1)) throw new RangeError("subsample は 0 より大きく 1 以下です");
  const n = rows.length, m = X[0].length;
  const p0 = rows.reduce((s, i) => s + y[i], 0) / n;
  if (p0 <= 0 || p0 >= 1) throw new RangeError("訓練の行に両方のクラスが要ります");
  const init = Math.log(p0 / (1 - p0));
  const F = new Float64Array(X.length);
  for (const i of rows) F[i] = init;
  const r = new Float64Array(X.length), w = new Float64Array(X.length);
  const trees: Tree[] = [], leafValues: number[][] = [], imps: number[][] = [];
  const nIn = Math.max(1, Math.floor(o.subsample * n));
  for (let s = 0; s < o.stages; s++) {
    for (const i of rows) r[i] = y[i] - expit(F[i]);
    let inBag = rows;
    if (o.subsample < 1) {
      const c = rows.slice();
      for (let k = 0; k < nIn; k++) { const q = k + rng.int(n - k); [c[k], c[q]] = [c[q], c[k]]; }
      const pick = new Set(c.slice(0, nIn));
      inBag = rows.filter((i) => pick.has(i));
    }
    w.fill(0);
    for (const i of inBag) w[i] = 1;
    const tree = buildTree(X, inBag, w, r, cols, { criterion: "mse", maxDepth: o.maxDepth, mtry: null }, null);
    const num = new Float64Array(tree.nodes.length), den = new Float64Array(tree.nodes.length), cnt = new Float64Array(tree.nodes.length);
    for (const i of inBag) {
      const k = leafOf(tree, X[i]);
      const p = y[i] - r[i];
      num[k] += r[i];
      den[k] += p * (1 - p);
      cnt[k] += 1;
    }
    const vals = tree.nodes.map((nd, k) => {
      if (nd.feature >= 0 || cnt[k] === 0) return 0;
      if (o.meanLeaf) return num[k] / cnt[k];
      const d = den[k] / cnt[k];
      return Math.abs(d) < 1e-150 ? 0 : num[k] / cnt[k] / d;
    });
    for (const i of rows) F[i] += o.learningRate * vals[leafOf(tree, X[i])];
    trees.push(tree);
    leafValues.push(vals);
    if (tree.nodes.length > 1) imps.push(rawImportance(tree, m));
  }
  const mean = new Array<number>(m).fill(0);
  for (const v of imps) v.forEach((x, j) => { mean[j] += x / imps.length; });
  const imp = normalize(mean);
  const raw = (x: ArrayLike<number>) => {
    let v = init;
    for (let t = 0; t < trees.length; t++) v += o.learningRate * leafValues[t][leafOf(trees[t], x)];
    return v;
  };
  return { cols, init, trees, leafValues, learningRate: o.learningRate, importance: cols.map((j) => imp[j]), raw, predict: (x) => expit(raw(x)) };
}
