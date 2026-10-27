/**
 * ランダムフォレスト（二値分類。Breiman 2001）。scikit-learn の RandomForestClassifier と同じ定義:
 * - 各木は、訓練の行から復元抽出で n 回引き、引かれた回数を重みにする（bootstrap=false なら全行を重み 1）。
 * - 特徴量は**分岐ごと**に mtry 個を無作為に選ぶ（木ごとに選ぶのではない）。
 * - 木は深さの上限なし・葉の最小の行数 1、Gini。
 * - 確率は、各木の葉の「クラス 1 の重みの割合」の平均。
 * - 重要度は、木ごとに正規化した不純度の減少の平均を、もう一度正規化したもの。
 */
import type { Rng } from "../rng";
import { buildTree, leafOf, normalize, rawImportance, type Tree } from "./tree";

export interface ForestOptions {
  trees: number;
  /** 分岐ごとに見る特徴量の数（1 以上、列の数以下）。 */
  mtry: number;
  /** 既定 true。false は試験用（乱数を使わない形にして sklearn と数値で照合する）。 */
  bootstrap?: boolean;
  /** **試験専用**の誤り: 特徴量を木ごとに選ぶ。負例として、分布の照合で落ちることを確かめる。 */
  perTreeFeatures?: boolean;
}

export interface Forest {
  cols: number[];
  trees: Tree[];
  /** cols の並びの重要度（合計 1）。 */
  importance: number[];
  /** X の行と同じ並びの 1 行から、クラス 1 の確率。 */
  predict(x: ArrayLike<number>): number;
}

/** y は X の行に揃えた 0/1。rows は訓練の行、cols は使う列（欠損の無い列）。 */
export function randomForest(X: number[][], y: number[], rows: number[], cols: number[], o: ForestOptions, rng: Rng): Forest {
  if (!Number.isInteger(o.trees) || o.trees < 1) throw new RangeError("木の数は 1 以上の整数です");
  if (!Number.isInteger(o.mtry) || o.mtry < 1 || o.mtry > cols.length) throw new RangeError(`mtry は 1 以上 ${cols.length} 以下の整数です`);
  const n = rows.length, m = X[0].length;
  const w = new Float64Array(X.length);
  const trees: Tree[] = [];
  const imps: number[][] = [];
  for (let t = 0; t < o.trees; t++) {
    w.fill(0);
    if (o.bootstrap === false) for (const i of rows) w[i] = 1;
    else for (let d = 0; d < n; d++) w[rows[rng.int(n)]] += 1;
    const inBag = rows.filter((i) => w[i] > 0);
    let useCols = cols, mtry: number | null = o.mtry;
    if (o.perTreeFeatures) {
      const c = cols.slice();
      for (let k = 0; k < o.mtry; k++) { const r = k + rng.int(c.length - k); [c[k], c[r]] = [c[r], c[k]]; }
      useCols = c.slice(0, o.mtry);
      mtry = null;
    }
    const tree = buildTree(X, inBag, w, y, useCols, { criterion: "gini", maxDepth: null, mtry: mtry === cols.length ? null : mtry }, mtry === null || mtry === cols.length ? null : rng);
    trees.push(tree);
    imps.push(normalize(rawImportance(tree, m)));
  }
  const mean = new Array<number>(m).fill(0);
  for (const v of imps) v.forEach((x, j) => { mean[j] += x / imps.length; });
  const imp = normalize(mean);
  return {
    cols, trees,
    importance: cols.map((j) => imp[j]),
    predict: (x) => trees.reduce((s, tr) => s + tr.nodes[leafOf(tr, x)].value, 0) / trees.length,
  };
}

/** mtry の "sqrt"（sklearn の max_features="sqrt" と同じ: ⌊√列数⌋、最小 1）。 */
export const sqrtMtry = (nCols: number) => Math.max(1, Math.floor(Math.sqrt(nCols)));
