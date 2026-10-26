/**
 * 上位 k 個の特徴量のロジスティック回帰による判別を、leave-one-out（LOO）で検証する。
 *
 * 特徴量の選択（Mann–Whitney の p の小さい順、任意で BH の q < alpha）を各折の訓練データの内側でやり直す
 * （selectInside = true、正しい）か、全データで 1 回だけやってしまう（false、漏れ。比較の検出器用）かを選べる。
 * 標準化も訓練データだけで行う。選択が 0 個の折は、訓練データの陽性割合を予測にする。
 */
import { mannWhitney } from "../survival/mannwhitney";
import { benjaminiHochberg } from "../survival/multiplicity";
import { logisticFit } from "./logistic";

export interface LooOptions {
  k: number;
  /** null なら q による絞り込みをしない（上位 k 個をそのまま使う）。 */
  alpha: number | null;
  selectInside: boolean;
}

export interface LooResult {
  /** 左に除いた症例の予測確率。モデルを作れなかった折は null。 */
  pred: (number | null)[];
  /** 特徴量ごとに、何回の折で選ばれたか。 */
  selectionCounts: number[];
  failed: number;
}

/** idx の症例で特徴量を選ぶ。返すのは特徴量の番号（p の小さい順）。 */
export function selectTopK(X: number[][], y: number[], idx: number[], k: number, alpha: number | null): number[] {
  const m = X[0].length;
  const p: (number | null)[] = [];
  for (let j = 0; j < m; j++) {
    const a = idx.filter((i) => y[i] === 1).map((i) => X[i][j]), b = idx.filter((i) => y[i] === 0).map((i) => X[i][j]);
    const r = mannWhitney(a, b);
    p.push("error" in r ? null : r.p);
  }
  const q = alpha === null ? p : benjaminiHochberg(p);
  return p.map((pv, j) => ({ j, pv, qv: q[j] }))
    .filter((x) => x.pv !== null && (alpha === null || (x.qv as number) < alpha))
    .sort((u, v) => (u.pv as number) - (v.pv as number) || u.j - v.j)
    .slice(0, k).map((x) => x.j);
}

export type Predictor = (row: number[]) => number;

/** 選んだ特徴量で、訓練データだけで標準化してロジスティック回帰を当てはめる。 */
export function fitOnColumns(X: number[][], y: number[], idx: number[], cols: number[]): Predictor | { error: string } {
  if (!cols.length) {
    const prev = idx.reduce((a, i) => a + y[i], 0) / idx.length;
    return () => prev;
  }
  const st = cols.map((j) => {
    const v = idx.map((i) => X[i][j]);
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return { j, m, s: Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) };
  });
  if (st.some((t) => !(t.s > 0))) return { error: "訓練データで値が同じ特徴量があります" };
  const row = (r: number[]) => st.map((t) => (r[t.j] - t.m) / t.s);
  const fit = logisticFit(idx.map((i) => row(X[i])), idx.map((i) => y[i]));
  if (!fit.ok) return { error: fit.error };
  return (r) => fit.fit.predict(row(r));
}

export function looTopK(X: number[][], y: number[], opts: LooOptions): LooResult {
  const n = X.length, all = Array.from({ length: n }, (_, i) => i);
  const m = X[0]?.length ?? 0;
  const counts = new Array<number>(m).fill(0);
  const outside = opts.selectInside ? null : selectTopK(X, y, all, opts.k, opts.alpha);
  const pred: (number | null)[] = [];
  let failed = 0;
  for (let i = 0; i < n; i++) {
    const train = all.filter((t) => t !== i);
    const cols = outside ?? selectTopK(X, y, train, opts.k, opts.alpha);
    for (const j of cols) counts[j]++;
    const f = fitOnColumns(X, y, train, cols);
    if (typeof f !== "function") { pred.push(null); failed++; continue; }
    pred.push(f(X[i]));
  }
  return { pred, selectionCounts: counts, failed };
}
