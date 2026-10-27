/**
 * 前処理を**訓練に使う行だけ**から求めて、他の行に当てる（データ漏れを防ぐ）。
 * 欠損の補完は訓練の行の中央値。標準化は補完した後の訓練の行の平均・SD（n−1）。
 */

export interface Imputed {
  /** 全行の値。補完しないで欠けている値は NaN（その列を使う予測はできない）。 */
  X: number[][];
  /** 列ごとの訓練の行の中央値（欠損を除く）。全部欠けていれば null。 */
  medians: (number | null)[];
  /** この訓練の行で使える列（補完するなら中央値がある、しないなら訓練の行に欠損が無い）。 */
  usable: boolean[];
}

export function median(v: number[]): number | null {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b), h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

export function imputeByTrain(X: (number | null)[][], train: number[], impute: boolean): Imputed {
  const m = X[0]?.length ?? 0;
  const medians: (number | null)[] = [], usable: boolean[] = [];
  for (let j = 0; j < m; j++) {
    const vals = train.map((i) => X[i][j]);
    const present = vals.filter((v): v is number => v !== null);
    medians.push(impute ? median(present) : null);
    usable.push(impute ? present.length > 0 : present.length === vals.length && vals.length > 0);
  }
  const out = X.map((row) => row.map((v, j) => (v !== null ? v : impute && medians[j] !== null ? (medians[j] as number) : NaN)));
  return { X: out, medians, usable };
}

export interface Standardizer {
  cols: number[];
  means: number[];
  sds: number[];
}

/** 訓練の行の平均・SD。SD が 0 の列があればエラー（割れないので）。 */
export function fitStandardizer(X: number[][], train: number[], cols: number[]): Standardizer | { error: string } {
  const means: number[] = [], sds: number[] = [];
  for (const j of cols) {
    const v = train.map((i) => X[i][j]);
    if (v.some((x) => !Number.isFinite(x))) return { error: `訓練の行に欠損がある列を使おうとしました（列 ${j + 1}）` };
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / (v.length - 1));
    if (!(sd > 0)) return { error: `訓練の行で値が同じ特徴量があります（列 ${j + 1}）` };
    means.push(mu);
    sds.push(sd);
  }
  return { cols, means, sds };
}

/** 標準化した値。欠けた値があれば null。 */
export function standardizeRow(s: Standardizer, row: number[]): number[] | null {
  const z = s.cols.map((j, a) => (row[j] - s.means[a]) / s.sds[a]);
  return z.every(Number.isFinite) ? z : null;
}
