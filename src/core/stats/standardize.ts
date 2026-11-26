/**
 * 標準化（平均 0・標本 SD（n−1）で 1）。平均と SD は、与えた行だけから推定する（欠損を除く）。
 * 推定に使わなかった行の値は読まない。SD が 0（または値が 2 個未満）の列は適用不可で、0 で割らない。
 */
/** 平均と標本 SD（n − 1）。標準化の算術はここ 1 か所（欠損のある列を扱う版・訓練の行の行列版・LASSO の版がすべてこれを使う）。 */
export function meanSd(v: readonly number[]): { mean: number; sd: number } {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
  return { mean, sd };
}

export interface FittedStandardizer {
  means: (number | null)[];
  sds: (number | null)[];
  /** 適用できる列（SD > 0）。 */
  usable: boolean[];
  /** 推定に読んだ行。 */
  fitRows: number[];
  /** rows の行を標準化して返す（rows の順）。適用不可の列と欠損は null。 */
  apply(X: (number | null)[][], rows: number[]): (number | null)[][];
}

/** 欠損（null）を除いて列ごとに推定する版。訓練の行の行列（欠損なし）に使う版は pipeline/preprocess.ts の `fitStandardizer`。 */
export function fitStandardizerNullable(X: (number | null)[][], rows: number[]): FittedStandardizer {
  const p = rows.length > 0 ? X[rows[0]].length : 0;
  const means: (number | null)[] = [], sds: (number | null)[] = [], usable: boolean[] = [];
  for (let j = 0; j < p; j++) {
    const v = rows.map((i) => X[i][j]).filter((x): x is number => x !== null && Number.isFinite(x));
    if (v.length < 2) { means.push(v.length === 1 ? v[0] : null); sds.push(null); usable.push(false); continue; }
    const { mean: mu, sd } = meanSd(v);
    means.push(mu);
    sds.push(sd > 0 ? sd : null);
    usable.push(sd > 0);
  }
  const fitRows = [...rows];
  return {
    means, sds, usable, fitRows,
    apply(Y, target) {
      return target.map((i) => Y[i].map((v, j) => (usable[j] && v !== null && Number.isFinite(v) ? (v - (means[j] as number)) / (sds[j] as number) : null)));
    },
  };
}

export type ScaleModel = "lasso" | "svm" | "logistic" | "knn" | "tree" | "rf" | "gbm" | "coxboost";

/** 特徴量の尺度に敏感なモデル（罰則・カーネル・距離・勾配法）は標準化が要り、分割の大小だけを見る木系は要らない。 */
const SENSITIVE: Record<ScaleModel, boolean> = {
  lasso: true,
  svm: true,
  logistic: true,
  knn: true,
  tree: false,
  rf: false,
  gbm: false,
  coxboost: true, // 係数の更新に、列ごとの単回帰（尺度に依存する）を使う
};

export function needsStandardization(model: ScaleModel): boolean {
  return SENSITIVE[model];
}
