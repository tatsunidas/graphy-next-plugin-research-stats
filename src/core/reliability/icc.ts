/**
 * 級内相関係数（ICC）。二元配置（症例 × 条件）の分散分析から、McGraw & Wong の
 * 「absolute agreement」の ICC(A,1)（単一測定）と ICC(A,k)（平均測定）、95% 信頼区間（F 近似）を求める。
 * 数値は two-way mixed と two-way random の absolute agreement で同じ（母集団の仮定が違うだけ）。
 *
 * 区分（再現性の論文）: 信頼区間（本実装は下限）が 0.5 未満 poor、0.75 未満 moderate、0.9 以下 good、0.9 超 excellent。
 * 負の ICC・区間は 0 に切る。境界の等号は論文に無く、ここで決めた。
 */
import { fQuantile } from "../survival/dist";

export type IccCategory = "poor" | "moderate" | "good" | "excellent";

export interface IccResult {
  n: number;
  k: number;
  msr: number;
  msc: number;
  mse: number;
  /** ICC(A,1)・ICC(A,k) とその 95% CI（切る前の値）。 */
  single: { icc: number; lower: number; upper: number };
  average: { icc: number; lower: number; upper: number };
  /** 負の値を 0 に切った値（論文の扱い）。区分はこちらの下限で付ける。 */
  singleTruncated: { icc: number; lower: number; upper: number };
  averageTruncated: { icc: number; lower: number; upper: number };
  category: IccCategory;
  /** MSE = 0 など、区間の式の極限を使ったとき。 */
  note: string | null;
}

export function iccCategory(lower: number): IccCategory {
  return lower < 0.5 ? "poor" : lower < 0.75 ? "moderate" : lower <= 0.9 ? "good" : "excellent";
}

const trunc = (r: { icc: number; lower: number; upper: number }) => ({ icc: Math.max(0, r.icc), lower: Math.max(0, r.lower), upper: Math.max(0, r.upper) });
const sb = (x: number, k: number) => (k * x) / (1 + (k - 1) * x);

/** data[症例][条件]。欠損を含んではいけない（呼び出し側が完全症例だけを渡す）。 */
export function icc(data: number[][], level = 0.95): IccResult | { error: string } {
  const n = data.length, k = data[0]?.length ?? 0;
  if (k < 2) return { error: "条件が 2 つ以上要ります" };
  if (n < 3) return { error: `症例が ${n} 人しかありません（3 人以上要ります）` };
  for (const row of data) { if (row.length !== k) return { error: "条件の数が症例ごとに違います" }; for (const v of row) if (!Number.isFinite(v)) return { error: "欠損・非数があります" }; }
  const rowMean = data.map((r) => r.reduce((a, b) => a + b, 0) / k);
  const colMean = Array.from({ length: k }, (_, j) => data.reduce((a, r) => a + r[j], 0) / n);
  const gm = rowMean.reduce((a, b) => a + b, 0) / n;
  const ssr = k * rowMean.reduce((a, m) => a + (m - gm) ** 2, 0);
  const ssc = n * colMean.reduce((a, m) => a + (m - gm) ** 2, 0);
  // 誤差の平方和は残差から直接求める（全体 − 行 − 列 の引き算だと、完全に一致するデータで 1e-16 の誤差が残る）
  let sse = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) sse += (data[i][j] - rowMean[i] - colMean[j] + gm) ** 2;
  const msr = ssr / (n - 1), msc = ssc / (k - 1), mse = sse / ((n - 1) * (k - 1));
  if (msr === 0 && mse === 0) return { error: "全症例で同じ値です（ICC を定義できません）" };
  const denom = msr + (k - 1) * mse + (k * (msc - mse)) / n;
  const i1 = (msr - mse) / denom;
  const ik = (msr - mse) / (msr + (msc - mse) / n);
  const a = (1 - level) / 2;
  let lo: number, hi: number, note: string | null = null;
  if (mse === 0) {
    // 誤差の分散が 0（完全一致、または条件ごとの定数のずれだけ）。F 近似の式の極限（Fj → ∞ で自由度 v → k−1）を使う
    const v = k - 1;
    const F3 = fQuantile(1 - a, n - 1, v), F3u = fQuantile(1 - a, v, n - 1);
    lo = (n * msr) / (F3 * k * msc + n * msr);
    hi = (n * F3u * msr) / (k * msc + n * F3u * msr);
    note = "誤差の分散が 0 のため、信頼区間は式の極限で求めました";
  } else {
    const Fj = msc / mse;
    const t = n * (1 + (k - 1) * i1) - k * i1;
    const vn = (k - 1) * (n - 1) * (k * i1 * Fj + t) ** 2;
    const vd = (n - 1) * k * k * i1 * i1 * Fj * Fj + t * t;
    const v = vn / vd;
    const F3 = fQuantile(1 - a, n - 1, v), F3u = fQuantile(1 - a, v, n - 1);
    lo = (n * (msr - F3 * mse)) / (F3 * (k * msc + (k * n - k - n) * mse) + n * msr);
    hi = (n * (F3u * msr - mse)) / (k * msc + (k * n - k - n) * mse + n * F3u * msr);
  }
  const single = { icc: i1, lower: lo, upper: hi };
  const average = { icc: ik, lower: sb(lo, k), upper: sb(hi, k) };
  const singleTruncated = trunc(single);
  return { n, k, msr, msc, mse, single, average, singleTruncated, averageTruncated: trunc(average), category: iccCategory(singleTruncated.lower), note };
}
