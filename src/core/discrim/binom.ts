/** 二項割合の Clopper–Pearson（正確）信頼区間。感度・特異度・精度などに使う。 */
import { betaQuantile } from "../survival/dist";

export interface Proportion {
  x: number;
  n: number;
  /** n = 0 のとき null（0 や 1 にしない）。 */
  value: number | null;
  lower: number | null;
  upper: number | null;
}

export function proportion(x: number, n: number, level = 0.95): Proportion {
  if (n <= 0) return { x, n, value: null, lower: null, upper: null };
  const a = (1 - level) / 2;
  return {
    x, n, value: x / n,
    lower: x === 0 ? 0 : betaQuantile(a, x, n - x + 1),
    upper: x === n ? 1 : betaQuantile(1 - a, x + 1, n - x),
  };
}
