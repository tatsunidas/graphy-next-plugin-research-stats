/**
 * 較正（予測確率と実際の割合の一致）。予測確率の分位でビンに分け（件数が揃う）、ビンごとの平均予測と
 * 実際の陽性割合（Clopper–Pearson の CI）を出す。平滑化はしない。傾きと切片は logit(予測) への回帰。
 */
import { proportion, type Proportion } from "./binom";
import { logisticFit } from "./logistic";

export interface CalibrationBin {
  n: number;
  meanPredicted: number;
  observed: Proportion;
}

export function calibrationBins(pred: number[], label: number[], bins = 10): CalibrationBin[] {
  const n = pred.length, b = Math.min(bins, n);
  const order = pred.map((_, i) => i).sort((x, y) => pred[x] - pred[y]);
  const out: CalibrationBin[] = [];
  for (let k = 0; k < b; k++) {
    const idx = order.slice(Math.floor((k * n) / b), Math.floor(((k + 1) * n) / b));
    if (!idx.length) continue;
    out.push({
      n: idx.length,
      meanPredicted: idx.reduce((a, i) => a + pred[i], 0) / idx.length,
      observed: proportion(idx.reduce((a, i) => a + label[i], 0), idx.length),
    });
  }
  return out;
}

/** logit(予測) で観測を回帰した切片と傾き（理想は 0 と 1）。予測が 0 か 1 ちょうどなら計算しない。 */
export function calibrationSlope(pred: number[], label: number[]): { intercept: number; slope: number } | { error: string } {
  if (pred.some((p) => !(p > 0 && p < 1))) return { error: "予測確率が 0 か 1 ちょうどの症例があり、logit を取れません" };
  const r = logisticFit(pred.map((p) => [Math.log(p / (1 - p))]), label);
  return r.ok ? { intercept: r.fit.beta[0], slope: r.fit.beta[1] } : { error: r.error };
}
