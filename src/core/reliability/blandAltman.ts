/**
 * Bland–Altman。2 つの条件 A・B の差 d = B − A と平均 (A+B)/2。バイアス = d の平均、
 * 一致限界 = バイアス ± 1.96 SD。信頼区間は Bland & Altman (1999) の近似:
 * バイアス ± t·SD/√n、一致限界 ± t·√(SD²(1/n + 1.96²/(2(n−1))))、t は自由度 n−1 の 97.5% 点。
 */
import { tQuantile } from "../survival/dist";

export interface BlandAltman {
  n: number;
  bias: number;
  sd: number;
  biasCi: [number, number];
  lowerLoa: number;
  upperLoa: number;
  lowerLoaCi: [number, number];
  upperLoaCi: [number, number];
  /** 図の点: [平均, 差]。 */
  points: [number, number][];
}

export function blandAltman(a: number[], b: number[]): BlandAltman | { error: string } {
  const n = a.length;
  if (n !== b.length) return { error: "2 つの条件の症例数が違います" };
  if (n < 3) return { error: `症例が ${n} 人しかありません（3 人以上要ります）` };
  if (a.some((v) => !Number.isFinite(v)) || b.some((v) => !Number.isFinite(v))) return { error: "欠損・非数があります" };
  const d = a.map((v, i) => b[i] - v);
  const bias = d.reduce((x, y) => x + y, 0) / n;
  const sd = Math.sqrt(d.reduce((x, y) => x + (y - bias) ** 2, 0) / (n - 1));
  const t = tQuantile(0.975, n - 1);
  const seBias = sd / Math.sqrt(n), seLoa = Math.sqrt(sd * sd * (1 / n + (1.96 * 1.96) / (2 * (n - 1))));
  const lo = bias - 1.96 * sd, hi = bias + 1.96 * sd;
  return {
    n, bias, sd, biasCi: [bias - t * seBias, bias + t * seBias], lowerLoa: lo, upperLoa: hi,
    lowerLoaCi: [lo - t * seLoa, lo + t * seLoa], upperLoaCi: [hi - t * seLoa, hi + t * seLoa],
    points: a.map((v, i) => [(v + b[i]) / 2, d[i]] as [number, number]),
  };
}
