/**
 * Wilcoxon の符号順位検定（対応あり、両側）。scipy.stats.wilcoxon の既定
 * （zero_method="wilcox"＝差が 0 のペアを除く、correction=False、method="auto"）と同じ定義。
 *
 * auto の振る舞い（scipy 1.15 と同じ）:
 *  - 組の数（0 差を含む）が 50 を超える → 正規近似（"approx"）
 *  - 0 差もタイもない → 正確な分布（"exact"。部分和の数え上げ）
 *  - 0 差かタイがあり、組の数が 13 以下 → 符号の全割り当て 2^n を数え上げる並べ替え検定（"permutation"）
 *  - それ以外（0 差かタイがあり 14 以上）→ 正規近似（"approx"。タイ補正つき）
 */
import { normalTwoSided } from "../survival/dist";

export type WilcoxonMethod = "exact" | "permutation" | "approx";

export interface WilcoxonResult {
  /** 計算できないときは p が null。 */
  p: number | null;
  /** min(W+, W−)。計算できないときは null。 */
  statistic: number | null;
  /** 正規近似のときの z（そうでなければ null）。 */
  z: number | null;
  method: WilcoxonMethod | null;
  /** 除いた 0 差のペアの数。 */
  zerosDropped: number;
  /** 検定に使ったペアの数（0 差を除いたあと）。 */
  n: number;
  reason?: string;
}

function avgRanks(a: number[]): { r: number[]; tieTerm: number; ties: boolean } {
  const o = a.map((_, i) => i).sort((i, j) => a[i] - a[j]);
  const r = new Array<number>(a.length);
  let tieTerm = 0, ties = false;
  for (let i = 0; i < o.length;) {
    let j = i;
    while (j < o.length && a[o[j]] === a[o[i]]) j++;
    for (let q = i; q < j; q++) r[o[q]] = (i + 1 + j) / 2;
    const t = j - i;
    if (t > 1) ties = true;
    tieTerm += t ** 3 - t;
    i = j;
  }
  return { r, tieTerm, ties };
}

/** 1..n の部分和の個数（W+ の帰無分布の度数）。n <= 50 で 2^50 < 2^53 なので整数のまま正確。 */
function subsetSumCounts(n: number): number[] {
  const max = (n * (n + 1)) / 2;
  const c = new Array<number>(max + 1).fill(0);
  c[0] = 1;
  for (let k = 1; k <= n; k++) for (let s = max; s >= k; s--) c[s] += c[s - k];
  return c;
}

export function wilcoxonSignedRank(x: number[], y: number[]): WilcoxonResult | { error: string } {
  if (x.length !== y.length) return { error: "対の個数が違います" };
  const total = x.length;
  const d = x.map((v, i) => v - y[i]);
  if (d.some((v) => !Number.isFinite(v))) return { error: "有限でない値があります" };
  const nz = d.filter((v) => v !== 0);
  const zerosDropped = total - nz.length;
  const n = nz.length;
  if (n === 0) return { p: null, statistic: null, z: null, method: null, zerosDropped, n, reason: "差が 0 でないペアがありません" };

  const { r, tieTerm, ties } = avgRanks(nz.map(Math.abs));
  let rPlus = 0, rMinus = 0;
  nz.forEach((v, i) => { if (v > 0) rPlus += r[i]; else rMinus += r[i]; });
  const statistic = Math.min(rPlus, rMinus);

  let method: WilcoxonMethod;
  if (total > 50) method = "approx";
  else if (!ties && zerosDropped === 0) method = "exact";
  else if (total <= 13) method = "permutation";
  else method = "approx";

  if (method === "approx") {
    const mn = (n * (n + 1)) / 4;
    const se = Math.sqrt((n * (n + 1) * (2 * n + 1) - tieTerm / 2) / 24);
    if (!(se > 0)) return { p: null, statistic, z: null, method, zerosDropped, n, reason: "標準誤差が 0 です" };
    const z = (rPlus - mn) / se;
    return { p: Math.min(1, normalTwoSided(z)), statistic, z, method, zerosDropped, n };
  }
  if (method === "exact") {
    const c = subsetSumCounts(n);
    const tot = 2 ** n;
    const lo = Math.ceil(rPlus), hi = Math.floor(rPlus);
    let cdf = 0, sf = 0;
    for (let s = 0; s <= lo; s++) cdf += c[s];
    for (let s = hi; s < c.length; s++) sf += c[s];
    return { p: Math.min(1, (2 * Math.min(cdf, sf)) / tot), statistic, z: null, method, zerosDropped, n };
  }
  // 符号の全割り当て。W+ は順位の部分和（順位は 0.5 刻みなので 2 倍して整数で数える）
  const r2 = r.map((v) => Math.round(2 * v));
  const sums = new Map<number, number>([[0, 1]]);
  let cur = sums;
  for (const v of r2) {
    const next = new Map<number, number>();
    for (const [s, k] of cur) {
      next.set(s, (next.get(s) ?? 0) + k);
      next.set(s + v, (next.get(s + v) ?? 0) + k);
    }
    cur = next;
  }
  const obs = 2 * rPlus;
  const tol = Math.abs(obs) * Number.EPSILON * 100;
  let less = 0, greater = 0;
  for (const [s, k] of cur) {
    if (s <= obs + tol) less += k;
    if (s >= obs - tol) greater += k;
  }
  return { p: Math.min(1, (2 * Math.min(less, greater)) / 2 ** n), statistic, z: null, method, zerosDropped, n };
}
