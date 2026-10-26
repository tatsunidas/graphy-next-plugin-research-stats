/**
 * log-rank 検定（k 群。2 群なら df = 1）。各イベント時刻で、観測 O と期待 E、超幾何分散から
 * χ² = (O−E)' V⁻¹ (O−E)（最後の群を除いた k−1 個）を求める。
 */
import { chi2Sf } from "./dist";
import { solve } from "./linalg";
import type { Surv } from "./types";

export interface LogrankResult {
  chi2: number;
  df: number;
  p: number;
  observed: number[];
  expected: number[];
}

/** group は 0..k-1 の群番号。 */
export function logrank(s: Surv, group: number[], k: number): LogrankResult | { error: string } {
  const n = s.time.length;
  const present = new Set(group);
  if (present.size < 2) return { error: "群が 1 つしかありません" };
  const times = [...new Set(s.time.filter((_, i) => s.event[i] === 1))].sort((a, b) => a - b);
  if (!times.length) return { error: "イベントがありません" };
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s.time[a] - s.time[b]);
  const nk = new Array<number>(k).fill(0);
  for (const g of group) nk[g]++;
  const O = new Array<number>(k).fill(0), E = new Array<number>(k).fill(0);
  const V = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  let pos = 0, total = n;
  for (const t of times) {
    // t より前に脱落した人を除く
    while (pos < n && s.time[idx[pos]] < t) { nk[group[idx[pos]]]--; total--; pos++; }
    const dk = new Array<number>(k).fill(0);
    let d = 0;
    for (let q = pos; q < n && s.time[idx[q]] === t; q++) {
      if (s.event[idx[q]] === 1) { dk[group[idx[q]]]++; d++; }
    }
    for (let a = 0; a < k; a++) {
      O[a] += dk[a];
      E[a] += (d * nk[a]) / total;
    }
    if (total > 1) {
      const f = (d * (total - d)) / (total - 1);
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          V[a][b] += f * (nk[a] / total) * ((a === b ? 1 : 0) - nk[b] / total);
        }
      }
    }
  }
  const m = k - 1;
  const diff = Array.from({ length: m }, (_, a) => O[a] - E[a]);
  const Vm = Array.from({ length: m }, (_, a) => V[a].slice(0, m));
  const x = solve(Vm, diff);
  if (!x) return { error: "分散行列が特異です（群の中にイベントが無い、など）" };
  const chi2 = diff.reduce((acc, v, a) => acc + v * x[a], 0);
  return { chi2, df: m, p: chi2Sf(chi2, m), observed: O, expected: E };
}
