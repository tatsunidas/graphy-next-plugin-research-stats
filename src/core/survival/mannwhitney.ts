/**
 * Mann–Whitney U 検定（正規近似・同順位補正・連続性補正。scipy の method="asymptotic" と同じ定義）。
 * 小標本では近似が粗い（n が小さいと正確確率と差が出る）ので、呼び出し側は症例数を併記する。
 */
import { normalTwoSided } from "./dist";

export interface MannWhitney {
  u1: number;
  u2: number;
  z: number;
  p: number;
  n1: number;
  n2: number;
}

export function mannWhitney(x: number[], y: number[]): MannWhitney | { error: string } {
  const n1 = x.length, n2 = y.length;
  if (n1 === 0 || n2 === 0) return { error: "片方の群が空です" };
  const all = [...x.map((v) => ({ v, g: 0 })), ...y.map((v) => ({ v, g: 1 }))].sort((a, b) => a.v - b.v);
  const N = n1 + n2;
  const ranks = new Array<number>(N);
  let tieTerm = 0;
  for (let i = 0; i < N;) {
    let j = i;
    while (j < N && all[j].v === all[i].v) j++;
    const r = (i + 1 + j) / 2;
    for (let q = i; q < j; q++) ranks[q] = r;
    const t = j - i;
    tieTerm += t ** 3 - t;
    i = j;
  }
  let r1 = 0;
  for (let i = 0; i < N; i++) if (all[i].g === 0) r1 += ranks[i];
  const u1 = r1 - (n1 * (n1 + 1)) / 2, u2 = n1 * n2 - u1;
  const mu = (n1 * n2) / 2;
  const variance = ((n1 * n2) / 12) * (N + 1 - tieTerm / (N * (N - 1)));
  if (!(variance > 0)) return { error: "すべての値が同じです" };
  const u = Math.max(u1, u2);
  const z = (u - mu - 0.5) / Math.sqrt(variance);
  return { u1, u2, z, p: Math.min(1, normalTwoSided(z)), n1, n2 };
}
