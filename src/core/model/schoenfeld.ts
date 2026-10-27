/**
 * Schoenfeld 残差と比例ハザードの検定（Grambsch–Therneau 1994 の近似）。
 * 残差: イベントごとに x_i − x̄(t_i)。x̄ はその時刻のリスク集合の exp(η) で重み付けた平均（同時刻は Breslow と同じ集合）。
 * スケールした残差: β + d·V·r_i（d はイベント数、V は係数の共分散）。
 * 検定: 時間の変換 g は 1 − KM(t−)（左連続の Kaplan–Meier）。変数ごとに
 *   χ² = (Σ (g_i − ḡ)·(d·V·r_i)_j)² / (V_jj · d · Σ(g_i − ḡ)²)（自由度 1）、全体は d·uVuᵀ/Σ(g_i − ḡ)²（u = Σ(g_i − ḡ) r_i、自由度 p）。
 * R の survival 3.x の cox.zph は別の式（スコア検定）なので、同じ値にはならない。
 */
import { chi2Sf } from "../survival/dist";
import type { Surv } from "../survival/types";
import { riskSets } from "./breslow";

export interface SchoenfeldTest {
  chi2: number;
  df: number;
  p: number;
}

export interface SchoenfeldResult {
  /** イベントごと（時間の昇順）の行の位置（s の並び）と時間。 */
  rows: number[];
  times: number[];
  /** 時間の変換 1 − KM(t−)。 */
  transformed: number[];
  resid: number[][];
  scaled: number[][];
  tests: SchoenfeldTest[];
  global: SchoenfeldTest;
  /** スケールした残差の移動平均（窓は ⌈√d⌉ 個のイベント、時間の昇順）。図の線に使う。 */
  smooth: number[][];
  window: number;
}

/** Z は当てはめに使った共変量（s の並び）、beta・cov はその当てはめの係数と共分散。 */
export function schoenfeld(s: Surv, Z: number[][], beta: number[], cov: number[][]): SchoenfeldResult | { error: string } {
  const p = beta.length;
  const r = riskSets(s);
  const eta = Z.map((z) => z.reduce((a, v, j) => a + v * beta[j], 0));
  const c = Math.max(...eta);
  const w = eta.map((e) => Math.exp(e - c));
  const G = r.groups.length;
  const S0 = new Array<number>(G), S1: number[][] = new Array(G);
  let a0 = 0;
  const a1 = new Array<number>(p).fill(0);
  for (let k = G - 1; k >= 0; k--) {
    for (let q = r.groups[k].start; q < r.groups[k].end; q++) {
      const i = r.order[q];
      a0 += w[i];
      for (let j = 0; j < p; j++) a1[j] += w[i] * Z[i][j];
    }
    S0[k] = a0;
    S1[k] = [...a1];
  }
  const rows: number[] = [], times: number[] = [], transformed: number[] = [], resid: number[][] = [];
  let km = 1, atRisk = s.time.length;
  for (let k = 0; k < G; k++) {
    const grp = r.groups[k];
    const t = s.time[r.order[grp.start]];
    for (let q = grp.start; q < grp.end; q++) {
      const i = r.order[q];
      if (s.event[i] !== 1) continue;
      rows.push(i);
      times.push(t);
      transformed.push(1 - km);
      resid.push(Z[i].map((v, j) => v - S1[k][j] / S0[k]));
    }
    if (grp.d > 0) km *= 1 - grp.d / atRisk;
    atRisk -= grp.end - grp.start;
  }
  const d = rows.length;
  if (d < 2) return { error: "イベントが 2 件未満なので検定できません" };
  const gm = transformed.reduce((a, b) => a + b, 0) / d;
  const xx = transformed.map((g) => g - gm);
  const sxx = xx.reduce((a, b) => a + b * b, 0);
  if (!(sxx > 0)) return { error: "イベントの時間がすべて同じなので検定できません" };
  const Vr = resid.map((ri) => cov.map((row) => row.reduce((a, v, j) => a + v * ri[j], 0) * d));
  const scaled = Vr.map((v) => v.map((x, j) => x + beta[j]));
  const tests: SchoenfeldTest[] = [];
  for (let j = 0; j < p; j++) {
    const num = xx.reduce((a, x, i) => a + x * Vr[i][j], 0);
    const chi2 = (num * num) / (cov[j][j] * d * sxx);
    tests.push({ chi2, df: 1, p: chi2Sf(chi2, 1) });
  }
  const u = new Array<number>(p).fill(0);
  xx.forEach((x, i) => { for (let j = 0; j < p; j++) u[j] += x * resid[i][j]; });
  let quad = 0;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) quad += u[a] * cov[a][b] * u[b];
  const gchi = (d * quad) / sxx;
  const window = Math.ceil(Math.sqrt(d));
  const half = Math.floor(window / 2);
  const smooth = scaled.map((_, i) => {
    const lo = Math.max(0, i - half), hi = Math.min(d, lo + window);
    const from = Math.max(0, hi - window);
    return Array.from({ length: p }, (_, j) => { let sm = 0; for (let q = from; q < hi; q++) sm += scaled[q][j]; return sm / (hi - from); });
  });
  return { rows, times, transformed, resid, scaled, tests, global: { chi2: gchi, df: p, p: chi2Sf(gchi, p) }, smooth, window };
}
