/**
 * 制限付き平均生存時間 RMST(τ) = ∫₀^τ S(t) dt。標準誤差は
 * Var = Σ_{t_i ≤ τ} ( ∫_{t_i}^{τ} S(u) du )² · d_i / (n_i (n_i − d_i))。
 * τ は、比べる群のどれの最大観察時間も超えてはいけない（超えると推定が定義できない）。
 */
import { normalTwoSided } from "./dist";
import { kaplanMeier, type KmResult } from "./km";
import type { Surv } from "./types";

export interface Rmst {
  tau: number;
  value: number;
  se: number;
}

export function maxFollowUp(s: Surv): number {
  return Math.max(...s.time);
}

export function rmst(km: KmResult, tau: number): Rmst | { error: string } {
  const last = km.points[km.points.length - 1]?.time ?? 0;
  if (!(tau > 0)) return { error: "τ は正の値にしてください" };
  if (tau > last) return { error: `τ（${tau}）が最大観察時間（${last}）を超えています` };
  // 階段関数の積分: 区間 [t_k, t_{k+1}) で S = S(t_k)
  const knots = [0, ...km.points.map((p) => p.time)];
  const level = [1, ...km.points.map((p) => p.surv)];
  const area = (from: number, to: number): number => {
    let a = 0;
    for (let k = 0; k < knots.length; k++) {
      const lo = Math.max(knots[k], from), hi = Math.min(k + 1 < knots.length ? knots[k + 1] : Infinity, to);
      if (hi > lo) a += level[k] * (hi - lo);
    }
    return a;
  };
  const value = area(0, tau);
  let variance = 0;
  for (const p of km.points) {
    if (p.time > tau || p.events === 0) continue;
    const denom = p.atRisk * (p.atRisk - p.events);
    if (denom <= 0) continue;
    variance += (area(p.time, tau) ** 2 * p.events) / denom;
  }
  return { tau, value, se: Math.sqrt(variance) };
}

export function rmstDifference(a: Surv, b: Surv, tau: number): { diff: number; se: number; p: number; a: Rmst; b: Rmst } | { error: string } {
  const ra = rmst(kaplanMeier(a), tau), rb = rmst(kaplanMeier(b), tau);
  if ("error" in ra) return { error: `群 1: ${ra.error}` };
  if ("error" in rb) return { error: `群 2: ${rb.error}` };
  const diff = rb.value - ra.value, se = Math.sqrt(ra.se ** 2 + rb.se ** 2);
  return { diff, se, p: se > 0 ? normalTwoSided(diff / se) : NaN, a: ra, b: rb };
}
