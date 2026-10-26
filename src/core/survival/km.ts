/**
 * Kaplan–Meier 推定。分散は Greenwood、信頼区間は log(−log) 変換（既定 95%）。
 * 中央値は S(t) <= 0.5 となる最小の観察時間（無ければ null＝到達せず）。
 */
import { normalQuantile } from "./dist";
import type { Surv } from "./types";

export interface KmPoint {
  time: number;
  atRisk: number;
  events: number;
  censored: number;
  surv: number;
  se: number;
  lower: number;
  upper: number;
}

export interface KmResult {
  points: KmPoint[];
  n: number;
  events: number;
  median: number | null;
}

export function kaplanMeier(s: Surv, level = 0.95): KmResult {
  const z = normalQuantile(1 - (1 - level) / 2);
  const times = [...new Set(s.time)].sort((a, b) => a - b);
  const d = new Map<number, number>(), c = new Map<number, number>();
  for (let i = 0; i < s.time.length; i++) {
    const m = s.event[i] === 1 ? d : c;
    m.set(s.time[i], (m.get(s.time[i]) ?? 0) + 1);
  }
  let atRisk = s.time.length, surv = 1, gw = 0;
  const points: KmPoint[] = [];
  let median: number | null = null;
  for (const t of times) {
    const di = d.get(t) ?? 0, ci = c.get(t) ?? 0;
    if (di > 0) {
      surv *= 1 - di / atRisk;
      gw += atRisk > di ? di / (atRisk * (atRisk - di)) : Infinity;
    }
    let lower: number, upper: number;
    if (surv >= 1) { lower = 1; upper = 1; }
    else if (surv <= 0) { lower = 0; upper = 0; }
    else {
      const seLL = Math.sqrt(gw) / Math.abs(Math.log(surv));
      lower = Math.pow(surv, Math.exp(z * seLL));
      upper = Math.pow(surv, Math.exp(-z * seLL));
    }
    const se = surv > 0 && Number.isFinite(gw) ? surv * Math.sqrt(gw) : 0;
    points.push({ time: t, atRisk, events: di, censored: ci, surv, se, lower, upper });
    if (median === null && surv <= 0.5) median = t;
    atRisk -= di + ci;
  }
  return { points, n: s.time.length, events: s.event.reduce((a, b) => a + b, 0), median };
}

/** 時刻 t でのリスク集合の人数（time >= t）。リスク表に使う。 */
export function atRiskAt(s: Surv, t: number): number {
  let n = 0;
  for (const x of s.time) if (x >= t) n++;
  return n;
}

/** 時刻 t での生存確率（階段関数。最初のイベントより前は 1）。 */
export function survAt(km: KmResult, t: number): number {
  let v = 1;
  for (const p of km.points) {
    if (p.time > t) break;
    v = p.surv;
  }
  return v;
}
