/**
 * Cox の部分尤度（同時刻のイベントは Breslow）と、線形予測子 η についての勾配・対角のヘッセ行列。
 * LASSO-Cox・Cox ブースティング・Schoenfeld 残差で使う。罰則なしの coxFit は Efron なので、同時刻があると値が違う。
 */
import type { Surv } from "../survival/types";

export interface Risk {
  /** 時間の昇順に並べた行の位置。 */
  order: number[];
  /** 同じ時間の行のまとまり（order 上の [start, end)）と、その中のイベント数。 */
  groups: { start: number; end: number; d: number }[];
}

export function riskSets(s: Surv): Risk {
  const order = Array.from({ length: s.time.length }, (_, i) => i).sort((a, b) => s.time[a] - s.time[b] || a - b);
  const groups: Risk["groups"] = [];
  let i = 0;
  while (i < order.length) {
    let j = i, d = 0;
    while (j < order.length && s.time[order[j]] === s.time[order[i]]) { d += s.event[order[j]]; j++; }
    groups.push({ start: i, end: j, d });
    i = j;
  }
  return { order, groups };
}

export interface BreslowEval {
  /** 対数部分尤度。 */
  ll: number;
  /** ∂ll/∂η_i（マルチンゲール残差: δ_i − exp(η_i)·累積ハザード）。 */
  g: number[];
  /** −∂²ll/∂η_i²（対角だけ）。 */
  h: number[];
}

/** η は s の並びの線形予測子。 */
export function breslow(s: Surv, r: Risk, eta: number[]): BreslowEval {
  const n = eta.length;
  let c = -Infinity;
  for (const e of eta) if (e > c) c = e;
  if (!Number.isFinite(c)) c = 0;
  const w = eta.map((e) => Math.exp(e - c));
  const G = r.groups.length;
  const S0 = new Array<number>(G);
  let acc = 0;
  for (let k = G - 1; k >= 0; k--) {
    for (let q = r.groups[k].start; q < r.groups[k].end; q++) acc += w[r.order[q]];
    S0[k] = acc;
  }
  let ll = 0, H1 = 0, H2 = 0;
  const g = new Array<number>(n), h = new Array<number>(n);
  for (let k = 0; k < G; k++) {
    const grp = r.groups[k];
    if (grp.d > 0) {
      ll -= grp.d * (Math.log(S0[k]) + c);
      H1 += grp.d / S0[k];
      H2 += grp.d / (S0[k] * S0[k]);
    }
    for (let q = grp.start; q < grp.end; q++) {
      const i = r.order[q];
      if (s.event[i] === 1) ll += eta[i];
      g[i] = s.event[i] - w[i] * H1;
      h[i] = w[i] * H1 - w[i] * w[i] * H2;
    }
  }
  return { ll, g, h };
}

/** 対数部分尤度だけ。 */
export const breslowLl = (s: Surv, r: Risk, eta: number[]): number => breslow(s, r, eta).ll;

export const subSurv = (s: Surv, ix: number[]): Surv => ({ time: ix.map((i) => s.time[i]), event: ix.map((i) => s.event[i]) });
