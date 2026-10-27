/**
 * Cox 比例ハザードモデル。同時刻のイベントは Efron の近似。Newton–Raphson（ステップ半減つき）。
 * 収束しない・完全分離・特異な情報行列は**エラーで返す**（NaN や発散した係数を結果にしない）。
 */
import { chi2Sf, normalQuantile, normalTwoSided } from "./dist";
import { inverse, solve } from "./linalg";
import type { Surv } from "./types";

export interface CoxFit {
  n: number;
  events: number;
  beta: number[];
  se: number[];
  hr: number[];
  hrLower: number[];
  hrUpper: number[];
  z: number[];
  p: number[];
  loglik: number;
  loglik0: number;
  lrChi2: number;
  lrP: number;
  iterations: number;
  /** 係数の共分散行列（情報行列の逆）。 */
  cov: number[][];
  /** 元の共変量 X に対する線形予測子 X·β（C-index に使う。定数のずれは順位に影響しない）。 */
  linearPredictor: number[];
}

export type CoxResult = { ok: true; fit: CoxFit } | { ok: false; error: string };

interface Prep {
  order: number[];
  groups: { start: number; end: number; d: number }[];
}

function prepare(s: Surv): Prep {
  const order = Array.from({ length: s.time.length }, (_, i) => i).sort((a, b) => s.time[a] - s.time[b]);
  const groups: Prep["groups"] = [];
  let i = 0;
  while (i < order.length) {
    let j = i, d = 0;
    while (j < order.length && s.time[order[j]] === s.time[order[i]]) { d += s.event[order[j]]; j++; }
    groups.push({ start: i, end: j, d });
    i = j;
  }
  return { order, groups };
}

interface Eval { ll: number; g: number[]; I: number[][] }

function evaluate(s: Surv, X: number[][], beta: number[], prep: Prep, withDerivs: boolean): Eval | null {
  const p = beta.length, n = s.time.length;
  const eta = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let e = 0;
    for (let a = 0; a < p; a++) e += X[i][a] * beta[a];
    if (Math.abs(e) > 700) return null;
    eta[i] = e;
  }
  let S0 = 0;
  const S1 = new Array<number>(p).fill(0);
  const S2 = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  let ll = 0;
  const g = new Array<number>(p).fill(0);
  const I = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  for (let gi = prep.groups.length - 1; gi >= 0; gi--) {
    const grp = prep.groups[gi];
    let T0 = 0;
    const T1 = new Array<number>(p).fill(0);
    const T2 = Array.from({ length: p }, () => new Array<number>(p).fill(0));
    for (let q = grp.start; q < grp.end; q++) {
      const i = prep.order[q];
      const w = Math.exp(eta[i]);
      S0 += w;
      for (let a = 0; a < p; a++) {
        S1[a] += w * X[i][a];
        if (withDerivs) for (let b = 0; b <= a; b++) S2[a][b] += w * X[i][a] * X[i][b];
      }
      if (s.event[i] === 1) {
        ll += eta[i];
        T0 += w;
        for (let a = 0; a < p; a++) {
          g[a] += X[i][a];
          T1[a] += w * X[i][a];
          if (withDerivs) for (let b = 0; b <= a; b++) T2[a][b] += w * X[i][a] * X[i][b];
        }
      }
    }
    for (let l = 0; l < grp.d; l++) {
      const phi = l / grp.d;
      const A0 = S0 - phi * T0;
      if (!(A0 > 0)) return null;
      ll -= Math.log(A0);
      for (let a = 0; a < p; a++) {
        const A1a = S1[a] - phi * T1[a];
        g[a] -= A1a / A0;
        if (withDerivs) {
          for (let b = 0; b <= a; b++) {
            const A1b = S1[b] - phi * T1[b];
            const A2 = S2[a][b] - phi * T2[a][b];
            I[a][b] += A2 / A0 - (A1a * A1b) / (A0 * A0);
          }
        }
      }
    }
  }
  if (withDerivs) for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) I[a][b] = I[b][a];
  return { ll, g, I };
}

export function coxFit(s: Surv, X: number[][], level = 0.95): CoxResult {
  const n = s.time.length, p = X[0]?.length ?? 0;
  const events = s.event.reduce((a, b) => a + b, 0);
  if (n !== X.length) return { ok: false, error: "症例数と共変量の行数が違います" };
  if (p === 0) return { ok: false, error: "共変量がありません" };
  if (events === 0) return { ok: false, error: "イベントがありません" };
  for (const row of X) for (const v of row) if (!Number.isFinite(v)) return { ok: false, error: "共変量に欠損・非数があります" };
  // 平均を引いて数値を安定させる（係数・分散は変わらない）
  const mean = new Array<number>(p).fill(0);
  for (const row of X) for (let a = 0; a < p; a++) mean[a] += row[a] / n;
  const Xc = X.map((row) => row.map((v, a) => v - mean[a]));
  for (let a = 0; a < p; a++) {
    if (Xc.every((row) => row[a] === 0)) return { ok: false, error: `共変量 ${a + 1} 番目は全症例で同じ値です` };
  }
  const prep = prepare(s);
  let beta = new Array<number>(p).fill(0);
  const e0 = evaluate(s, Xc, beta, prep, true)!;
  const loglik0 = e0.ll;
  let cur = e0;
  let iterations = 0, converged = false;
  for (; iterations < 100; iterations++) {
    const step = solve(cur.I, cur.g);
    if (!step) return { ok: false, error: "情報行列が特異です（共変量が線形従属、またはイベントが少なすぎます）" };
    let scale = 1, next: Eval | null = null, nb = beta;
    for (let h = 0; h < 30; h++) {
      nb = beta.map((b, a) => b + scale * step[a]);
      next = evaluate(s, Xc, nb, prep, true);
      if (next && next.ll >= cur.ll - 1e-12) break;
      next = null;
      scale /= 2;
    }
    if (!next) return { ok: false, error: "尤度が改善せず収束しませんでした" };
    const dll = Math.abs(next.ll - cur.ll);
    beta = nb;
    cur = next;
    if (dll < 1e-10 * (Math.abs(cur.ll) + 1)) { converged = true; iterations++; break; }
  }
  if (!converged) return { ok: false, error: "100 回で収束しませんでした（完全分離の疑い）" };
  const cov = inverse(cur.I);
  if (!cov) return { ok: false, error: "情報行列が特異です" };
  const se = beta.map((_, a) => Math.sqrt(cov[a][a]));
  if (beta.some((b) => Math.abs(b) > 20) || se.some((v) => !Number.isFinite(v) || v > 1e3)) {
    return { ok: false, error: "係数が発散しました（完全分離の疑い。群のどちらかにイベントが無い、など）" };
  }
  const zc = normalQuantile(1 - (1 - level) / 2);
  const z = beta.map((b, a) => b / se[a]);
  const lrChi2 = Math.max(0, 2 * (cur.ll - loglik0));
  return {
    ok: true,
    fit: {
      n, events, beta, se,
      hr: beta.map(Math.exp),
      hrLower: beta.map((b, a) => Math.exp(b - zc * se[a])),
      hrUpper: beta.map((b, a) => Math.exp(b + zc * se[a])),
      z, p: z.map(normalTwoSided),
      loglik: cur.ll, loglik0, lrChi2, lrP: chi2Sf(lrChi2, p), iterations, cov,
      linearPredictor: X.map((row) => row.reduce((acc, v, a) => acc + v * beta[a], 0)),
    },
  };
}

/** 2 つの入れ子のモデルの −2LL の差（尤度比検定）。追加した変数の数が自由度。 */
export function lrDifference(smaller: CoxFit, larger: CoxFit, extraDf: number): { chi2: number; p: number } {
  const chi2 = Math.max(0, 2 * (larger.loglik - smaller.loglik));
  return { chi2, p: chi2Sf(chi2, extraDf) };
}
