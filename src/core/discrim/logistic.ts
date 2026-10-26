/**
 * ロジスティック回帰（切片つき、Newton 法・ステップ半減）。完全分離・収束しない・特異は**エラーで返す**。
 * 特徴量の組み合わせ（R2）、較正の傾きと切片に使う。罰則は付けない。
 */
import { normalTwoSided } from "../survival/dist";
import { inverse, solve } from "../survival/linalg";

export interface LogisticFit {
  /** [切片, 各変数の係数]。 */
  beta: number[];
  se: number[];
  p: number[];
  loglik: number;
  iterations: number;
  predict(x: number[]): number;
}

export type LogisticResult = { ok: true; fit: LogisticFit } | { ok: false; error: string };

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

function evaluate(X: number[][], y: number[], beta: number[]): { ll: number; g: number[]; I: number[][] } {
  const k = beta.length;
  let ll = 0;
  const g = new Array<number>(k).fill(0);
  const I = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  for (let i = 0; i < X.length; i++) {
    let z = beta[0];
    for (let a = 1; a < k; a++) z += beta[a] * X[i][a - 1];
    const p = sigmoid(z);
    // log(1+e^z) を桁あふれしないように
    ll += y[i] * z - (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
    const w = p * (1 - p), r = y[i] - p;
    for (let a = 0; a < k; a++) {
      const xa = a === 0 ? 1 : X[i][a - 1];
      g[a] += r * xa;
      for (let b = 0; b <= a; b++) I[a][b] += w * xa * (b === 0 ? 1 : X[i][b - 1]);
    }
  }
  for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) I[a][b] = I[b][a];
  return { ll, g, I };
}

export function logisticFit(X: number[][], y: number[]): LogisticResult {
  const n = X.length, p = X[0]?.length ?? 0;
  if (n !== y.length) return { ok: false, error: "症例数と変数の行数が違います" };
  if (p === 0) return { ok: false, error: "変数がありません" };
  const pos = y.reduce((a, b) => a + b, 0);
  if (pos === 0 || pos === n) return { ok: false, error: "片方の群しかありません" };
  for (const row of X) for (const v of row) if (!Number.isFinite(v)) return { ok: false, error: "変数に欠損・非数があります" };
  let beta = new Array<number>(p + 1).fill(0);
  let cur = evaluate(X, y, beta);
  let converged = false, it = 0;
  for (; it < 100; it++) {
    const step = solve(cur.I, cur.g);
    if (!step) return { ok: false, error: "情報行列が特異です（変数が線形従属、など）" };
    let scale = 1, next: ReturnType<typeof evaluate> | null = null, nb = beta;
    for (let h = 0; h < 30; h++) {
      nb = beta.map((b, a) => b + scale * step[a]);
      next = evaluate(X, y, nb);
      if (Number.isFinite(next.ll) && next.ll >= cur.ll - 1e-12) break;
      next = null;
      scale /= 2;
    }
    if (!next) return { ok: false, error: "尤度が改善せず収束しませんでした" };
    const d = Math.abs(next.ll - cur.ll);
    beta = nb;
    cur = next;
    if (d < 1e-11 * (Math.abs(cur.ll) + 1)) { converged = true; it++; break; }
  }
  if (!converged) return { ok: false, error: "100 回で収束しませんでした（完全分離の疑い）" };
  const cov = inverse(cur.I);
  if (!cov) return { ok: false, error: "情報行列が特異です" };
  const se = beta.map((_, a) => Math.sqrt(cov[a][a]));
  if (beta.some((b) => Math.abs(b) > 30) || se.some((v) => !Number.isFinite(v) || v > 1e3)) {
    return { ok: false, error: "係数が発散しました（完全分離の疑い）" };
  }
  const b = beta;
  return {
    ok: true,
    fit: {
      beta: b, se, p: b.map((v, a) => normalTwoSided(v / se[a])), loglik: cur.ll, iterations: it,
      predict: (x) => sigmoid(b.reduce((acc, v, a) => acc + v * (a === 0 ? 1 : x[a - 1]), 0)),
    },
  };
}
