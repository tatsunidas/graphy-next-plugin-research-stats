/**
 * サポートベクターマシン（C-SVC、RBF カーネル、二値）。scikit-learn の SVC（中身は libsvm）と同じ双対問題を解く:
 *   min ½ αᵀQα − Σα,  0 ≤ α ≤ C,  Σ yα = 0,  Q_ij = y_i y_j K(x_i, x_j),  K = exp(−γ‖x_i − x_j‖²),  y ∈ {−1, +1}
 * 解き方は SMO。作業集合は二次の情報で選ぶ（Fan, Chen & Lin 2005, JMLR 6:1889。libsvm の方式）。縮小（shrinking）は
 * しない（解は変わらず、速さだけが変わる）。止める条件は、最大の KKT 違反 m(α) − M(α) < tol。
 * b は自由なサポートベクター（0 < α < C）の y∇f の平均（無ければ上限と下限の中点）。
 * 出力は決定値 f(x) = Σ α_i y_i K(x_i, x) − ρ（正ならクラス 1）。確率ではない。
 */

/** sklearn の既定 1e-3 より厳しくして、真の最適解との差を AUC の比較で無視できる大きさにする。 */
export const SVM_TOL = 1e-6;
const TAU = 1e-12;

export interface SvmOptions {
  C: number;
  gamma: number;
  tol?: number;
  /** **試験専用**の誤り: ρ を全サポートベクターの平均にする。 */
  wrongRho?: boolean;
}

export interface Svm {
  /** 決定値。x は学習に渡した行と同じ長さ（標準化済み）。 */
  decision(x: ArrayLike<number>): number;
  /** 学習に渡した行の並びで、α > 0 の位置。 */
  support: number[];
  alpha: number[];
  rho: number;
  iterations: number;
  converged: boolean;
}

/** Z は学習の行（標準化済み）、y は 0/1。 */
export function svmFit(Z: number[][], y: number[], o: SvmOptions): Svm {
  if (!(o.C > 0)) throw new RangeError("C は正の数です");
  if (!(o.gamma > 0)) throw new RangeError("γ は正の数です");
  const l = Z.length, C = o.C, eps = o.tol ?? SVM_TOL;
  if (!y.includes(0) || !y.includes(1)) throw new RangeError("学習の行に両方のクラスが要ります");
  const s = y.map((v) => (v === 1 ? 1 : -1));
  const rbf = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    let d = 0;
    for (let k = 0; k < a.length; k++) { const t = a[k] - b[k]; d += t * t; }
    return Math.exp(-o.gamma * d);
  };
  const K = new Float64Array(l * l);
  for (let i = 0; i < l; i++) for (let j = i; j < l; j++) { const v = rbf(Z[i], Z[j]); K[i * l + j] = v; K[j * l + i] = v; }
  const Q = (i: number, j: number) => s[i] * s[j] * K[i * l + j];
  const alpha = new Float64Array(l), G = new Float64Array(l).fill(-1);
  const upper = (t: number) => alpha[t] >= C, lower = (t: number) => alpha[t] <= 0;
  const maxIter = Math.max(10_000_000, 100 * l); // libsvm と同じ上限
  let iter = 0, converged = false;
  while (iter < maxIter) {
    let Gmax = -Infinity, Gmax2 = -Infinity, i = -1, j = -1, objMin = Infinity;
    for (let t = 0; t < l; t++) {
      if (s[t] === 1) { if (!upper(t) && -G[t] >= Gmax) { Gmax = -G[t]; i = t; } }
      else if (!lower(t) && G[t] >= Gmax) { Gmax = G[t]; i = t; }
    }
    for (let t = 0; i >= 0 && t < l; t++) {
      if (s[t] === 1) {
        if (lower(t)) continue;
        const gd = Gmax + G[t];
        if (G[t] >= Gmax2) Gmax2 = G[t];
        if (gd > 0) {
          const qc = K[i * l + i] + K[t * l + t] - 2 * s[i] * Q(i, t);
          const od = -(gd * gd) / (qc > 0 ? qc : TAU);
          if (od <= objMin) { j = t; objMin = od; }
        }
      } else {
        if (upper(t)) continue;
        const gd = Gmax - G[t];
        if (-G[t] >= Gmax2) Gmax2 = -G[t];
        if (gd > 0) {
          const qc = K[i * l + i] + K[t * l + t] + 2 * s[i] * Q(i, t);
          const od = -(gd * gd) / (qc > 0 ? qc : TAU);
          if (od <= objMin) { j = t; objMin = od; }
        }
      }
    }
    if (Gmax + Gmax2 < eps || j < 0) { converged = true; break; }
    iter++;
    const ai = alpha[i], aj = alpha[j];
    const Qij = Q(i, j);
    if (s[i] !== s[j]) {
      let qc = K[i * l + i] + K[j * l + j] + 2 * Qij;
      if (qc <= 0) qc = TAU;
      const delta = (-G[i] - G[j]) / qc, diff = alpha[i] - alpha[j];
      alpha[i] += delta; alpha[j] += delta;
      if (diff > 0) { if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff; } }
      else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff; }
      if (diff > 0) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = C - diff; } }
      else if (alpha[j] > C) { alpha[j] = C; alpha[i] = C + diff; }
    } else {
      let qc = K[i * l + i] + K[j * l + j] - 2 * Qij;
      if (qc <= 0) qc = TAU;
      const delta = (G[i] - G[j]) / qc, sum = alpha[i] + alpha[j];
      alpha[i] -= delta; alpha[j] += delta;
      if (sum > C) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = sum - C; } }
      else if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum; }
      if (sum > C) { if (alpha[j] > C) { alpha[j] = C; alpha[i] = sum - C; } }
      else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum; }
    }
    const di = alpha[i] - ai, dj = alpha[j] - aj;
    for (let t = 0; t < l; t++) G[t] += Q(i, t) * di + Q(j, t) * dj;
  }
  let ub = Infinity, lb = -Infinity, nFree = 0, sumFree = 0, sumAll = 0, nAll = 0;
  for (let t = 0; t < l; t++) {
    const yG = s[t] * G[t];
    if (alpha[t] > 0) { sumAll += yG; nAll++; }
    if (upper(t)) { if (s[t] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else if (lower(t)) { if (s[t] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else { nFree++; sumFree += yG; }
  }
  const rho = o.wrongRho ? sumAll / nAll : nFree > 0 ? sumFree / nFree : (ub + lb) / 2;
  const support: number[] = [];
  for (let t = 0; t < l; t++) if (alpha[t] > 0) support.push(t);
  const coef = support.map((t) => alpha[t] * s[t]), sv = support.map((t) => Z[t]);
  return {
    support, alpha: Array.from(alpha), rho, iterations: iter, converged,
    decision: (x) => {
      let f = -rho;
      for (let q = 0; q < sv.length; q++) f += coef[q] * rbf(sv[q], x);
      return f;
    },
  };
}

/** γ の "scale"（sklearn と同じ: 1 / (列の数 × 学習の行の全要素の分散［自由度 n］)）。分散が 0 なら 1。 */
export function scaleGamma(Z: number[][]): number {
  let n = 0, s = 0, s2 = 0;
  for (const r of Z) for (const v of r) { n++; s += v; s2 += v * v; }
  const mean = s / n, v = s2 / n - mean * mean;
  return v > 0 ? 1 / (Z[0].length * v) : 1;
}
