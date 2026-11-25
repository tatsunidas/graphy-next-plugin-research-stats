/**
 * Shapiro–Wilk の正規性検定。Royston (1995) の AS R94（scipy.stats.shapiro と同じアルゴリズム）。
 * 3 <= n <= 5000。全値が同じなら計算できない（p は null。0 や 1 にしない）。
 */
import { normalCdf, normalQuantile } from "../survival/dist";

export interface ShapiroResult {
  n: number;
  /** 計算できないときは null。 */
  w: number | null;
  p: number | null;
  /** 計算できない理由。 */
  reason?: string;
}

const C1 = [0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
const C2 = [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
const C3 = [0.544, -0.39978, 0.025054, -6.714e-4];
const C4 = [1.3822, -0.77857, 0.062767, -0.0020322];
const C5 = [-1.5861, -0.31082, -0.083751, 0.0038915];
const C6 = [-0.4803, -0.082676, 0.0030302];
const G = [-2.273, 0.459];

/** c[0] + c[1] x + c[2] x² + … */
function poly(c: number[], x: number): number {
  let r = 0;
  for (let i = c.length - 1; i >= 0; i--) r = r * x + c[i];
  return r;
}

/** 順序統計量の係数の下半分（先頭が最も端の値で、符号は正）。 */
function coefficients(n: number): number[] {
  const nn2 = Math.floor(n / 2);
  if (n === 3) return [Math.SQRT1_2];
  const m: number[] = [];
  let summ2 = 0;
  for (let i = 1; i <= nn2; i++) {
    const v = normalQuantile((i - 0.375) / (n + 0.25));
    m.push(v);
    summ2 += v * v;
  }
  summ2 *= 2;
  const ssumm2 = Math.sqrt(summ2);
  const rsn = 1 / Math.sqrt(n);
  const a1 = poly(C1, rsn) - m[0] / ssumm2;
  const a = [a1];
  let fac: number;
  let i1: number;
  if (n > 5) {
    const a2 = -m[1] / ssumm2 + poly(C2, rsn);
    fac = Math.sqrt((summ2 - 2 * m[0] ** 2 - 2 * m[1] ** 2) / (1 - 2 * a1 ** 2 - 2 * a2 ** 2));
    a.push(a2);
    i1 = 2;
  } else {
    fac = Math.sqrt((summ2 - 2 * m[0] ** 2) / (1 - 2 * a1 ** 2));
    i1 = 1;
  }
  for (let i = i1; i < nn2; i++) a.push(-m[i] / fac);
  return a;
}

export function shapiroWilk(values: number[]): ShapiroResult | { error: string } {
  const n = values.length;
  if (n < 3) return { error: `Shapiro–Wilk には 3 個以上の値が要ります（${n} 個）` };
  if (n > 5000) return { error: `Shapiro–Wilk は 5000 個までです（${n} 個）` };
  if (values.some((v) => !Number.isFinite(v))) return { error: "有限でない値があります" };
  const x = [...values].sort((p, q) => p - q);
  const range = x[n - 1] - x[0];
  if (!(range > 1e-19)) return { n, w: null, p: null, reason: "すべての値が同じです" };

  const half = coefficients(n);
  const a = (i: number) => {
    const j = Math.min(i, n - 1 - i);
    if (i === n - 1 - i) return 0;
    return i < n - 1 - i ? -half[j] : half[j];
  };
  let sx = 0, sa = 0;
  for (let i = 0; i < n; i++) { sx += x[i] / range; sa += a(i); }
  sx /= n; sa /= n;
  let ssa = 0, ssx = 0, sax = 0;
  for (let i = 0; i < n; i++) {
    const asa = a(i) - sa, xsx = x[i] / range - sx;
    ssa += asa * asa; ssx += xsx * xsx; sax += asa * xsx;
  }
  const ssassx = Math.sqrt(ssa * ssx);
  const w1 = ((ssassx - sax) * (ssassx + sax)) / (ssa * ssx);
  const w = 1 - w1;

  if (n === 3) {
    const pw = (6 / Math.PI) * (Math.asin(Math.sqrt(w)) - Math.PI / 3);
    return { n, w, p: Math.max(0, pw) };
  }
  let y = Math.log(w1);
  let mu: number, s: number;
  if (n <= 11) {
    const gamma = poly(G, n);
    if (y >= gamma) return { n, w, p: 1e-99 };
    y = -Math.log(gamma - y);
    mu = poly(C3, n);
    s = Math.exp(poly(C4, n));
  } else {
    const lx = Math.log(n);
    mu = poly(C5, lx);
    s = Math.exp(poly(C6, lx));
  }
  return { n, w, p: 1 - normalCdf((y - mu) / s) };
}
