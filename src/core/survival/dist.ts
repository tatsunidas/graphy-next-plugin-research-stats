/**
 * 確率分布の関数（外部の統計ライブラリに頼らない）。すべて不完全ガンマ関数の上側 Q(a, x) から作る。
 * 精度は参照実装（scipy）との突き合わせ（test/survival.dist.test.ts）で確かめる。
 */

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** 正則化された上側不完全ガンマ関数 Q(a, x)。 */
export function gammaQ(a: number, x: number): number {
  if (!(a > 0) || x < 0 || Number.isNaN(x)) return NaN;
  if (x === 0) return 1;
  if (x < a + 1) {
    // P(a, x) の級数
    let sum = 1 / a, term = sum, n = a;
    for (let i = 0; i < 1000; i++) {
      n += 1;
      term *= x / n;
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-17) break;
    }
    return Math.max(0, 1 - sum * Math.exp(-x + a * Math.log(x) - lgamma(a)));
  }
  // 連分数（Lentz 法）
  const tiny = 1e-300;
  let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}

/** カイ二乗分布の上側確率。 */
export function chi2Sf(x: number, df: number): number {
  if (x <= 0) return 1;
  return gammaQ(df / 2, x / 2);
}

/** 標準正規分布の両側確率 P(|Z| >= |z|)。 */
export function normalTwoSided(z: number): number {
  return gammaQ(0.5, (z * z) / 2);
}

export function normalCdf(z: number): number {
  const half = 0.5 * gammaQ(0.5, (z * z) / 2); // P(Z > |z|)
  return z >= 0 ? 1 - half : half;
}

/** 標準正規分布の分位点（二分法。CI の係数に使うだけなので速さは要らない）。 */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must be in (0, 1): ${p}`);
  let lo = -40, hi = 40;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (normalCdf(mid) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

function betacf(a: number, b: number, x: number): number {
  const tiny = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

/** 正則化不完全ベータ関数 I_x(a, b)（ベータ分布の累積分布）。 */
export function betaCdf(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** ベータ分布の分位点（二分法）。Clopper–Pearson の信頼区間とモンテカルロの抽出に使う。 */
export function betaQuantile(p: number, a: number, b: number): number {
  if (!(p >= 0 && p <= 1)) throw new RangeError(`p must be in [0, 1]: ${p}`);
  let lo = 0, hi = 1;
  for (let i = 0; i < 64; i++) {
    const mid = (lo + hi) / 2;
    if (betaCdf(mid, a, b) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
