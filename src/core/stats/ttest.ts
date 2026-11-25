/** t 検定（Student・Welch・対応あり）。両側 p。t 分布の確率は不完全ベータ関数から。 */
import { betaCdf } from "../survival/dist";

export interface TTestResult {
  /** 計算できないときは t・p とも null。 */
  t: number | null;
  df: number | null;
  p: number | null;
  /** 平均の差（x − y。対応ありは差の平均）。 */
  meanDiff: number;
  reason?: string;
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const ss = (v: number[], m: number) => v.reduce((a, b) => a + (b - m) ** 2, 0);

/** t 分布の両側確率 P(|T| >= |t|)。 */
export function tTwoSided(t: number, df: number): number {
  return betaCdf(df / (df + t * t), df / 2, 0.5);
}

function finish(meanDiff: number, se: number, df: number): TTestResult {
  if (!(se > 0)) return { t: null, df: Number.isFinite(df) ? df : null, p: null, meanDiff, reason: "標準誤差が 0 で t が定まりません（値が同じ）" };
  const t = meanDiff / se;
  return { t, df, p: tTwoSided(t, df), meanDiff };
}

function bad(x: number[]): string | null {
  return x.some((v) => !Number.isFinite(v)) ? "有限でない値があります" : null;
}

export function studentT(x: number[], y: number[]): TTestResult | { error: string } {
  const e = bad(x) ?? bad(y);
  if (e) return { error: e };
  const n1 = x.length, n2 = y.length;
  if (n1 < 1 || n2 < 1 || n1 + n2 < 3) return { error: "Student の t 検定には合計 3 個以上の値が要ります" };
  const m1 = mean(x), m2 = mean(y);
  const df = n1 + n2 - 2;
  const sp2 = (ss(x, m1) + ss(y, m2)) / df;
  return finish(m1 - m2, Math.sqrt(sp2 * (1 / n1 + 1 / n2)), df);
}

export function welchT(x: number[], y: number[]): TTestResult | { error: string } {
  const e = bad(x) ?? bad(y);
  if (e) return { error: e };
  const n1 = x.length, n2 = y.length;
  if (n1 < 2 || n2 < 2) return { error: "Welch の t 検定には各群 2 個以上の値が要ります" };
  const m1 = mean(x), m2 = mean(y);
  const a = ss(x, m1) / (n1 - 1) / n1, b = ss(y, m2) / (n2 - 1) / n2;
  const se2 = a + b;
  if (!(se2 > 0)) return finish(m1 - m2, 0, NaN);
  const df = (se2 * se2) / ((a * a) / (n1 - 1) + (b * b) / (n2 - 1));
  return finish(m1 - m2, Math.sqrt(se2), df);
}

/** 対応のある t 検定。x[i] と y[i] が同じ対象。 */
export function pairedT(x: number[], y: number[]): TTestResult | { error: string } {
  if (x.length !== y.length) return { error: "対の個数が違います" };
  const e = bad(x) ?? bad(y);
  if (e) return { error: e };
  const n = x.length;
  if (n < 2) return { error: "対応のある t 検定には 2 対以上が要ります" };
  const d = x.map((v, i) => v - y[i]);
  const m = mean(d);
  return finish(m, Math.sqrt(ss(d, m) / (n - 1) / n), n - 1);
}
