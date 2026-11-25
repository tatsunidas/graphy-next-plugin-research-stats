/**
 * Levene / Brown–Forsythe の等分散性検定（scipy.stats.levene と同じ定義）。
 * 各観測の「群の中心からの絶対偏差」に対する一元配置の分散分析の F。
 * 中心が median なら Brown–Forsythe（scipy の既定）、mean なら Levene の原型。
 */
import { betaCdf } from "../survival/dist";

export type LeveneCenter = "median" | "mean";

export interface LeveneResult {
  /** 計算できないときは null。 */
  f: number | null;
  p: number | null;
  df1: number;
  df2: number;
  reason?: string;
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function levene(groups: number[][], center: LeveneCenter = "median"): LeveneResult | { error: string } {
  const k = groups.length;
  if (k < 2) return { error: "2 群以上が要ります" };
  if (groups.some((g) => g.length < 1)) return { error: "空の群があります" };
  if (groups.some((g) => g.some((v) => !Number.isFinite(v)))) return { error: "有限でない値があります" };
  const N = groups.reduce((s, g) => s + g.length, 0);
  const df1 = k - 1, df2 = N - k;
  if (df2 < 1) return { error: "自由度が足りません（全体の個数が群の数より多い必要があります）" };
  const z = groups.map((g) => {
    const c = center === "median" ? median(g) : g.reduce((a, b) => a + b, 0) / g.length;
    return g.map((v) => Math.abs(v - c));
  });
  const zbar = z.map((g) => g.reduce((a, b) => a + b, 0) / g.length);
  const zall = z.reduce((s, g) => s + g.reduce((a, b) => a + b, 0), 0) / N;
  let num = 0, den = 0;
  for (let i = 0; i < k; i++) {
    num += z[i].length * (zbar[i] - zall) ** 2;
    for (const v of z[i]) den += (v - zbar[i]) ** 2;
  }
  if (!(den > 0)) return { f: null, p: null, df1, df2, reason: "群の中の偏差がすべて同じで、F が定まりません" };
  const f = (num / df1) / (den / df2);
  // 上側確率 = I_{d2/(d2+d1 f)}(d2/2, d1/2)（1 − CDF の桁落ちを避ける）
  return { f, p: betaCdf(df2 / (df2 + df1 * f), df2 / 2, df1 / 2), df1, df2 };
}
