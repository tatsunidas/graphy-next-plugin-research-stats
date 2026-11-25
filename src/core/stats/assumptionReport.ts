/**
 * 前提の検定の報告（特徴量ごと）。2 群それぞれの Shapiro–Wilk と、群間の Levene（median）を、
 * 検定の選択（selectUnpaired）とは独立に必ず計算し、選ばれる検定とその理由を並べる。
 * 報告だけで、解析の結果は変えない。α は呼び出し側が決める。計算できない値は null（理由つき）で、0 にしない。
 */
import { levene } from "./levene";
import { selectUnpaired, type Check, type Selection, type TestId } from "./assumptions";
import { shapiroWilk } from "./shapiro";

export interface AssumptionRow {
  index: number;
  feature: string;
  /** 群 0（陰性）・群 1（陽性）の、欠損を除いた個数。 */
  n: [number, number];
  /** 群ごとの Shapiro–Wilk（W は stat、p）。 */
  shapiro: [Check, Check];
  /** Levene（median）。計算できなければ p は null。 */
  levene: Check;
  /** selectUnpaired が選ぶ検定。選べなければ null。 */
  test: TestId | null;
  /** 選んだ検定の p。 */
  p: number | null;
  why: { ja: string; en: string } | null;
  /** 検定できなかった理由（個数不足など）。 */
  reason: string | null;
}

const finite = (X: (number | null)[][], rows: number[], j: number) =>
  rows.map((i) => X[i][j]).filter((v): v is number => v !== null && Number.isFinite(v));

function shapiro(v: number[]): Check {
  const r = shapiroWilk(v);
  if ("error" in r) return { p: null, stat: null, n: v.length, reason: r.error };
  return { p: r.p, stat: r.w, n: v.length, reason: r.reason };
}

/** X は [行][特徴量]。rows0・rows1 は群 0・群 1 の行番号（読むのはこれだけ）。 */
export function assumptionReport(X: (number | null)[][], names: string[], rows0: number[], rows1: number[], alpha: number): AssumptionRow[] | { error: string } {
  if (!(alpha > 0 && alpha < 1)) return { error: "α は 0 と 1 の間で指定してください" };
  return names.map((feature, j) => {
    const a = finite(X, rows0, j), b = finite(X, rows1, j);
    const lv = a.length && b.length ? levene([a, b], "median") : { error: "片方の群が空です" };
    const levCheck: Check = "error" in lv ? { p: null, stat: null, n: a.length + b.length, reason: lv.error } : { p: lv.p, stat: lv.f, n: a.length + b.length, reason: lv.reason };
    const sel: Selection | { error: string } = selectUnpaired(a, b, alpha);
    const base = { index: j, feature, n: [a.length, b.length] as [number, number], shapiro: [shapiro(a), shapiro(b)] as [Check, Check], levene: levCheck };
    if ("error" in sel) return { ...base, test: null, p: null, why: null, reason: sel.error };
    return { ...base, test: sel.test, p: sel.p, why: sel.why, reason: sel.reason ?? null };
  });
}

/** 選ばれた検定ごとの件数（検定できなかった列は "none"）。 */
export function testCounts(rows: readonly { test: TestId | null }[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rows) out[r.test ?? "none"] = (out[r.test ?? "none"] ?? 0) + 1;
  return out;
}
