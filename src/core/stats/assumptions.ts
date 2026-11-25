/**
 * 検定の選び方。前提（正規性・等分散性）を検定して、使う検定を決める。
 * α は呼び出し側が決める（この部品は値を持たない）。
 *
 * 対応なし 2 群: 各群の Shapiro–Wilk の p が両方 α 以上（各群 3 個以上）なら「正規」。
 *   正規で Levene（median）の p が α 以上 → Student、正規で等分散でない → Welch、
 *   正規でない（検定できない場合を含む）→ Mann–Whitney。
 *   Levene が計算できなかったときは、等分散を仮定しない Welch にする。
 * 対応あり: 差の Shapiro–Wilk の p が α 以上（3 組以上）→ 対応のある t、そうでなければ Wilcoxon の符号順位。
 */
import { mannWhitney } from "../survival/mannwhitney";
import { levene } from "./levene";
import { shapiroWilk } from "./shapiro";
import { pairedT, studentT, welchT } from "./ttest";
import { wilcoxonSignedRank, type WilcoxonMethod } from "./wilcoxon";

export type TestId = "student" | "welch" | "mannwhitney" | "paired-t" | "wilcoxon";

/** 前提の検定の 1 件。p が null なら検定できなかった（理由つき）。 */
export interface Check {
  p: number | null;
  stat: number | null;
  n: number;
  reason?: string;
}

export interface Selection {
  test: TestId;
  alpha: number;
  /** 本検定の検定統計量（t、Mann–Whitney は z、Wilcoxon は min(W+, W−)）。 */
  statistic: number | null;
  df: number | null;
  /** 計算できなければ null。 */
  p: number | null;
  /** 本検定が計算できなかった理由。 */
  reason?: string;
  /** Wilcoxon のみ。 */
  wilcoxon?: { method: WilcoxonMethod | null; zerosDropped: number };
  /** 対応なしは [群 1, 群 2]、対応ありは [差]。 */
  normality: Check[];
  levene: Check | null;
  why: { ja: string; en: string };
}

function normality(v: number[]): Check {
  const r = shapiroWilk(v);
  if ("error" in r) return { p: null, stat: null, n: v.length, reason: r.error };
  return { p: r.p, stat: r.w, n: v.length, reason: r.reason };
}

const fmt = (p: number | null) => (p === null ? "NA" : p < 1e-4 ? p.toExponential(2) : p.toFixed(4));
const isNormal = (c: Check, alpha: number) => c.p !== null && c.p >= alpha;

export function selectUnpaired(x: number[], y: number[], alpha: number): Selection | { error: string } {
  if (!(alpha > 0 && alpha < 1)) return { error: "α は 0 と 1 の間で指定してください" };
  if (x.length < 1 || y.length < 1) return { error: "片方の群が空です" };
  const nx = normality(x), ny = normality(y);
  const normal = isNormal(nx, alpha) && isNormal(ny, alpha);
  let lv: Check | null = null;
  if (normal) {
    const l = levene([x, y], "median");
    lv = "error" in l ? { p: null, stat: null, n: x.length + y.length, reason: l.error } : { p: l.p, stat: l.f, n: x.length + y.length, reason: l.reason };
  }
  const normTxtJa = `正規性（Shapiro–Wilk）p = ${fmt(nx.p)}・${fmt(ny.p)}`;
  const normTxtEn = `normality (Shapiro–Wilk) p = ${fmt(nx.p)}, ${fmt(ny.p)}`;

  let test: TestId;
  let why: { ja: string; en: string };
  if (!normal) {
    test = "mannwhitney";
    why = {
      ja: `${normTxtJa}。両群が正規とは言えない（α = ${alpha}、検定できない場合を含む）ので Mann–Whitney。`,
      en: `${normTxtEn}. Normality is not supported in both groups (alpha = ${alpha}, including untestable cases), so Mann–Whitney.`,
    };
  } else if (lv !== null && lv.p !== null && lv.p >= alpha) {
    test = "student";
    why = {
      ja: `${normTxtJa}、等分散性（Levene、median）p = ${fmt(lv.p)} ≥ ${alpha} なので Student の t 検定。`,
      en: `${normTxtEn}; homogeneity of variance (Levene, median) p = ${fmt(lv.p)} >= ${alpha}, so Student's t-test.`,
    };
  } else {
    test = "welch";
    why = {
      ja: `${normTxtJa}、等分散性（Levene、median）p = ${fmt(lv?.p ?? null)}（${lv?.p === null ? "検定できない" : `< ${alpha}`}）なので Welch の t 検定。`,
      en: `${normTxtEn}; homogeneity of variance (Levene, median) p = ${fmt(lv?.p ?? null)} (${lv?.p === null ? "untestable" : `< ${alpha}`}), so Welch's t-test.`,
    };
  }

  let statistic: number | null = null, df: number | null = null, p: number | null = null;
  let reason: string | undefined;
  if (test === "mannwhitney") {
    const r = mannWhitney(x, y);
    if ("error" in r) reason = r.error; else { statistic = r.z; p = r.p; }
  } else {
    const r = test === "student" ? studentT(x, y) : welchT(x, y);
    if ("error" in r) reason = r.error; else { statistic = r.t; df = r.df; p = r.p; reason = r.reason; }
  }
  return { test, alpha, statistic, df, p, reason, normality: [nx, ny], levene: lv, why };
}

export function selectPaired(x: number[], y: number[], alpha: number): Selection | { error: string } {
  if (!(alpha > 0 && alpha < 1)) return { error: "α は 0 と 1 の間で指定してください" };
  if (x.length !== y.length) return { error: "対の個数が違います" };
  if (x.length < 1) return { error: "対がありません" };
  const d = x.map((v, i) => v - y[i]);
  const nd = normality(d);
  const normal = isNormal(nd, alpha);
  if (normal) {
    const r = pairedT(x, y);
    const base = { test: "paired-t" as const, alpha, normality: [nd], levene: null };
    const why = {
      ja: `差の正規性（Shapiro–Wilk）p = ${fmt(nd.p)} ≥ ${alpha} なので対応のある t 検定。`,
      en: `Normality of the differences (Shapiro–Wilk) p = ${fmt(nd.p)} >= ${alpha}, so paired t-test.`,
    };
    if ("error" in r) return { ...base, statistic: null, df: null, p: null, reason: r.error, why };
    return { ...base, statistic: r.t, df: r.df, p: r.p, reason: r.reason, why };
  }
  const w = wilcoxonSignedRank(x, y);
  const why = {
    ja: `差の正規性（Shapiro–Wilk）p = ${fmt(nd.p)}。正規とは言えない（α = ${alpha}、検定できない場合を含む）ので Wilcoxon の符号順位検定。`,
    en: `Normality of the differences (Shapiro–Wilk) p = ${fmt(nd.p)}. Normality is not supported (alpha = ${alpha}, including untestable cases), so Wilcoxon signed-rank.`,
  };
  const base = { test: "wilcoxon" as const, alpha, normality: [nd], levene: null, why };
  if ("error" in w) return { ...base, statistic: null, df: null, p: null, reason: w.error };
  return { ...base, statistic: w.statistic, df: null, p: w.p, reason: w.reason, wilcoxon: { method: w.method, zerosDropped: w.zerosDropped } };
}
