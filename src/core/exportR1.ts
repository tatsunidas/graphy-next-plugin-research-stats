/** R1 の結果を CSV にする。計算できなかった値は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { FeatureContext, FeatureResult } from "./r1";
import { kmForFeature } from "./r1";

const COLUMNS: [string, (r: FeatureResult) => number | string | null][] = [
  ["feature", (r) => r.feature], ["n", (r) => r.n], ["events", (r) => r.events],
  ["cut1", (r) => r.cuts[0] ?? null], ["cut2", (r) => r.cuts[1] ?? null],
  ["group_sizes", (r) => r.groupSizes.join("/")], ["group_events", (r) => r.groupEvents.join("/")],
  ["group_median_survival", (r) => r.medians.map((m) => (m === null ? "NR" : String(m))).join("/")],
  ["logrank_chi2", (r) => r.logrankChi2], ["logrank_p", (r) => r.logrankP],
  ["optimal_raw_p", (r) => r.rawOptimalP], ["optimal_permutation_p", (r) => r.correctedP],
  ["p_used", (r) => r.pUsed], ["q_bh", (r) => r.q],
  ["hr_group", (r) => r.hrGroup], ["hr_group_lower", (r) => r.hrGroupLower], ["hr_group_upper", (r) => r.hrGroupUpper], ["hr_group_p", (r) => r.hrGroupP],
  ["hr_per_sd", (r) => r.hrPerSd], ["hr_per_sd_lower", (r) => r.hrPerSdLower], ["hr_per_sd_upper", (r) => r.hrPerSdUpper], ["hr_per_sd_p", (r) => r.hrPerSdP],
  ["mannwhitney_p", (r) => r.mannWhitneyP], ["rmst_diff", (r) => r.rmstDiff], ["rmst_p", (r) => r.rmstP], ["error", (r) => r.error],
];

export function r1ResultsCsv(rows: FeatureResult[]): string {
  return toCsv(COLUMNS.map(([h]) => h), rows.map((r) => COLUMNS.map(([, f]) => f(r))));
}

/** 全特徴量の KM の点列（図を作り直せる元の数値）。 */
export function kmPointsCsv(ctx: FeatureContext, rows: FeatureResult[]): string {
  const out: (number | string | null)[][] = [];
  for (const r of rows) {
    const gs = kmForFeature(ctx, r);
    if (!gs) continue;
    gs.forEach((g, k) => g.km.points.forEach((p) => out.push([r.feature, k, g.label, p.time, p.atRisk, p.events, p.censored, p.surv, p.se, p.lower, p.upper])));
  }
  return toCsv(["feature", "group", "label", "time", "at_risk", "events", "censored", "survival", "se", "lower", "upper"], out);
}
