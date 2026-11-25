/** 前処理の報告を CSV にする。計算できなかった値は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { AssumptionRow } from "./stats/assumptionReport";
import type { PDropEntry } from "./pipeline/nestedCv";

/** 特徴量ごとの前提の検定（正規性・等分散性）と、選ばれる検定。 */
export function assumptionsCsv(rows: readonly AssumptionRow[], alpha: number): string {
  return toCsv(
    ["feature", "alpha", "n_group0", "n_group1", "shapiro_w_group0", "shapiro_p_group0", "shapiro_reason_group0", "shapiro_w_group1", "shapiro_p_group1", "shapiro_reason_group1",
      "levene_f", "levene_p", "levene_reason", "selected_test", "test_p", "why_ja", "why_en", "reason"],
    rows.map((r) => [r.feature, alpha, r.n[0], r.n[1], r.shapiro[0].stat, r.shapiro[0].p, r.shapiro[0].reason ?? null, r.shapiro[1].stat, r.shapiro[1].p, r.shapiro[1].reason ?? null,
      r.levene.stat, r.levene.p, r.levene.reason ?? null, r.test, r.p, r.why?.ja ?? null, r.why?.en ?? null, r.reason]),
  );
}

/** 訓練の行ごとの、p 値による除外（対応なし）で除いた特徴量。除いた特徴量が無い訓練の行は 1 行（feature 空欄）で残す。 */
export function pdropCsv(log: readonly PDropEntry[]): string {
  const rows: (string | number | null)[][] = [];
  for (const e of log) {
    if (!e.dropped.length) rows.push([e.scope, e.nTrain, e.tested, e.kept, null, null, null, null]);
    for (const d of e.dropped) rows.push([e.scope, e.nTrain, e.tested, e.kept, d.feature, d.test, d.p, d.reason.ja]);
  }
  return toCsv(["scope", "n_train", "tested", "kept", "dropped_feature", "test", "p", "reason"], rows);
}
