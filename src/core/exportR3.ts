/** R3 の結果を CSV にする。計算できなかった値は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { BlandAltman } from "./reliability/blandAltman";
import type { ReliabilityRow } from "./r3";

export function r3ResultsCsv(rows: ReliabilityRow[]): string {
  const header = ["feature", "n", "rows_dropped_missing", "icc", "icc_lower", "icc_upper", "icc_truncated", "icc_lower_truncated", "icc_upper_truncated", "category",
    "icc_avg", "icc_avg_lower", "icc_avg_upper", "msr", "msc", "mse", "note", "error"];
  return toCsv(header, rows.map((r) => [r.feature, r.n, r.dropped, r.result?.single.icc ?? null, r.result?.single.lower ?? null, r.result?.single.upper ?? null,
    r.result?.singleTruncated.icc ?? null, r.result?.singleTruncated.lower ?? null, r.result?.singleTruncated.upper ?? null, r.result?.category ?? null,
    r.result?.average.icc ?? null, r.result?.average.lower ?? null, r.result?.average.upper ?? null, r.result?.msr ?? null, r.result?.msc ?? null, r.result?.mse ?? null,
    r.result?.note ?? null, r.error]));
}

/** 除外候補（基準を選んだときだけ中身がある）。次の段（パイプライン）には、この一覧を渡す。 */
export function exclusionCsv(rows: ReliabilityRow[], minCategory: string | null): string {
  return toCsv(["feature", "category", "icc_lower_truncated", "reason"], rows.map((r) => [r.feature, r.result?.category ?? null, r.result?.singleTruncated.lower ?? null,
    r.result ? `区分が ${minCategory} より低い` : `ICC を計算できない: ${r.error}`]));
}

export function blandAltmanCsv(ba: BlandAltman, feature: string, a: string, b: string): string {
  const head = toCsv(["feature", "condition_a", "condition_b", "n", "bias", "bias_lower", "bias_upper", "sd", "loa_lower", "loa_lower_lower", "loa_lower_upper", "loa_upper", "loa_upper_lower", "loa_upper_upper"],
    [[feature, a, b, ba.n, ba.bias, ba.biasCi[0], ba.biasCi[1], ba.sd, ba.lowerLoa, ba.lowerLoaCi[0], ba.lowerLoaCi[1], ba.upperLoa, ba.upperLoaCi[0], ba.upperLoaCi[1]]]);
  return `${head}${toCsv(["mean", "difference"], ba.points)}`;
}
