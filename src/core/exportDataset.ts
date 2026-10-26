/** 結合済みのデータセットを CSV（1 行 = 1 患者）にする。欠損は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { Dataset } from "./dataset";

export function datasetCsv(d: Dataset, idHeader: string): string {
  const header = [idHeader, ...d.clinical.map((c) => c.name), ...d.featureNames];
  const rows = d.ids.map((id, i) => [id, ...d.clinical.map((c) => c.values[i]), ...d.features[i]]);
  return toCsv(header, rows);
}
