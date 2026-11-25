/** 結合済みのデータセットを CSV（1 行 = 1 患者。行のままのときは 1 行 = 1 観測で、患者の列を足す）にする。欠損は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { Dataset } from "./dataset";

export function datasetCsv(d: Dataset, idHeader: string): string {
  const g = d.groups;
  const header = [idHeader, ...(g ? ["Patient"] : []), ...d.clinical.map((c) => c.name), ...d.featureNames];
  const rows = d.ids.map((id, i) => [id, ...(g ? [g[i]] : []), ...d.clinical.map((c) => c.values[i]), ...d.features[i]]);
  return toCsv(header, rows);
}
