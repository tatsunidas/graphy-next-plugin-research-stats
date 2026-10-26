/**
 * 連続値を群に分ける。しきい値で分けるものは「値 > しきい値」を高群とする（ちょうど等しい値は低群）。
 *  - median: 中央値（既定。論文で一般的）。
 *  - value: 指定した値。
 *  - tertile: 三分位（type 7 の分位点 1/3・2/3 で 3 群）。
 *  - optimal: log-rank のχ²が最大（p が最小）になるしきい値。探索的で、p は多重比較で小さく出る。
 *    補正 p は permutation.ts。各群が全体の minFraction 以上になる候補だけを見る
 *    （既定 0.1 は本実装の仮定で、論文には記載が無い）。
 */
export type GroupMode =
  | { kind: "median" }
  | { kind: "value"; cut: number }
  | { kind: "tertile" }
  | { kind: "optimal"; cut: number };

export interface Grouping {
  /** 0 が最も低い群。 */
  group: number[];
  k: number;
  cuts: number[];
}

export function quantile7(sorted: number[], q: number): number {
  const h = (sorted.length - 1) * q, lo = Math.floor(h), hi = Math.ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

export function groupByCuts(values: number[], cuts: number[]): Grouping | { error: string } {
  const group = values.map((v) => cuts.filter((c) => v > c).length);
  const k = cuts.length + 1;
  const counts = new Array<number>(k).fill(0);
  for (const g of group) counts[g]++;
  const empty = counts.findIndex((c) => c === 0);
  if (empty >= 0) {
    return { error: `群 ${empty + 1} が空です（同じ値が多くて分けられません。しきい値 ${cuts.join(", ")}）` };
  }
  return { group, k, cuts };
}

export function groupBy(values: number[], mode: GroupMode): Grouping | { error: string } {
  if (values.some((v) => !Number.isFinite(v))) return { error: "欠損・非数の値があります" };
  const sorted = [...values].sort((a, b) => a - b);
  switch (mode.kind) {
    case "median": return groupByCuts(values, [quantile7(sorted, 0.5)]);
    case "value": return groupByCuts(values, [mode.cut]);
    case "optimal": return groupByCuts(values, [mode.cut]);
    case "tertile": return groupByCuts(values, [quantile7(sorted, 1 / 3), quantile7(sorted, 2 / 3)]);
  }
}
