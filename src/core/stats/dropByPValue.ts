/**
 * 列ごとに検定して、p が「しきい値以上」または計算できない列を落とす（結果ラベルを使う教師ありの選択）。
 * 推定には rows / pairs に挙げた行だけを読む。開発集合の行だけを渡すこと（テスト集合を読まない）。
 * しきい値・α は呼び出し側が決める。
 */
import { selectPaired, selectUnpaired, type Selection } from "./assumptions";

export type Design =
  /** 2 群。群 1 の行番号と群 2 の行番号。 */
  | { kind: "unpaired"; rows0: number[]; rows1: number[] }
  /** 対応あり。[1 つめの行, 2 つめの行] の組。 */
  | { kind: "paired"; pairs: [number, number][] };

export interface ColumnRecord {
  index: number;
  feature: string;
  /** 使った値の個数（対応なしは群ごと、対応ありは組の数）。 */
  n: number[];
  /** 選択の記録。検定できなかった（個数不足など）ときは null。 */
  selection: Selection | null;
  p: number | null;
  kept: boolean;
  /** 落とした理由。残したときは null。 */
  dropReason: { ja: string; en: string } | null;
}

export interface DropByPResult {
  keep: number[];
  records: ColumnRecord[];
}

export function dropByPValue(
  X: (number | null)[][],
  names: string[],
  design: Design,
  alpha: number,
  threshold: number,
): DropByPResult | { error: string } {
  if (!(threshold > 0 && threshold <= 1)) return { error: "しきい値は 0 より大きく 1 以下で指定してください" };
  if (!(alpha > 0 && alpha < 1)) return { error: "α は 0 と 1 の間で指定してください" };
  const records: ColumnRecord[] = [];
  const keep: number[] = [];
  for (let j = 0; j < names.length; j++) {
    let sel: Selection | { error: string };
    let n: number[];
    if (design.kind === "unpaired") {
      const g = (rows: number[]) => rows.map((i) => X[i][j]).filter((v): v is number => v !== null && Number.isFinite(v));
      const a = g(design.rows0), b = g(design.rows1);
      n = [a.length, b.length];
      sel = selectUnpaired(a, b, alpha);
    } else {
      const a: number[] = [], b: number[] = [];
      for (const [i, k] of design.pairs) {
        const u = X[i][j], v = X[k][j];
        if (u !== null && v !== null && Number.isFinite(u) && Number.isFinite(v)) { a.push(u); b.push(v); }
      }
      n = [a.length];
      sel = selectPaired(a, b, alpha);
    }
    const base = { index: j, feature: names[j], n };
    if ("error" in sel) {
      records.push({ ...base, selection: null, p: null, kept: false, dropReason: { ja: `検定できない（${sel.error}）`, en: `could not be tested (${sel.error})` } });
      continue;
    }
    const p = sel.p;
    if (p === null) {
      records.push({ ...base, selection: sel, p, kept: false, dropReason: { ja: `p が計算できない（${sel.reason ?? "理由不明"}）`, en: `p-value not computable (${sel.reason ?? "unknown"})` } });
    } else if (p >= threshold) {
      records.push({ ...base, selection: sel, p, kept: false, dropReason: { ja: `p = ${p.toPrecision(3)} ≥ しきい値 ${threshold}`, en: `p = ${p.toPrecision(3)} >= threshold ${threshold}` } });
    } else {
      records.push({ ...base, selection: sel, p, kept: true, dropReason: null });
      keep.push(j);
    }
  }
  return { keep, records };
}
