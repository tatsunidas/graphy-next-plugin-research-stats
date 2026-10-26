/**
 * 治療前後の変化率。2 つの時点の特徴量の表（同じ病変の行どうし）から、患者ごとの変化率を作る。
 *
 * 順序は 2 通り（文献で違う）:
 *  - "lesion-then-mean": 病変ごとに変化率を出し、患者で平均する（既定）。
 *  - "mean-then-change": 患者で平均してから変化率を出す。
 * 2 つは一般に違う値になる。基準値が 0 の病変・欠損は 0 にせず欠損にして、理由を残す。
 *
 * 変化率 = (後 − 前) / |前| × 100。基準値が負でも「増えたら正」になるよう |前| で割る
 * （この式は出典の記述から確かめていない仮定。変えるときはここだけ）。
 */
import { meanOrNull, readFeatureRows, type TableSource } from "./dataset";
import { parseCsv } from "./csv";

export type DeltaOrder = "lesion-then-mean" | "mean-then-change";

export interface DeltaSpec {
  baseline: TableSource[];
  followup: TableSource[];
  idColumn: string;
  patientColumn: string | null;
  featureColumns: string[] | null;
  order: DeltaOrder;
}

export interface DeltaProblem {
  id: string;
  feature: string;
  reason: "missing" | "zero-baseline";
}

export interface DeltaResult {
  patients: string[];
  featureNames: string[];
  /** [患者][特徴量] の変化率（%）。作れないものは null。 */
  values: (number | null)[][];
  problems: DeltaProblem[];
  /** 片方の時点にしか無い病変（変化率に使わない）。 */
  unmatched: { id: string; where: "baseline" | "followup" }[];
}

export type DeltaBuild = { ok: true; result: DeltaResult } | { ok: false; errors: string[] };

/** 1 つの変化率。基準値が 0 なら "zero-baseline"、どちらかが欠損なら "missing"。 */
export function percentChange(before: number | null, after: number | null): number | "missing" | "zero-baseline" {
  if (before === null || after === null) return "missing";
  if (before === 0) return "zero-baseline";
  return ((after - before) / Math.abs(before)) * 100;
}

export function buildDelta(spec: DeltaSpec): DeltaBuild {
  const errors: string[] = [];
  const b = readFeatureRows(spec.baseline, spec.idColumn, spec.patientColumn, spec.featureColumns, errors);
  // 後の時点は、前の時点で決まった列に合わせて読む
  const f = readFeatureRows(spec.followup, spec.idColumn, spec.patientColumn, b.columns.length ? b.columns : spec.featureColumns, errors);
  if (errors.length) return { ok: false, errors };

  const fById = new Map(f.rows.map((r) => [r.id, r]));
  const bIds = new Set(b.rows.map((r) => r.id));
  const unmatched: DeltaResult["unmatched"] = [
    ...b.rows.filter((r) => !fById.has(r.id)).map((r) => ({ id: r.id, where: "baseline" as const })),
    ...f.rows.filter((r) => !bIds.has(r.id)).map((r) => ({ id: r.id, where: "followup" as const })),
  ];
  const pairs = b.rows.filter((r) => fById.has(r.id)).map((r) => ({ before: r, after: fById.get(r.id)! }));
  const patients: string[] = [];
  const group = new Map<string, typeof pairs>();
  for (const p of pairs) {
    if (p.before.patient !== p.after.patient) return { ok: false, errors: [`病変「${p.before.id}」の患者が 2 時点で違います（${p.before.patient} / ${p.after.patient}）`] };
    if (!group.has(p.before.patient)) {
      group.set(p.before.patient, []);
      patients.push(p.before.patient);
    }
    group.get(p.before.patient)!.push(p);
  }
  const problems: DeltaProblem[] = [];
  const values = patients.map((pid) => {
    const list = group.get(pid)!;
    return b.columns.map((name, m) => {
      if (spec.order === "lesion-then-mean") {
        const changes: (number | null)[] = list.map((p) => {
          const c = percentChange(p.before.values[m], p.after.values[m]);
          if (typeof c === "number") return c;
          problems.push({ id: p.before.id, feature: name, reason: c });
          return null;
        });
        return meanOrNull(changes);
      }
      const c = percentChange(meanOrNull(list.map((p) => p.before.values[m])), meanOrNull(list.map((p) => p.after.values[m])));
      if (typeof c === "number") return c;
      problems.push({ id: pid, feature: name, reason: c });
      return null;
    });
  });
  return { ok: true, result: { patients, featureNames: b.columns, values, problems, unmatched } };
}

/** 変化率の結果を、結合（buildDataset）にそのまま渡せる特徴量の表（1 行 = 1 患者）にする。 */
export function deltaToTable(r: DeltaResult, idHeader: string): { header: string[]; rows: string[][] } {
  return {
    header: [idHeader, ...r.featureNames.map((n) => `Delta_${n}`)],
    rows: r.patients.map((p, i) => [p, ...r.values[i].map((v) => (v === null ? "" : String(v)))]),
  };
}

/** テスト・UI 用: CSV 文字列から表にして TableSource にする。 */
export const sourceOf = (name: string, csv: string, sha256: string | null = null): TableSource => ({ name, sha256, table: parseCsv(csv) });
