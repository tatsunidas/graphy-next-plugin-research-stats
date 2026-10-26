/**
 * レシピ R3「再現性」の計算本体。条件ごとの特徴量の表（1 ファイル = 1 条件）を行 ID で突き合わせ、
 * 特徴量ごとに ICC（absolute agreement）と区分を求める。結合・生存・結果ラベルは使わない。
 * 欠損は 0 にせず、その特徴量では欠損のある行を除く（除いた数を出す）。
 */
import { parseCell } from "./csv";
import type { TableSource } from "./dataset";
import type { ParamValue } from "./recipe";
import { blandAltman, type BlandAltman } from "./reliability/blandAltman";
import { icc, type IccCategory, type IccResult } from "./reliability/icc";

export interface R3Params {
  idColumn: string;
  /** これより下の区分の特徴量を除外候補にする。null なら候補を作らない（既定。基準は論文に無い）。 */
  minCategory: IccCategory | null;
}

export const R3_DEFAULTS: Record<string, ParamValue> = { reliabilityIdColumn: "CaseLabel", minCategory: null };

const CATS: IccCategory[] = ["poor", "moderate", "good", "excellent"];

export function readR3Params(p: Record<string, ParamValue>): R3Params | { error: string } {
  const id = p.reliabilityIdColumn;
  if (typeof id !== "string" || id === "") return { error: "行を識別する列を選んでください" };
  const mc = p.minCategory;
  if (mc !== null && !(typeof mc === "string" && (CATS as string[]).includes(mc))) return { error: "除外の基準は poor / moderate / good / excellent のどれか（または使わない）です" };
  return { idColumn: id, minCategory: mc as IccCategory | null };
}

export interface ConditionReport {
  conditions: string[];
  rowsPerCondition: number[];
  matched: number;
  /** どれかの条件にしか無い行 ID と、無い条件。 */
  unmatched: { id: string; missingIn: string[] }[];
  /** 使えなかった列と理由（一部の条件にしか無い、数でない値がある、数が 1 つも無い）。 */
  excludedColumns: { name: string; reason: string }[];
}

export interface ConditionData {
  conditions: string[];
  ids: string[];
  features: string[];
  /** [特徴量][症例][条件]。欠損は null。 */
  values: (number | null)[][][];
  report: ConditionReport;
}

export type ConditionResult = { ok: true; data: ConditionData } | { ok: false; errors: string[] };

const preview = (xs: string[], n = 5) => (xs.length > n ? `${xs.slice(0, n).join(", ")} …（ほか ${xs.length - n}）` : xs.join(", "));

export function conditionNames(sources: TableSource[]): string[] {
  const seen = new Map<string, number>();
  return sources.map((s) => {
    const base = s.name.replace(/\.csv$/i, "");
    const c = (seen.get(base) ?? 0) + 1;
    seen.set(base, c);
    return c === 1 ? base : `${base} (${c})`;
  });
}

export function buildConditions(sources: TableSource[], idColumn: string): ConditionResult {
  const errors: string[] = [];
  if (sources.length < 2) return { ok: false, errors: ["条件が 2 つ以上要ります（特徴量の CSV を 2 つ以上選んでください。1 ファイル = 1 条件）"] };
  const names = conditionNames(sources);
  const maps: Map<string, number>[] = sources.map((s) => {
    const j = s.table.header.indexOf(idColumn);
    const m = new Map<string, number>();
    if (j < 0) { errors.push(`${s.name}: 列「${idColumn}」がありません`); return m; }
    const dup: string[] = [];
    s.table.rows.forEach((r, i) => {
      const id = (r[j] ?? "").trim();
      if (id === "") { errors.push(`${s.name}: ${i + 2} 行目の ${idColumn} が空です`); return; }
      if (m.has(id)) dup.push(id); else m.set(id, i);
    });
    if (dup.length) errors.push(`${s.name}: ${idColumn} が重複しています: ${preview([...new Set(dup)])}`);
    return m;
  });
  if (errors.length) return { ok: false, errors };

  const allIds = [...new Set(maps.flatMap((m) => [...m.keys()]))];
  const ids = [...maps[0].keys()].filter((id) => maps.every((m) => m.has(id)));
  const unmatched = allIds.filter((id) => !maps.every((m) => m.has(id))).map((id) => ({ id, missingIn: names.filter((_, c) => !maps[c].has(id)) }));
  if (ids.length < 3) return { ok: false, errors: [`全条件に揃った行が ${ids.length} 個しかありません（3 個以上要ります）`] };

  const excluded: { name: string; reason: string }[] = [];
  const headerSets = sources.map((s) => new Set(s.table.header));
  const union = [...new Set(sources.flatMap((s) => s.table.header))].filter((h) => h !== idColumn);
  const features: string[] = [], values: (number | null)[][][] = [];
  for (const name of union) {
    if (!headerSets.every((hs) => hs.has(name))) { excluded.push({ name, reason: `一部の条件にしかありません（無い条件: ${names.filter((_, c) => !headerSets[c].has(name)).join(", ")}）` }); continue; }
    let text = false, any = false;
    const cols: (number | null)[][] = ids.map(() => []);
    sources.forEach((s, c) => {
      const j = s.table.header.indexOf(name);
      ids.forEach((id, r) => {
        const cell = parseCell(s.table.rows[maps[c].get(id)!][j]);
        if (cell.kind === "text") text = true;
        if (cell.kind === "number") any = true;
        cols[r].push(cell.kind === "number" ? cell.value : null);
      });
    });
    if (text) { excluded.push({ name, reason: "数でない値があります" }); continue; }
    if (!any) { excluded.push({ name, reason: "数が 1 つもありません" }); continue; }
    features.push(name);
    values.push(cols);
  }
  if (!features.length) return { ok: false, errors: ["全条件に共通する数値の列がありません"] };
  return { ok: true, data: { conditions: names, ids, features, values, report: { conditions: names, rowsPerCondition: sources.map((s) => s.table.rows.length), matched: ids.length, unmatched, excludedColumns: excluded } } };
}

export interface ReliabilityRow {
  feature: string;
  /** ICC に使った行（全条件が揃った行）。 */
  n: number;
  /** 欠損のため除いた行。 */
  dropped: number;
  result: IccResult | null;
  error: string | null;
}

export function analyzeReliability(d: ConditionData): ReliabilityRow[] {
  return d.features.map((feature, f) => {
    const complete = d.values[f].filter((row) => row.every((v) => v !== null)) as number[][];
    const row: ReliabilityRow = { feature, n: complete.length, dropped: d.ids.length - complete.length, result: null, error: null };
    const r = icc(complete);
    if ("error" in r) row.error = r.error; else row.result = r;
    return row;
  });
}

export const categoryRank = (c: IccCategory): number => CATS.indexOf(c);

/** 基準（min）より低い区分、または ICC を計算できなかった特徴量。基準が無ければ空。 */
export function exclusionCandidates(rows: ReliabilityRow[], min: IccCategory | null): ReliabilityRow[] {
  if (min === null) return [];
  return rows.filter((r) => r.result === null || categoryRank(r.result.category) < categoryRank(min));
}

/** 選んだ特徴量・2 条件の Bland–Altman（A → B の差）。 */
export function blandAltmanFor(d: ConditionData, feature: string, a: number, b: number): BlandAltman | { error: string } {
  const f = d.features.indexOf(feature);
  if (f < 0) return { error: `特徴量「${feature}」がありません` };
  if (a === b) return { error: "異なる 2 つの条件を選んでください" };
  const rows = d.values[f].filter((r) => r[a] !== null && r[b] !== null);
  return blandAltman(rows.map((r) => r[a] as number), rows.map((r) => r[b] as number));
}

