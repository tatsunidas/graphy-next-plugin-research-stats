/**
 * 特徴量の表と臨床の表を結ぶ ID の列を、実際の値の重なりから探す。
 * 値の比較にだけ正規化（NFKC・前後の空白・ゼロ幅文字・制御文字の除去・大文字小文字の無視）を使い、
 * 結合そのもの（dataset.ts）は完全一致（前後の空白を除く）のまま。正規化すると一致する ID は警告に出す。
 */
import { parseCell, type CsvTable } from "./csv";
import type { Suggestion } from "./suggest";

/** U+200B〜U+200D・U+FEFF（ゼロ幅）と制御文字を除く。 */
const INVISIBLE = /[​-‍﻿\u0000-\u001F\u007F-\u009F]/g;

export function normalizeId(v: string): string {
  return v.normalize("NFKC").replace(INVISIBLE, "").trim().toLowerCase();
}

/** 結合と同じ比較（前後の空白だけ除く）。 */
const exactId = (v: string) => v.trim();

export interface PairOverlap {
  featureColumn: string;
  clinicalColumn: string;
  /** 正規化したあとの値の集合の重なり（異なる値の数）。 */
  normalized: number;
  /** 完全一致（前後の空白を除く）での重なり。結合できる人数の上限。 */
  exact: number;
  featureDistinct: number;
  clinicalDistinct: number;
}

interface ColumnValues {
  name: string;
  exact: Set<string>;
  normalized: Set<string>;
  /** 空でない行数（重複を含む）。 */
  nonEmpty: number;
}

function columnValues(tables: readonly CsvTable[], name: string): ColumnValues {
  const exact = new Set<string>(), normalized = new Set<string>();
  let nonEmpty = 0;
  for (const t of tables) {
    const j = t.header.indexOf(name);
    if (j < 0) continue;
    for (const r of t.rows) {
      const e = exactId(r[j] ?? "");
      if (e === "") continue;
      nonEmpty++;
      exact.add(e);
      const n = normalizeId(e);
      if (n !== "") normalized.add(n);
    }
  }
  return { name, exact, normalized, nonEmpty };
}

/** 小数を含む数値の列は ID ではない（特徴量の列が、たまたま臨床の数値と重なるのを候補から外す）。 */
function hasFraction(tables: readonly CsvTable[], name: string): boolean {
  for (const t of tables) {
    const j = t.header.indexOf(name);
    if (j < 0) continue;
    for (const r of t.rows) {
      const c = parseCell(r[j] ?? "");
      if (c.kind === "number" && !Number.isInteger(c.value)) return true;
    }
  }
  return false;
}

const intersect = (a: ReadonlySet<string>, b: ReadonlySet<string>) => {
  const [s, l] = a.size <= b.size ? [a, b] : [b, a];
  let n = 0;
  for (const x of s) if (l.has(x)) n++;
  return n;
};

/** 見出しの重複を除く（同じ名前の列を 2 度数えない）。 */
const uniqueNames = (tables: readonly CsvTable[]) => [...new Set(tables.flatMap((t) => t.header))];

/** 特徴量の表の各列 × 臨床の表の各列の重なり。重なりが大きい順（同点は見出しの順）。 */
export function overlapPairs(features: readonly CsvTable[], clinical: CsvTable): PairOverlap[] {
  const candidates = (tables: readonly CsvTable[]) =>
    uniqueNames(tables).filter((n) => !hasFraction(tables, n)).map((n) => columnValues(tables, n)).filter((c) => c.normalized.size > 0);
  const fcols = candidates(features);
  const ccols = candidates([clinical]);
  const out: PairOverlap[] = [];
  for (const f of fcols) {
    for (const c of ccols) {
      out.push({
        featureColumn: f.name, clinicalColumn: c.name,
        normalized: intersect(f.normalized, c.normalized), exact: intersect(f.exact, c.exact),
        featureDistinct: f.exact.size, clinicalDistinct: c.exact.size,
      });
    }
  }
  return out;
}

export interface JoinSuggestion {
  featureId: string;
  patient: string | null;
  clinicalId: string;
  best: PairOverlap | null;
  /** 重なりが大きい順（0 の組も含む）。 */
  pairs: PairOverlap[];
}

/**
 * 重なりが最大の組を提案する。同点のときは、列名による推測（suggest.ts）の組を優先する。
 * 臨床の ID は患者単位なので、特徴量の側の列が行ごとに 1 つ（重複なし）ならそれを行 ID、
 * そうでなければ名前の推測の行 ID を残して、その列を「患者」にする。
 */
export function suggestJoin(
  features: readonly CsvTable[], clinical: CsvTable, hint: Pick<Suggestion, "featureId" | "patient" | "clinicalId">,
): JoinSuggestion {
  const hintKey = hint.patient ?? hint.featureId;
  const pairs = overlapPairs(features, clinical);
  const fidx = uniqueNames(features), cidx = clinical.header;
  const order = (p: PairOverlap) => fidx.indexOf(p.featureColumn) * (cidx.length + 1) + cidx.indexOf(p.clinicalColumn);
  const hinted = (p: PairOverlap) => (p.featureColumn === hintKey ? 1 : 0) + (p.clinicalColumn === hint.clinicalId ? 1 : 0);
  pairs.sort((a, b) => b.normalized - a.normalized || hinted(b) - hinted(a) || order(a) - order(b));
  const best = pairs[0] && pairs[0].normalized > 0 ? pairs[0] : null;
  if (!best) return { featureId: hint.featureId, patient: hint.patient, clinicalId: hint.clinicalId, best: null, pairs };

  return { ...assignPair(features, best, hint.featureId), clinicalId: best.clinicalColumn, best, pairs };
}

/** 選んだ組から、行 ID と患者の列を決める（列が行ごとに 1 つならそれが行 ID）。 */
export function assignPair(features: readonly CsvTable[], pair: PairOverlap, hintFeatureId: string): { featureId: string; patient: string | null } {
  const rowCount = features.reduce((n, t) => n + t.rows.length, 0);
  const unique = (col: string) => columnValues(features, col).exact.size === rowCount;
  if (unique(pair.featureColumn)) return { featureId: pair.featureColumn, patient: null };
  let featureId = hintFeatureId;
  if (featureId === pair.featureColumn || !unique(featureId)) {
    const alt = uniqueNames(features).find((c) => c !== pair.featureColumn && unique(c));
    if (alt) featureId = alt;
  }
  return { featureId, patient: pair.featureColumn };
}

/** 完全一致では結合できないが、正規化すると一致する ID の組（片側にだけいる ID どうし）。 */
export function idVariants(featureOnly: readonly string[], clinicalOnly: readonly string[]): { feature: string; clinical: string }[] {
  const byNorm = new Map<string, string>();
  for (const c of clinicalOnly) {
    const n = normalizeId(c);
    if (n !== "" && !byNorm.has(n)) byNorm.set(n, c);
  }
  const out: { feature: string; clinical: string }[] = [];
  for (const f of featureOnly) {
    const c = byNorm.get(normalizeId(f));
    if (c !== undefined) out.push({ feature: f, clinical: c });
  }
  return out;
}

/** 両方の表にある同名の列（使っている ID の列を除く）。 */
export function sharedColumns(features: readonly CsvTable[], clinical: CsvTable, idColumns: readonly (string | null)[]): string[] {
  const f = new Set(features[0]?.header ?? []);
  return clinical.header.filter((h) => f.has(h) && !idColumns.includes(h));
}

export type JoinNote =
  | { kind: "noOverlap"; pairs: PairOverlap[] }
  | { kind: "betterPair"; used: PairOverlap | null; better: PairOverlap }
  | { kind: "variants"; items: { feature: string; clinical: string }[] }
  | { kind: "sharedColumns"; columns: string[] };

export interface JoinNoteInput {
  features: readonly CsvTable[];
  clinical: CsvTable;
  /** 結合に使っている列（患者の列があればそれ、無ければ行 ID）。 */
  featureKey: string;
  clinicalKey: string;
  /** 行 ID の列（同名列の判定から除く）。 */
  featureId: string;
  pairs: readonly PairOverlap[];
  featureOnly: readonly string[];
  clinicalOnly: readonly string[];
}

export function joinNotes(i: JoinNoteInput): JoinNote[] {
  const notes: JoinNote[] = [];
  const used = i.pairs.find((p) => p.featureColumn === i.featureKey && p.clinicalColumn === i.clinicalKey) ?? null;
  const best = i.pairs[0];
  if (i.pairs.every((p) => p.normalized === 0)) notes.push({ kind: "noOverlap", pairs: i.pairs.slice(0, 3) });
  else if (best && best.normalized > (used?.normalized ?? 0)) notes.push({ kind: "betterPair", used, better: best });
  const items = idVariants(i.featureOnly, i.clinicalOnly);
  if (items.length) notes.push({ kind: "variants", items });
  const shared = sharedColumns(i.features, i.clinical, [i.featureKey, i.clinicalKey, i.featureId]);
  if (shared.length) notes.push({ kind: "sharedColumns", columns: shared });
  return notes;
}
