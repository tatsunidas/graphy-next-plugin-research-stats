/**
 * 特徴量の表（1 行 = 1 ROI・病変）と臨床の表（1 行 = 1 患者）を結合して、患者単位のデータセットにする。
 *
 * 守ること:
 *  - 欠損は欠損のまま持つ。平均は欠損を除いて取り、すべて欠損なら欠損（0 にしない）。
 *  - 片側にしか無い患者は一覧にして出し、黙って捨てない。重複した ID は止める。
 *  - 役割の列（時間・イベント）が決まりの外の値なら、直さずに止める。
 */
import { numberOrNull, parseCell, type CsvTable } from "./csv";

export interface TableSource {
  name: string;
  /** 入力ファイルの SHA-256（取れなければ null＝「未取得」と記録に残す）。 */
  sha256: string | null;
  table: CsvTable;
}

export interface Roles {
  time: string | null;
  event: string | null;
  outcome: string | null;
  covariates: string[];
}

export interface DatasetSpec {
  /** 同じ見出しの表。行を連ねる（症例ごとに 1 ファイルの書き出しを並べる使い方）。 */
  features: TableSource[];
  /** 行（ROI・病変）を識別する列。 */
  featureIdColumn: string;
  /** 患者を示す列。null なら featureIdColumn が患者（1 行 = 1 患者）。 */
  patientColumn: string | null;
  /** 使う特徴量の列。null なら {@link defaultFeatureColumns}。 */
  featureColumns: string[] | null;
  clinical: TableSource;
  clinicalIdColumn: string;
  roles: Roles;
}

export type ClinicalValue = number | string | null;

export interface Dataset {
  ids: string[];
  featureNames: string[];
  /** [患者][特徴量]。欠損は null。 */
  features: (number | null)[][];
  /** 患者ごとの病変（行）数。 */
  lesionCounts: number[];
  clinical: { name: string; role: "time" | "event" | "outcome" | "covariate"; values: ClinicalValue[] }[];
}

export interface Exclusion {
  id: string;
  reason: string;
}

export interface JoinReport {
  featureFiles: number;
  featureRows: number;
  patientsInFeatures: number;
  patientsInClinical: number;
  joined: number;
  featureOnly: string[];
  clinicalOnly: string[];
  lesionsPerPatient: { min: number; max: number } | null;
  /** 欠損が 1 つ以上ある列だけ。 */
  missing: { name: string; count: number; of: number }[];
  /** 時間とイベントの両方が揃っている患者数（役割が決まっていなければ null）。 */
  survivalUsable: number | null;
  exclusions: Exclusion[];
  warnings: string[];
}

export type BuildResult =
  | { ok: true; dataset: Dataset; report: JoinReport }
  | { ok: false; errors: string[] };

/** 数値列の既定の候補から外す列（記録・ROI の大きさ・警告・フィルタの記録）。 */
const NON_FEATURE = /^(Provenance_|Roi_|Warning_|Filter_|Diagnostics_)|^Scale[0-9.]+mm_(Provenance_|Filter_)/;

/** 既定の特徴量の列: 値が（欠損を除いて）すべて数で、1 つ以上数がある列。識別の列と記録の列は除く。 */
export function defaultFeatureColumns(table: CsvTable, exclude: readonly (string | null)[]): string[] {
  return table.header.filter((h, j) => {
    if (exclude.includes(h) || NON_FEATURE.test(h)) return false;
    let any = false;
    for (const r of table.rows) {
      const c = parseCell(r[j]);
      if (c.kind === "text") return false;
      if (c.kind === "number") any = true;
    }
    return any;
  });
}

const unique = <T,>(xs: readonly T[]) => [...new Set(xs)];
const preview = (xs: readonly string[], n = 5) => (xs.length > n ? `${xs.slice(0, n).join(", ")} …（ほか ${xs.length - n}）` : xs.join(", "));

/** 欠損を除いた平均。1 つも無ければ null（0 にしない）。 */
export function meanOrNull(xs: readonly (number | null)[]): number | null {
  let s = 0, n = 0;
  for (const x of xs) {
    if (x !== null) {
      s += x;
      n++;
    }
  }
  return n === 0 ? null : s / n;
}

export interface FeatureRow {
  id: string;
  patient: string;
  values: (number | null)[];
}

/** 特徴量の表（複数）を読んで、行ごとの値にする。問題があれば errors に積む。 */
export function readFeatureRows(
  sources: readonly TableSource[], idColumn: string, patientColumn: string | null, columns: string[] | null, errors: string[],
): { rows: FeatureRow[]; columns: string[] } {
  if (sources.length === 0) {
    errors.push("特徴量の表がありません");
    return { rows: [], columns: [] };
  }
  const head = sources[0].table.header;
  for (const s of sources.slice(1)) {
    const missing = head.filter((h) => !s.table.header.includes(h));
    const extra = s.table.header.filter((h) => !head.includes(h));
    if (missing.length || extra.length) {
      errors.push(`${s.name}: 見出しが ${sources[0].name} と違います（足りない列: ${preview(missing) || "なし"} ／ 余分な列: ${preview(extra) || "なし"}）`);
    }
  }
  for (const col of [idColumn, ...(patientColumn ? [patientColumn] : [])]) {
    if (!head.includes(col)) errors.push(`特徴量の表に列「${col}」がありません`);
  }
  if (errors.length) return { rows: [], columns: [] };
  const cols = columns ?? defaultFeatureColumns(mergeTables(sources), [idColumn, patientColumn]);
  const bad = cols.filter((c) => !head.includes(c));
  if (bad.length) errors.push(`特徴量の列がありません: ${preview(bad)}`);
  if (cols.length === 0) errors.push("使える特徴量の列がありません（数値の列が 1 つもありません）");
  if (errors.length) return { rows: [], columns: [] };
  const rows: FeatureRow[] = [];
  const seen = new Set<string>();
  const nonNumeric: string[] = [];
  for (const s of sources) {
    const t = s.table;
    const iId = t.header.indexOf(idColumn);
    const iPat = patientColumn ? t.header.indexOf(patientColumn) : iId;
    const iCols = cols.map((c) => t.header.indexOf(c));
    t.rows.forEach((r, k) => {
      const id = r[iId].trim();
      const patient = r[iPat].trim();
      if (id === "" || patient === "") {
        errors.push(`${s.name} の ${k + 2} 行目: 「${id === "" ? idColumn : patientColumn}」が空です`);
        return;
      }
      if (seen.has(id)) errors.push(`行の ID「${id}」が重複しています（${s.name}）`);
      seen.add(id);
      const values = iCols.map((j, m) => {
        const v = numberOrNull(r[j]);
        if (v === undefined) {
          if (nonNumeric.length < 5) nonNumeric.push(`${s.name} の ${k + 2} 行目「${cols[m]}」= ${r[j]}`);
          return null;
        }
        return v;
      });
      rows.push({ id, patient, values });
    });
  }
  if (nonNumeric.length) errors.push(`特徴量の列に数でない値があります: ${nonNumeric.join(" ／ ")}`);
  return { rows, columns: cols };
}

/** 見出しの合わせ用に、全行を 1 つの表にする（既定の列の判定だけに使う）。 */
function mergeTables(sources: readonly TableSource[]): CsvTable {
  return { header: sources[0].table.header, rows: sources.flatMap((s) => s.table.rows) };
}

export function buildDataset(spec: DatasetSpec): BuildResult {
  const errors: string[] = [];
  const { rows, columns } = readFeatureRows(spec.features, spec.featureIdColumn, spec.patientColumn, spec.featureColumns, errors);

  // 臨床
  const ct = spec.clinical.table;
  const iCid = ct.header.indexOf(spec.clinicalIdColumn);
  if (iCid < 0) errors.push(`臨床の表に列「${spec.clinicalIdColumn}」がありません`);
  const roleCols: { name: string; role: "time" | "event" | "outcome" | "covariate" }[] = [];
  const addRole = (name: string | null, role: "time" | "event" | "outcome") => {
    if (name) roleCols.push({ name, role });
  };
  addRole(spec.roles.time, "time");
  addRole(spec.roles.event, "event");
  addRole(spec.roles.outcome, "outcome");
  for (const c of spec.roles.covariates) roleCols.push({ name: c, role: "covariate" });
  for (const rc of roleCols) if (!ct.header.includes(rc.name)) errors.push(`臨床の表に列「${rc.name}」がありません`);
  const dupRole = roleCols.map((r) => r.name).filter((n, i, a) => a.indexOf(n) !== i);
  if (dupRole.length) errors.push(`同じ列を複数の役割に割り当てています: ${unique(dupRole).join(", ")}`);
  if (errors.length) return { ok: false, errors };

  const clinicalById = new Map<string, string[]>();
  ct.rows.forEach((r, k) => {
    const id = r[iCid].trim();
    if (id === "") errors.push(`臨床の表の ${k + 2} 行目: 「${spec.clinicalIdColumn}」が空です`);
    else if (clinicalById.has(id)) errors.push(`臨床の表で患者 ID「${id}」が重複しています`);
    else clinicalById.set(id, r);
  });

  // 患者ごとの病変
  const byPatient = new Map<string, FeatureRow[]>();
  for (const r of rows) {
    const list = byPatient.get(r.patient);
    if (list) list.push(r);
    else byPatient.set(r.patient, [r]);
  }
  if (!spec.patientColumn) {
    for (const [p, list] of byPatient) if (list.length > 1) errors.push(`患者 ID「${p}」が特徴量の表で重複しています（1 行 = 1 患者の指定）`);
  }
  if (errors.length) return { ok: false, errors };

  const featureIds = [...byPatient.keys()];
  const joinedIds = featureIds.filter((id) => clinicalById.has(id));
  const featureOnly = featureIds.filter((id) => !clinicalById.has(id));
  const clinicalOnly = [...clinicalById.keys()].filter((id) => !byPatient.has(id));
  const exclusions: Exclusion[] = [
    ...featureOnly.map((id) => ({ id, reason: "臨床の表にいない" })),
    ...clinicalOnly.map((id) => ({ id, reason: "特徴量の表にいない" })),
  ];
  const warnings: string[] = [];
  const lowerOf = (xs: string[]) => new Map(xs.map((x) => [x.toLowerCase(), x]));
  const lc = lowerOf(clinicalOnly);
  const caseHits = featureOnly.filter((f) => lc.has(f.toLowerCase()));
  if (caseHits.length) {
    warnings.push(`大文字小文字だけが違う ID があります（${preview(caseHits.map((f) => `${f} ↔ ${lc.get(f.toLowerCase())}`))}）。ID は一致した場合だけ結合します`);
  }

  // 臨床の役割の列を型どおりに読む（決まりの外は止める）
  const clinical = roleCols.map((rc) => {
    const j = ct.header.indexOf(rc.name);
    const values: ClinicalValue[] = [];
    const bad: string[] = [];
    for (const id of joinedIds) {
      const raw = clinicalById.get(id)![j];
      const c = parseCell(raw);
      let v: ClinicalValue;
      if (c.kind === "missing") v = null;
      else if (rc.role === "time") {
        if (c.kind !== "number" || !(c.value > 0)) { bad.push(`${id}=${raw}`); v = null; } else v = c.value;
      } else if (rc.role === "event") {
        if (c.kind !== "number" || (c.value !== 0 && c.value !== 1)) { bad.push(`${id}=${raw}`); v = null; } else v = c.value;
      } else v = c.kind === "number" ? c.value : c.value.trim();
      values.push(v);
    }
    if (bad.length) {
      errors.push(`列「${rc.name}」に ${rc.role === "time" ? "0 以下または数でない値（時間は正の数）" : "0・1 以外の値（イベントは 0 か 1）"}があります: ${preview(bad)}`);
    }
    return { name: rc.name, role: rc.role, values };
  });
  if (errors.length) return { ok: false, errors };

  const features = joinedIds.map((id) => columns.map((_, m) => meanOrNull(byPatient.get(id)!.map((r) => r.values[m]))));
  const lesionCounts = joinedIds.map((id) => byPatient.get(id)!.length);

  const missing: JoinReport["missing"] = [];
  columns.forEach((name, m) => {
    const count = features.filter((r) => r[m] === null).length;
    if (count > 0) missing.push({ name, count, of: joinedIds.length });
  });
  for (const c of clinical) {
    const count = c.values.filter((v) => v === null).length;
    if (count > 0) missing.push({ name: c.name, count, of: joinedIds.length });
  }
  const tc = clinical.find((c) => c.role === "time");
  const ec = clinical.find((c) => c.role === "event");
  const survivalUsable = tc && ec ? joinedIds.filter((_, i) => tc.values[i] !== null && ec.values[i] !== null).length : null;
  if (survivalUsable !== null) {
    for (let i = 0; i < joinedIds.length; i++) {
      if (tc!.values[i] === null || ec!.values[i] === null) exclusions.push({ id: joinedIds[i], reason: "時間またはイベントが欠損（生存解析から除く）" });
    }
  }
  if (joinedIds.length === 0) warnings.push("結合できた患者が 0 人です（ID の列の割り当てを確かめてください）");

  const counts = lesionCounts;
  return {
    ok: true,
    dataset: { ids: joinedIds, featureNames: columns, features, lesionCounts, clinical },
    report: {
      featureFiles: spec.features.length,
      featureRows: rows.length,
      patientsInFeatures: featureIds.length,
      patientsInClinical: clinicalById.size,
      joined: joinedIds.length,
      featureOnly,
      clinicalOnly,
      lesionsPerPatient: counts.length ? { min: Math.min(...counts), max: Math.max(...counts) } : null,
      missing,
      survivalUsable,
      exclusions,
      warnings,
    },
  };
}
