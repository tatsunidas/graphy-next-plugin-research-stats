/**
 * データリークの検査（純関数）。しきい値は使わず、定義で決まる条件だけで判定する。
 *  1. ID・由来・記録の列が特徴量に混ざっていないか（idLikeColumns）
 *  2. 特徴量が目的変数（結果ラベル・時間・イベント）そのもの・完全な代理でないか（targetLeaks）
 *  3. 同じ患者が訓練と検証に分かれていないか（checkPatientSplit）
 *  4. 前処理・特徴量選択が、訓練側の行だけから推定されているか（checkRowUse）
 */
import { parseCell, type CsvTable } from "./csv";
import type { Dataset } from "./dataset";
import { ranks } from "./pipeline/qc";
import type { RowTrace } from "./pipeline/rowTrace";

// 1. ID・由来・記録の列 -----------------------------------------------------------------------

const NAME_PREFIX = /^(Provenance_|Warning_|Roi_|Row|Filter_|Diagnostics_)/;
const NAME_EXACT = new Set(["Unit", "PlaneSource", "Plane", "RowKey", "CaseLabel", "PatientID", "SubjectCode", "Album"]);

export type IdLikeReason = "name" | "uid" | "dateTime" | "sequential";

export interface IdLikeColumn {
  column: string;
  reasons: IdLikeReason[];
  /** 特徴量として使われているか（使っていれば警告、使っていなければ「除外済み」）。 */
  used: boolean;
}

export function nameReasons(name: string): IdLikeReason[] {
  const r: IdLikeReason[] = [];
  if (NAME_PREFIX.test(name) || NAME_EXACT.has(name)) r.push("name");
  if (name.endsWith("UID")) r.push("uid");
  if (name.includes("Date") || name.includes("Time")) r.push("dateTime");
  return r;
}

/** ファイルの行の順に、整数が公差一定（0 でない）で厳密に単調に並ぶ（1,2,3,… / 10,20,30,… / 5,4,3,…）。欠損・空・小数があるか 3 行未満なら対象外。 */
export function isIntegerSequence(tables: readonly CsvTable[], column: string): boolean {
  const v: number[] = [];
  for (const t of tables) {
    const j = t.header.indexOf(column);
    if (j < 0) return false;
    for (const r of t.rows) {
      const c = parseCell(r[j] ?? "");
      if (c.kind !== "number" || !Number.isInteger(c.value)) return false;
      v.push(c.value);
    }
  }
  if (v.length < 3) return false;
  const d = v[1] - v[0];
  return d !== 0 && v.every((x, i) => i === 0 || x - v[i - 1] === d);
}

/** skip（行の ID・患者の列）以外で、名前か値が ID・由来・記録に当たる列。used は usedColumns に含まれるか。 */
export function idLikeColumns(tables: readonly CsvTable[], skip: readonly (string | null)[], usedColumns: readonly string[]): IdLikeColumn[] {
  const out: IdLikeColumn[] = [];
  const used = new Set(usedColumns);
  for (const column of tables[0]?.header ?? []) {
    if (skip.includes(column)) continue;
    const reasons = nameReasons(column);
    if (isIntegerSequence(tables, column)) reasons.push("sequential");
    if (reasons.length) out.push({ column, reasons, used: used.has(column) });
  }
  return out;
}

// 2. 目的変数の混入 ---------------------------------------------------------------------------

export type TargetRole = "time" | "event" | "outcome";
export type TargetLeakKind = "sameName" | "identical" | "separates" | "rankIdentical";

export interface TargetLeak {
  feature: string;
  role: TargetRole;
  /** 臨床の列の名前。 */
  target: string;
  kinds: TargetLeakKind[];
}

type Pair = { x: number; y: number };

/** 特徴量 x と目的変数の列 y の、両方が揃っている行。y が数でなければ null（数値化できない）。 */
function pairs(x: (number | null)[], y: (number | string | null)[]): Pair[] {
  const out: Pair[] = [];
  for (let i = 0; i < x.length; i++) {
    const v = y[i];
    if (x[i] === null || v === null || v === "") continue;
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n)) continue;
    out.push({ x: x[i] as number, y: n });
  }
  return out;
}

/** 2 群（分類の 2 値）を完全に分けるか: 一方の全員が他方の全員より大きい（AUC が正確に 1 か 0）。各群 2 人以上。 */
export function separatesCompletely(x: (number | null)[], label: (number | string | null)[]): boolean {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < x.length; i++) {
    const l = label[i];
    if (x[i] === null || l === null || l === "") continue;
    const k = String(l);
    const g = groups.get(k);
    if (g) g.push(x[i] as number);
    else groups.set(k, [x[i] as number]);
  }
  if (groups.size !== 2) return false;
  const [a, b] = [...groups.values()];
  if (a.length < 2 || b.length < 2) return false;
  return Math.min(...a) > Math.max(...b) || Math.max(...a) < Math.min(...b);
}

/** 順位が完全に一致（または完全に逆）: Spearman の |ρ| が正確に 1。両方が揃った 3 行以上で、どちらも定数でない。 */
export function rankIdentical(p: readonly Pair[]): boolean {
  if (p.length < 3) return false;
  const rx = ranks(p.map((q) => q.x)), ry = ranks(p.map((q) => q.y));
  if (new Set(rx).size < 2 || new Set(ry).size < 2) return false;
  const n = p.length;
  return rx.every((v, i) => v === ry[i]) || rx.every((v, i) => v === n + 1 - ry[i]);
}

/** 結合後のデータセットの特徴量が、役割の列（結果ラベル・時間・イベント）の混入でないか。 */
export function targetLeaks(d: Dataset): TargetLeak[] {
  const out: TargetLeak[] = [];
  for (const c of d.clinical) {
    if (c.role === "covariate") continue;
    const role = c.role;
    d.featureNames.forEach((feature, j) => {
      const x = d.features.map((r) => r[j]);
      const kinds: TargetLeakKind[] = [];
      if (feature === c.name) kinds.push("sameName");
      const p = pairs(x, c.values);
      if (p.length > 0 && p.every((q) => q.x === q.y)) kinds.push("identical");
      if (role !== "time" && separatesCompletely(x, c.values)) kinds.push("separates");
      if (role === "time" && rankIdentical(p)) kinds.push("rankIdentical");
      if (kinds.length) out.push({ feature, role, target: c.name, kinds });
    });
  }
  return out;
}

// 3. 同じ患者が訓練と検証に分かれていないか ----------------------------------------------------------

export interface SplitViolation {
  /** "outer"・"inner"・"test"。 */
  level: "outer" | "inner" | "test";
  repeat: number | null;
  fold: number | null;
  patient: string;
}

export interface SplitCheck {
  ok: boolean;
  /** 検査した分割（検証の側）の数。 */
  checked: number;
  /** 該当した患者 ID（重複を除く）。 */
  patients: string[];
  violations: SplitViolation[];
}

/** 割り当て assign[p]（p = 並びの位置）の分割ごとに、検証の側と訓練の側に同じ患者がいないか。 */
function crossCheck(assign: readonly number[], patient: readonly string[], k: number, level: "outer" | "inner", repeat: number, fold: number | null, out: SplitViolation[]): number {
  let checked = 0;
  for (let f = 0; f < k; f++) {
    const valid = new Set<string>();
    assign.forEach((a, p) => { if (a === f) valid.add(patient[p]); });
    if (!valid.size) continue;
    checked++;
    const seen = new Set<string>();
    assign.forEach((a, p) => {
      if (a !== f && valid.has(patient[p]) && !seen.has(patient[p])) {
        seen.add(patient[p]);
        out.push({ level, repeat, fold: level === "outer" ? f : fold, patient: patient[p] });
      }
    });
  }
  return checked;
}

export interface SplitInput {
  outerK: number;
  innerK: number;
  /** [反復][開発集合の位置] → 外側の分割番号。 */
  outerAssign: readonly (readonly number[])[];
  /** [反復][外側の分割][外側の訓練の行の並び] → 内側の分割番号（空なら内側を行わなかった）。 */
  innerAssign: readonly (readonly (readonly number[])[])[];
}

/**
 * 実際の割り当てから、同じ患者 ID が訓練と検証の両方にいないことを検査する。
 * patientOfDev は開発集合の位置ごとの患者 ID。testPatients があれば、開発集合との重なりも見る。
 */
export function checkPatientSplit(cv: SplitInput, patientOfDev: readonly string[], testPatients: readonly string[] = []): SplitCheck {
  const violations: SplitViolation[] = [];
  let checked = 0;
  cv.outerAssign.forEach((assign, r) => {
    checked += crossCheck(assign, patientOfDev, cv.outerK, "outer", r, null, violations);
    for (let f = 0; f < cv.outerK; f++) {
      const inner = cv.innerAssign[r]?.[f];
      if (!inner || !inner.length) continue;
      const trainPatients = patientOfDev.filter((_, p) => assign[p] !== f);
      checked += crossCheck(inner, trainPatients, cv.innerK, "inner", r, f, violations);
    }
  });
  if (testPatients.length) {
    const dev = new Set(patientOfDev);
    checked++;
    for (const p of new Set(testPatients)) if (dev.has(p)) violations.push({ level: "test", repeat: null, fold: null, patient: p });
  }
  return { ok: violations.length === 0, checked, patients: [...new Set(violations.map((v) => v.patient))], violations };
}

// 4. 前処理・特徴量選択が訓練側の行だけか -------------------------------------------------------------

export interface RowUseFinding {
  stage: string;
  where: "pre" | "fold" | "inner" | "final" | "finalInner";
  /** 目的変数を使う段階か（false は、特徴量の値だけを読む教師なしの段階）。 */
  usesOutcome: boolean;
  /** 記録した推定の数と、そのうち訓練側でない行を使っていたもの。 */
  entries: number;
  violating: number;
  /** 訓練側でない行の数の最大。 */
  maxOutside: number;
}

/**
 * ok: すべて訓練側の行だけ。
 * notice: 教師なしの段階だけが、訓練側でない行の値を読んだ（結果ラベルは使っていない）。
 * ng: 目的変数を使う段階が、訓練側でない行を使った。
 */
export type RowUseLevel = "ok" | "notice" | "ng";

export interface RowUseCheck {
  level: RowUseLevel;
  findings: RowUseFinding[];
}

/**
 * 記録した「推定に使った行」が、その範囲の訓練側の部分集合か（検証の側の行を含まないか）を検査する。
 * 交差検証の前（"pre"）に行った推定は、すべての外側の分割の訓練側の部分集合でなければ違反とする。
 */
export function checkRowUse(trace: RowTrace): RowUseCheck {
  const groups = new Map<string, RowUseFinding>();
  const folds = [...trace.contexts].filter(([k]) => /^r\d+f\d+$/.test(k));
  const whereOf = (scope: string): RowUseFinding["where"] =>
    scope === "pre" ? "pre" : scope === "final" ? "final" : scope.startsWith("final.") ? "finalInner" : scope.includes(".") ? "inner" : "fold";
  for (const u of trace.uses) {
    const where = whereOf(u.scope);
    const key = `${where}/${u.stage}`;
    const g = groups.get(key) ?? { stage: u.stage, where, usesOutcome: u.usesOutcome, entries: 0, violating: 0, maxOutside: 0 };
    const targets = where === "pre" ? folds.map(([, c]) => c) : trace.contexts.has(u.scope) ? [trace.contexts.get(u.scope)!] : [];
    for (const c of targets) {
      g.entries++;
      const train = new Set(c.train);
      const outside = u.rows.filter((i) => !train.has(i)).length;
      if (outside > 0) {
        g.violating++;
        g.maxOutside = Math.max(g.maxOutside, outside);
      }
    }
    groups.set(key, g);
  }
  const findings = [...groups.values()];
  const level: RowUseLevel = findings.some((f) => f.usesOutcome && f.violating > 0) ? "ng" : findings.some((f) => f.violating > 0) ? "notice" : "ok";
  return { level, findings };
}
