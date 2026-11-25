/**
 * 目的変数に選んだ臨床の列の型を、実行の前に検査する（dataset.ts が実行時に止める決まりと同じ:
 * 時間は正の数・イベントは 0 か 1）。欠損は欠損のまま数え、0 にしない。
 */
import { parseCell, type CsvTable } from "./csv";
import type { ParamValue } from "./skill";

const column = (t: CsvTable, name: string): string[] | null => {
  const j = t.header.indexOf(name);
  return j < 0 ? null : t.rows.map((r) => r[j] ?? "");
};
const sample = (xs: string[], n = 3) => xs.slice(0, n);

export interface TimeQc { rows: number; positive: number; missing: number; invalid: number; invalidExamples: string[] }
export interface EventQc { rows: number; zero: number; one: number; missing: number; invalid: number; invalidExamples: string[] }
export interface OutcomeQc { rows: number; missing: number; values: { label: string; count: number }[]; binary: boolean }
export interface CovariateQc { rows: number; numeric: boolean; missing: number; textExamples: string[] }

export function timeQc(t: CsvTable, name: string): TimeQc | null {
  const col = column(t, name);
  if (!col) return null;
  const q: TimeQc = { rows: col.length, positive: 0, missing: 0, invalid: 0, invalidExamples: [] };
  for (const raw of col) {
    const c = parseCell(raw);
    if (c.kind === "missing") q.missing++;
    else if (c.kind === "number" && c.value > 0) q.positive++;
    else {
      q.invalid++;
      q.invalidExamples.push(raw.trim());
    }
  }
  q.invalidExamples = sample(q.invalidExamples);
  return q;
}

export function eventQc(t: CsvTable, name: string): EventQc | null {
  const col = column(t, name);
  if (!col) return null;
  const q: EventQc = { rows: col.length, zero: 0, one: 0, missing: 0, invalid: 0, invalidExamples: [] };
  for (const raw of col) {
    const c = parseCell(raw);
    if (c.kind === "missing") q.missing++;
    else if (c.kind === "number" && c.value === 0) q.zero++;
    else if (c.kind === "number" && c.value === 1) q.one++;
    else {
      q.invalid++;
      q.invalidExamples.push(raw.trim());
    }
  }
  q.invalidExamples = sample([...new Set(q.invalidExamples)]);
  return q;
}

export function outcomeQc(t: CsvTable, name: string): OutcomeQc | null {
  const col = column(t, name);
  if (!col) return null;
  const counts = new Map<string, number>();
  let missing = 0;
  for (const raw of col) {
    if (parseCell(raw).kind === "missing") missing++;
    else counts.set(raw.trim(), (counts.get(raw.trim()) ?? 0) + 1);
  }
  const values = [...counts].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  return { rows: col.length, missing, values, binary: values.length === 2 };
}

export function covariateQc(t: CsvTable, name: string): CovariateQc | null {
  const col = column(t, name);
  if (!col) return null;
  let missing = 0;
  const text: string[] = [];
  for (const raw of col) {
    const c = parseCell(raw);
    if (c.kind === "missing") missing++;
    else if (c.kind === "text") text.push(raw.trim());
  }
  return { rows: col.length, numeric: text.length === 0, missing, textExamples: sample([...new Set(text)]) };
}

export interface RoleNeeds {
  time: boolean;
  event: boolean;
  outcome: boolean;
  positive: boolean;
  covariates: boolean;
}

const NONE: RoleNeeds = { time: false, event: false, outcome: false, positive: false, covariates: false };

/** 解析の種類ごとに、臨床の表から指定する役割。 */
export function rolesForSkill(skillId: string | null, params: Record<string, ParamValue>): RoleNeeds {
  switch (skillId) {
    case "r1-prognosis": return { ...NONE, time: true, event: true, covariates: true };
    case "r2-discrimination": return { ...NONE, outcome: true, positive: true };
    case "r5-model": return { ...NONE, time: true, event: true };
    case "p4-pipeline":
      return params.outcomeKind === "binary" ? { ...NONE, outcome: true, positive: true }
        : params.outcomeKind === "survival" ? { ...NONE, time: true, event: true } : NONE;
    default: return NONE;
  }
}

/** 実行に渡すとき、その解析で使わない役割の列を外す（使わない列の型の誤りで止めない）。 */
export function sanitizeRoles(needs: RoleNeeds, params: Record<string, ParamValue>): Record<string, ParamValue> {
  return {
    ...params,
    timeColumn: needs.time ? params.timeColumn : null,
    eventColumn: needs.event ? params.eventColumn : null,
    outcomeColumn: needs.outcome ? params.outcomeColumn : null,
    covariates: needs.covariates ? params.covariates : [],
  };
}
