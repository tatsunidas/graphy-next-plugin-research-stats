/** 列の割り当ての候補（見出しの名前から。確定は利用者）。 */
import type { CsvTable } from "./csv";

const first = (header: readonly string[], ...patterns: RegExp[]): string | null => {
  for (const p of patterns) {
    const hit = header.find((h) => p.test(h));
    if (hit) return hit;
  }
  return null;
};

const PATIENT = /^(patient|subject)[_ ]?(id|label|code)?$/i;

export interface Suggestion {
  featureId: string;
  patient: string | null;
  clinicalId: string;
  time: string | null;
  event: string | null;
  outcome: string | null;
}

export function suggestColumns(features: CsvTable | null, clinical: CsvTable | null): Suggestion {
  const fh = features?.header ?? [];
  const ch = clinical?.header ?? [];
  const featureId = first(fh, /^CaseLabel$/i, /^(case|lesion|roi)[_ ]?id$/i) ?? fh[0] ?? "";
  const patient = first(fh, PATIENT);
  const patientName = patient && patient !== featureId ? patient.toLowerCase() : null;
  const sameAsPatient = patientName ? ch.find((h) => h.toLowerCase() === patientName) : undefined;
  const clinicalId = sameAsPatient ?? first(ch, /^CaseLabel$/i, PATIENT) ?? ch[0] ?? "";
  return {
    featureId,
    patient: patient && patient !== featureId ? patient : null,
    clinicalId,
    time: first(ch, /^(os|pfs|rfs|dfs)[_ ]?(time|months?|days?)?$/i, /(time|month|day|duration|survival)/i),
    event: first(ch, /^(event|status)$/i, /(event|status|death|dead|censor|recur|progress)/i),
    outcome: first(ch, /^(outcome|label|class|group|response)$/i, /(outcome|label|class|response)/i),
  };
}
