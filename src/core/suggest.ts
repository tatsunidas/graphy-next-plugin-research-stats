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

const POSITIVE_WORDS = new Set(["1", "true", "yes", "positive", "pos", "mutant", "mutation", "mutated", "mut", "陽性", "変異", "変異あり", "あり"]);
const NEGATIVE_WORDS = new Set(["0", "false", "no", "negative", "neg", "wildtype", "wild-type", "wild type", "wt", "陰性", "野生型", "なし"]);

/**
 * 陽性とするラベルの初期値（選び直せる）。ラベルがちょうど 2 種類のときだけ返す。
 * 名前で分かれば陽性の側（片方が陰性の語なら、もう片方）。分からなければ少ない方（同数なら並びが後の方）。
 */
export function suggestPositive(values: readonly string[]): string | null {
  const count = new Map<string, number>();
  for (const v of values) { const x = v.trim(); if (x !== "") count.set(x, (count.get(x) ?? 0) + 1); }
  const labels = [...count.keys()].sort();
  if (labels.length !== 2) return null;
  const [a, b] = labels;
  const has = (set: Set<string>, x: string) => set.has(x.normalize("NFKC").toLowerCase());
  const score = (x: string, other: string) => Number(has(POSITIVE_WORDS, x)) + Number(has(NEGATIVE_WORDS, other));
  if (score(a, b) !== score(b, a)) return score(a, b) > score(b, a) ? a : b;
  return count.get(a)! < count.get(b)! ? a : b;
}
