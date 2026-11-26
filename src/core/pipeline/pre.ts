/**
 * 前処理の段（Skill 選択の後の共通の段）。分散 0・相関の除外、対応ありの p 値による除外、前提の検定の報告。
 * 既定はすべてオフ（何も有効にしなければ、この段は何もしない）。しきい値・α は既定値を持たず、利用者が入れる。
 * 対応なしの p 値による除外（目的変数を使う）は、交差検証の各訓練の行の内側で行うので nestedCv.ts にある。
 */
import type { Dataset } from "../dataset";
import type { DeltaResult } from "../delta";
import type { ParamValue } from "../skill";
import { qcFeatures } from "./qc";
import { dropByPValue } from "../stats/dropByPValue";
import { assumptionReport, testCounts, type AssumptionRow } from "../stats/assumptionReport";

/** 分割ごとに共通の設定（キーは skills の defaults と画面で共有する）。 */
export const PRE_QC_DEFAULTS: Record<string, ParamValue> = { dropConstant: false, correlationThreshold: null };
export const PRE_PAIRED_DEFAULTS: Record<string, ParamValue> = { prePairedOn: false, prePairedAlpha: 0.05, prePairedThr: 0.05 };
export const PRE_UNPAIRED_DEFAULTS: Record<string, ParamValue> = { preUnpairedOn: false, preUnpairedAlpha: 0.05, preUnpairedThr: 0.05 };
export const PRE_ASSUMP_DEFAULTS: Record<string, ParamValue> = { preAssumpOn: false, preAssumpAlpha: 0.05 };

export interface AlphaThr {
  /** 検定の選択に使う α。 */
  alpha: number;
  /** p がこの値以上の特徴量を除く。 */
  threshold: number;
}

export interface PreSpec {
  dropConstant: boolean;
  correlationThreshold: number | null;
  /** 対応あり（前と後の表があるときだけ）。 */
  paired: AlphaThr | null;
  /** 対応なし（P4・二値の結果）。 */
  unpaired: AlphaThr | null;
  /** 前提の検定の報告（R2・P4 の二値）。 */
  assumption: { alpha: number } | null;
}

export interface PreCaps {
  qc: boolean;
  paired: boolean;
  unpaired: boolean;
  assumption: boolean;
}

const inOpen = (v: unknown, hi: number): v is number => typeof v === "number" && Number.isFinite(v) && v > 0 && v < hi;

export function readPre(p: Record<string, ParamValue>, caps: PreCaps): PreSpec | { error: string } {
  const need = (what: string) => ({ error: `${what}を入れてください` });
  const pair = (on: ParamValue, a: ParamValue, t: ParamValue, label: string): AlphaThr | null | { error: string } => {
    if (on !== true) return null;
    if (!inOpen(a, 1)) return a === null || a === undefined ? need(`${label}の α`) : { error: `${label}の α は 0 と 1 の間で指定してください` };
    if (!(typeof t === "number" && Number.isFinite(t) && t > 0 && t <= 1)) return t === null || t === undefined ? need(`${label}の p のしきい値`) : { error: `${label}の p のしきい値は 0 より大きく 1 以下で指定してください` };
    return { alpha: a, threshold: t };
  };
  let ct: number | null = null;
  if (caps.qc) {
    const v = p.correlationThreshold;
    if (v !== null && v !== undefined) {
      if (typeof v !== "number" || !(v > 0 && v <= 1)) return { error: "相関のしきい値は 0 より大きく 1 以下にしてください" };
      ct = v;
    }
  }
  const paired = caps.paired ? pair(p.prePairedOn, p.prePairedAlpha, p.prePairedThr, "対応ありの p 値による除外") : null;
  if (paired && "error" in paired) return paired;
  const unpaired = caps.unpaired ? pair(p.preUnpairedOn, p.preUnpairedAlpha, p.preUnpairedThr, "対応なしの p 値による除外") : null;
  if (unpaired && "error" in unpaired) return unpaired;
  let assumption: { alpha: number } | null = null;
  if (caps.assumption && p.preAssumpOn === true) {
    if (!inOpen(p.preAssumpAlpha, 1)) return p.preAssumpAlpha === null || p.preAssumpAlpha === undefined ? need("前提の検定の報告の α") : { error: "前提の検定の報告の α は 0 と 1 の間で指定してください" };
    assumption = { alpha: p.preAssumpAlpha };
  }
  return { dropConstant: caps.qc && p.dropConstant === true, correlationThreshold: ct, paired: paired as AlphaThr | null, unpaired: unpaired as AlphaThr | null, assumption };
}

/** 画面・記録用: 前処理を 1 つでも有効にしているか。 */
export const preActive = (s: PreSpec) => s.dropConstant || s.correlationThreshold !== null || s.paired !== null || s.unpaired !== null || s.assumption !== null;

export interface PreRemoval {
  feature: string;
  step: "const" | "corr" | "paired";
  reason: { ja: string; en: string };
}

/** qcFeatures の理由（日本語）の英訳。qc.ts の文言は変えない。 */
function qcReasonEn(ja: string): string {
  if (ja === "開発集合で値が 1 種類しかない" || ja === "解析対象で値が 1 種類しかない") return "only one distinct value";
  if (ja === "開発集合で全部欠損" || ja === "解析対象で全部欠損") return "all values missing";
  const m = ja.match(/^「(.*)」と \|ρ\| = (.*) ≥ (.*)$/);
  if (m) return `|rho| = ${m[2]} >= ${m[3]} with "${m[1]}"`;
  return ja;
}

export interface PairedInfo {
  alpha: number;
  threshold: number;
  /** 検定に使った行（患者）の数。 */
  patients: number;
  /** 選ばれた検定ごとの件数。 */
  tests: Record<string, number>;
  /** 検定した特徴量ごと。 */
  records: { feature: string; n: number; test: string | null; p: number | null; kept: boolean }[];
}

export interface PreInfo {
  spec: PreSpec;
  /** 前処理の前後の特徴量の数。 */
  before: number;
  after: number;
  removed: PreRemoval[];
  paired: PairedInfo | null;
  /** 前提の検定の報告（報告だけ。結果は変えない）。 */
  assumptions: { alpha: number; rows: AssumptionRow[] } | null;
  /** 分散 0・相関が値を読んだ行と、対応ありが読んだ行（行番号は呼び出し側の行列の行）。 */
  qcRows: number[];
  pairedRows: number[];
}

export const emptyPre = (spec: PreSpec, n: number): PreInfo => ({ spec, before: n, after: n, removed: [], paired: null, assumptions: null, qcRows: [], pairedRows: [] });

export interface PairedMatrix {
  /** [行][列]。呼び出し側の行列と同じ並び。 */
  before: (number | null)[][];
  after: (number | null)[][];
}

/**
 * 変化率から作ったデータセットの特徴量（Delta_ 列）に対応する、患者ごとの前・後の値。rowsIdx は dataset の患者の番号。
 * 前後の表の患者・特徴量に見つからなければ理由を返す（欠損で埋めない）。
 */
export function pairedFromDelta(d: Dataset, delta: DeltaResult, rowsIdx: number[]): PairedMatrix | { error: string } {
  const pi = new Map(delta.patients.map((x, i) => [x, i]));
  const fi = d.featureNames.map((n) => delta.featureNames.indexOf(n.startsWith("Delta_") ? n.slice("Delta_".length) : n));
  const lost = d.featureNames.filter((_, j) => fi[j] < 0);
  if (lost.length) return { error: `前後の値が見つからない特徴量があります: ${lost.slice(0, 5).join(", ")}` };
  const rows = rowsIdx.map((i) => pi.get(d.ids[i]));
  if (rows.some((r) => r === undefined)) return { error: "前後の表に見つからない患者がいます" };
  const take = (m: (number | null)[][]) => rows.map((r) => d.featureNames.map((_, j) => m[r as number][fi[j]]));
  return { before: take(delta.before), after: take(delta.after) };
}

/**
 * 対応ありの p 値による除外。cols（X の列番号）の特徴量を、rows の患者の前・後で対にして検定し、
 * p がしきい値以上（または検定できない）列を落とす。読むのは rows の前・後の値だけ。
 */
export function pairedDrop(pm: PairedMatrix, names: string[], cols: number[], rows: number[], a: AlphaThr): { keep: number[]; removed: PreRemoval[]; info: PairedInfo } | { error: string } {
  const stacked = [...rows.map((i) => cols.map((j) => pm.before[i][j])), ...rows.map((i) => cols.map((j) => pm.after[i][j]))];
  const pairs: [number, number][] = rows.map((_, q) => [q, rows.length + q]);
  const r = dropByPValue(stacked, cols.map((j) => names[j]), { kind: "paired", pairs }, a.alpha, a.threshold);
  if ("error" in r) return r;
  const removed: PreRemoval[] = r.records.filter((x) => !x.kept).map((x) => ({ feature: x.feature, step: "paired", reason: x.dropReason as { ja: string; en: string } }));
  const info: PairedInfo = {
    alpha: a.alpha, threshold: a.threshold, patients: rows.length,
    tests: testCounts(r.records.map((x) => ({ test: x.selection?.test ?? null }))),
    records: r.records.map((x) => ({ feature: x.feature, n: x.n[0], test: x.selection?.test ?? null, p: x.p, kept: x.kept })),
  };
  return { keep: r.keep.map((q) => cols[q]), removed, info };
}

export interface PreInput {
  names: string[];
  /** [行][特徴量]（欠損は null）。 */
  X: (number | null)[][];
  /** 解析に使う行（分散 0・相関・対応ありが読む行）。 */
  rows: number[];
  spec: PreSpec;
  paired: PairedMatrix | null;
  /** 前提の検定の報告に使う群（行番号は X の行）。報告しないなら null。 */
  groups: { rows0: number[]; rows1: number[] } | null;
  /** false なら分散 0・相関は行わない（R5 は手順の中で行う）。 */
  qc: boolean;
  /** 理由の「開発集合」を「解析対象」に言い換える（交差検証のない R1・R2・R5）。 */
  wholeCohort: boolean;
}

/** 分散 0 → 相関 → 対応ありの順に除外し、報告を作る。有効な手順が無ければ何も読まず何も変えない。 */
export function runPre(inp: PreInput): { keep: number[]; info: PreInfo } | { error: string } {
  const { names, X, rows, spec } = inp;
  const info = emptyPre(spec, names.length);
  let keep = names.map((_, j) => j);
  if (inp.qc && (spec.dropConstant || spec.correlationThreshold !== null)) {
    const q = qcFeatures(X, rows, names, { dropConstant: spec.dropConstant, correlationThreshold: spec.correlationThreshold, exclude: null });
    keep = q.keep;
    info.qcRows = q.usedRows;
    for (const r of q.removed) {
      const ja = inp.wholeCohort ? r.reason.replace("開発集合", "解析対象") : r.reason;
      info.removed.push({ feature: r.feature, step: r.reason.includes("|ρ|") ? "corr" : "const", reason: { ja, en: qcReasonEn(ja) } });
    }
  }
  if (spec.paired) {
    if (!inp.paired) return { error: "対応ありの p 値による除外には、前と後の表が要ります" };
    const r = pairedDrop(inp.paired, names, keep, rows, spec.paired);
    if ("error" in r) return r;
    keep = r.keep;
    info.removed.push(...r.removed);
    info.paired = r.info;
    info.pairedRows = rows;
  }
  if (!keep.length) return { error: "前処理の後に残った特徴量がありません" };
  info.after = keep.length;
  if (spec.assumption && inp.groups) {
    const kept = keep.map((j) => names[j]);
    const sub = X.map((r) => keep.map((j) => r[j]));
    const rep = assumptionReport(sub, kept, inp.groups.rows0, inp.groups.rows1, spec.assumption.alpha);
    if ("error" in rep) return rep;
    info.assumptions = { alpha: spec.assumption.alpha, rows: rep };
  }
  return { keep, info };
}

/** 記録用の要約（RunRecord の summary は平らな値だけ）。 */
export function preSummary(info: PreInfo): Record<string, string | number | boolean | null> {
  const s = info.spec;
  const n = (st: PreRemoval["step"]) => info.removed.filter((r) => r.step === st).length;
  return {
    dropConstant: s.dropConstant, correlationThreshold: s.correlationThreshold,
    pairedAlpha: s.paired?.alpha ?? null, pairedThreshold: s.paired?.threshold ?? null,
    unpairedAlpha: s.unpaired?.alpha ?? null, unpairedThreshold: s.unpaired?.threshold ?? null,
    assumptionAlpha: s.assumption?.alpha ?? null,
    featuresBefore: info.before, featuresAfter: info.after,
    removedConstant: n("const"), removedCorrelation: n("corr"), removedPaired: n("paired"),
    pairedTests: info.paired ? Object.entries(info.paired.tests).map(([k, v]) => `${k}:${v}`).join(" ") : null,
    removedList: info.removed.map((r) => `${r.feature}(${r.step})`).join(", ") || null,
  };
}

/** 記録用の詳細（除外した特徴量と理由・検定の選択の件数）。 */
export function preDetail(info: PreInfo): Record<string, unknown> {
  return {
    removed: info.removed.map((r) => ({ feature: r.feature, step: r.step, reason: r.reason.ja })),
    paired: info.paired ? { alpha: info.paired.alpha, threshold: info.paired.threshold, patients: info.paired.patients, tests: info.paired.tests } : null,
    assumptions: info.assumptions ? { alpha: info.assumptions.alpha, tests: testCounts(info.assumptions.rows) } : null,
  };
}
