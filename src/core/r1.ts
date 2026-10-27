/**
 * スキル R1「予後」の計算本体。特徴量ごとに群分け → Kaplan–Meier / log-rank / Cox → 多重比較の補正 →
 * 有意な特徴量と臨床因子の多変量 Cox → 臨床モデルへの追加効果（−2LL の差）と補正 C-index。
 * 画面にも skills.ts にも依存しない（試験しやすいよう純粋に保つ）。
 */
import type { Dataset } from "./dataset";
import type { ParamValue } from "./skill";
import type { Rng } from "./rng";
import { chi2Sf } from "./survival/dist";
import { bootstrapCorrectedC, cIndex, type CorrectedC, type FitFn } from "./survival/cindex";
import { coxFit, lrDifference, type CoxFit } from "./survival/cox";
import { groupByCuts, quantile7, type Grouping } from "./survival/grouping";
import { kaplanMeier } from "./survival/km";
import { logrank } from "./survival/logrank";
import { mannWhitney } from "./survival/mannwhitney";
import { benjaminiHochberg } from "./survival/multiplicity";
import { CutScanner, PermutationRun } from "./survival/permutation";
import { rmstDifference } from "./survival/rmst";
import type { Surv } from "./survival/types";

export type GroupKind = "median" | "value" | "tertile" | "optimal";
export type Multiplicity = "bh" | "none";

export interface R1Params {
  groupMode: GroupKind;
  groupValue: number | null;
  multiplicity: Multiplicity;
  alpha: number;
  minGroupFraction: number;
  permutations: number;
  rmstTau: number | null;
  mannWhitney: boolean;
  bootstrapB: number;
}

export const R1_DEFAULTS: Record<string, ParamValue> = {
  groupMode: "median",
  groupValue: null,
  multiplicity: "bh",
  alpha: 0.05,
  minGroupFraction: 0.1,
  permutations: 1000,
  rmstTau: null,
  mannWhitney: false,
  bootstrapB: 500,
};

export function readR1Params(p: Record<string, ParamValue>): R1Params | { error: string } {
  const num = (k: string, lo: number, hi: number, int = false): number | { error: string } => {
    const v = p[k];
    if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi || (int && !Number.isInteger(v))) {
      return { error: `設定「${k}」は ${lo} 以上 ${hi} 以下の${int ? "整数" : "数"}にしてください` };
    }
    return v;
  };
  const gm = p.groupMode;
  if (gm !== "median" && gm !== "value" && gm !== "tertile" && gm !== "optimal") return { error: "群分けは median / value / tertile / optimal のどれかです" };
  const mu = p.multiplicity;
  if (mu !== "bh" && mu !== "none") return { error: "補正は bh か none です" };
  if (gm === "value" && (typeof p.groupValue !== "number" || !Number.isFinite(p.groupValue))) return { error: "しきい値を指定してください" };
  const alpha = num("alpha", 1e-6, 0.5), mgf = num("minGroupFraction", 0.01, 0.4), perms = num("permutations", 19, 100000, true), B = num("bootstrapB", 0, 100000, true);
  for (const v of [alpha, mgf, perms, B]) if (typeof v === "object") return v;
  const tau = p.rmstTau;
  if (tau !== null && (typeof tau !== "number" || !(tau > 0))) return { error: "τ は正の数にしてください（使わないなら空欄）" };
  return {
    groupMode: gm, groupValue: gm === "value" ? (p.groupValue as number) : null, multiplicity: mu,
    alpha: alpha as number, minGroupFraction: mgf as number, permutations: perms as number,
    rmstTau: tau as number | null, mannWhitney: p.mannWhitney === true, bootstrapB: B as number,
  };
}

export interface FeatureResult {
  feature: string;
  n: number;
  events: number;
  cuts: number[];
  groupSizes: number[];
  groupEvents: number[];
  /** 群ごとの生存時間の中央値（到達しなければ null）。 */
  medians: (number | null)[];
  logrankChi2: number | null;
  logrankP: number | null;
  /** 最適しきい値のときだけ: 探索した素の p と、並べ替え検定の補正 p。 */
  rawOptimalP: number | null;
  correctedP: number | null;
  /** 多重比較の補正の入力にした p。 */
  pUsed: number | null;
  q: number | null;
  /** 高群 vs 低群の HR（Cox）。三分位なら最上位 vs 最下位。 */
  hrGroup: number | null;
  hrGroupLower: number | null;
  hrGroupUpper: number | null;
  hrGroupP: number | null;
  /** 連続値（標準化。SD あたり）の単変量 Cox。 */
  hrPerSd: number | null;
  hrPerSdLower: number | null;
  hrPerSdUpper: number | null;
  hrPerSdP: number | null;
  mannWhitneyP: number | null;
  rmstDiff: number | null;
  rmstP: number | null;
  /** 計算できなかった理由。p などは null のまま（0 や 1 にしない）。 */
  error: string | null;
}

const emptyResult = (feature: string): FeatureResult => ({
  feature, n: 0, events: 0, cuts: [], groupSizes: [], groupEvents: [], medians: [], logrankChi2: null, logrankP: null,
  rawOptimalP: null, correctedP: null, pUsed: null, q: null, hrGroup: null, hrGroupLower: null, hrGroupUpper: null, hrGroupP: null,
  hrPerSd: null, hrPerSdLower: null, hrPerSdUpper: null, hrPerSdP: null, mannWhitneyP: null, rmstDiff: null, rmstP: null, error: null,
});

export interface SurvivalCohort {
  /** 解析に使う患者（Dataset の並びでの番号）。 */
  patients: number[];
  surv: Surv;
}

/** 時間とイベントが揃った患者だけの生存データ。 */
export function survivalCohort(d: Dataset): SurvivalCohort | { error: string } {
  const tc = d.clinical.find((c) => c.role === "time"), ec = d.clinical.find((c) => c.role === "event");
  if (!tc || !ec) return { error: "時間とイベントの列を割り当ててください" };
  const patients: number[] = [], time: number[] = [], event: number[] = [];
  d.ids.forEach((_, i) => {
    const t = tc.values[i], e = ec.values[i];
    if (typeof t === "number" && typeof e === "number") { patients.push(i); time.push(t); event.push(e); }
  });
  const events = event.reduce((a, b) => a + b, 0);
  if (patients.length < 6) return { error: `時間とイベントが揃った患者が ${patients.length} 人しかいません（6 人以上要ります）` };
  if (events < 2) return { error: `イベントが ${events} 件しかありません（2 件以上要ります）` };
  return { patients, surv: { time, event } };
}

const sd = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
};

function subSurv(s: Surv, ix: number[]): Surv {
  return { time: ix.map((i) => s.time[i]), event: ix.map((i) => s.event[i]) };
}

/** 群ごとの HR（高 vs 低）。三分位は中間群のダミーも入れて最上位 vs 最下位を出す。 */
function groupHr(s: Surv, g: Grouping): { hr: number; lo: number; hi: number; p: number } | { error: string } {
  const X = g.group.map((v) => (g.k === 2 ? [v] : [v === 1 ? 1 : 0, v === 2 ? 1 : 0]));
  const r = coxFit(s, X);
  if (!r.ok) return { error: r.error };
  const j = g.k === 2 ? 0 : 1;
  return { hr: r.fit.hr[j], lo: r.fit.hrLower[j], hi: r.fit.hrUpper[j], p: r.fit.p[j] };
}

export interface FeatureContext {
  surv: Surv;
  /** 特徴量の値（cohort の並び。欠損は null）。 */
  values: (number | null)[][];
  names: string[];
  event: number[];
}

export async function analyzeFeature(
  name: string, vals: (number | null)[], cohort: Surv, p: R1Params, rng: Rng,
  tick: (label: string, fraction: number) => Promise<void>, label: string,
): Promise<FeatureResult> {
  const r = emptyResult(name);
  const ix = vals.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  const x = ix.map((i) => vals[i] as number);
  const s = subSurv(cohort, ix);
  r.n = ix.length;
  r.events = s.event.reduce((a, b) => a + b, 0);
  if (r.n < 6 || r.events < 2) { r.error = `欠損を除くと患者 ${r.n} 人・イベント ${r.events} 件で、解析できません`; return r; }
  if (new Set(x).size < 2) { r.error = "値がすべて同じです"; return r; }

  let g: Grouping | { error: string };
  if (p.groupMode === "optimal") {
    const sc = new CutScanner(s, p.minGroupFraction);
    const run = new PermutationRun(sc, x, p.permutations, rng);
    while (!run.finished) {
      run.advance(25);
      await tick(label, run.progress);
    }
    const res = run.result();
    if ("error" in res) { r.error = res.error; return r; }
    r.rawOptimalP = res.rawP;
    r.correctedP = res.correctedP;
    g = groupByCuts(x, [res.observed.cut]);
  } else if (p.groupMode === "median") g = groupByCuts(x, [quantile7([...x].sort((a, b) => a - b), 0.5)]);
  else if (p.groupMode === "value") g = groupByCuts(x, [p.groupValue as number]);
  else { const sx = [...x].sort((a, b) => a - b); g = groupByCuts(x, [quantile7(sx, 1 / 3), quantile7(sx, 2 / 3)]); }
  if ("error" in g) { r.error = g.error; return r; }
  r.cuts = g.cuts;
  for (let k = 0; k < g.k; k++) {
    const members = g.group.map((v, i) => (v === k ? i : -1)).filter((i) => i >= 0);
    const gs = subSurv(s, members);
    r.groupSizes.push(members.length);
    r.groupEvents.push(gs.event.reduce((a, b) => a + b, 0));
    r.medians.push(kaplanMeier(gs).median);
  }
  const lr = logrank(s, g.group, g.k);
  if ("error" in lr) r.error = `log-rank: ${lr.error}`;
  else { r.logrankChi2 = lr.chi2; r.logrankP = lr.p; }
  r.pUsed = p.groupMode === "optimal" ? r.correctedP : r.logrankP;

  const hg = groupHr(s, g);
  if ("error" in hg) r.error = r.error ?? `群の HR: ${hg.error}`;
  else { r.hrGroup = hg.hr; r.hrGroupLower = hg.lo; r.hrGroupUpper = hg.hi; r.hrGroupP = hg.p; }

  const m = x.reduce((a, b) => a + b, 0) / x.length, sdx = sd(x);
  const cf = coxFit(s, x.map((v) => [(v - m) / sdx]));
  if (!cf.ok) r.error = r.error ?? `連続値の Cox: ${cf.error}`;
  else { r.hrPerSd = cf.fit.hr[0]; r.hrPerSdLower = cf.fit.hrLower[0]; r.hrPerSdUpper = cf.fit.hrUpper[0]; r.hrPerSdP = cf.fit.p[0]; }

  if (p.mannWhitney) {
    const a = x.filter((_, i) => s.event[i] === 1), b = x.filter((_, i) => s.event[i] === 0);
    const mw = mannWhitney(a, b);
    if (!("error" in mw)) r.mannWhitneyP = mw.p;
  }
  if (p.rmstTau !== null && g.k === 2) {
    const lo = subSurv(s, g.group.map((v, i) => (v === 0 ? i : -1)).filter((i) => i >= 0));
    const hi = subSurv(s, g.group.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0));
    const rd = rmstDifference(lo, hi, p.rmstTau);
    if (!("error" in rd)) { r.rmstDiff = rd.diff; r.rmstP = rd.p; }
    else r.error = r.error ?? `RMST: ${rd.error}`;
  }
  return r;
}

export function applyMultiplicity(rows: FeatureResult[], mode: Multiplicity): number {
  const q = benjaminiHochberg(rows.map((r) => r.pUsed));
  rows.forEach((r, i) => { r.q = mode === "bh" ? q[i] : null; });
  return rows.filter((r) => r.pUsed !== null).length;
}

export const isSelected = (r: FeatureResult, p: R1Params): boolean =>
  r.error === null && r.pUsed !== null && (p.multiplicity === "bh" ? (r.q as number) < p.alpha : r.pUsed < p.alpha);

export interface CovariateData {
  names: string[];
  /** cohort の並び。数でない・欠損は null。 */
  values: (number | null)[][];
}

export function covariatesOf(d: Dataset, cohort: SurvivalCohort): CovariateData | { error: string } {
  const cols = d.clinical.filter((c) => c.role === "covariate");
  const text = cols.filter((c) => cohort.patients.some((i) => typeof c.values[i] === "string"));
  if (text.length) return { error: `共変量に数でない値があります: ${text.map((c) => c.name).join(", ")}（カテゴリは 0/1 などの数に直してから使ってください）` };
  return { names: cols.map((c) => c.name), values: cols.map((c) => cohort.patients.map((i) => (typeof c.values[i] === "number" ? (c.values[i] as number) : null))) };
}

export interface MultivariateResult {
  selected: string[];
  covariates: string[];
  nUsed: number;
  eventsUsed: number;
  fit: CoxFit | null;
  terms: { name: string; kind: "feature" | "covariate"; hr: number; lower: number; upper: number; p: number }[];
  /** 臨床因子だけのモデル（共変量があるとき）と、特徴量を足した差の検定。 */
  addedValue: { clinicalLoglik: number; chi2: number; df: number; p: number } | null;
  cIndexApparent: number | null;
  corrected: CorrectedC | null;
  notes: string[];
  error: string | null;
}

/**
 * 解析に使う特徴量の選び方（本体と、ブートストラップの中で同じ規則を繰り返す）。
 * ブートストラップの中では並べ替えをしない（重すぎる）。最適しきい値は素の最小 p で選ぶ（簡略化。notes に書く）。
 */
function screenSync(names: string[], values: number[][], ix: number[], s: Surv, p: R1Params): number[] {
  const rows = names.map((nm, j) => analyzeFeatureFast(nm, ix.map((i) => values[j][i]), s, p));
  applyMultiplicity(rows, p.multiplicity);
  return rows.map((r, j) => (isSelected(r, p) ? j : -1)).filter((j) => j >= 0);
}

/** screen 用の軽い版: 群分けと log-rank だけ（Cox・RMST・並べ替えをしない）。 */
function analyzeFeatureFast(name: string, vals: (number | null)[], cohort: Surv, p: R1Params): FeatureResult {
  const r = emptyResult(name);
  const ix = vals.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  const x = ix.map((i) => vals[i] as number);
  const s = subSurv(cohort, ix);
  r.n = ix.length;
  if (r.n < 6 || new Set(x).size < 2 || s.event.reduce((a, b) => a + b, 0) < 2) { r.error = "解析できません"; return r; }
  let g: Grouping | { error: string };
  if (p.groupMode === "optimal") {
    const sc = new CutScanner(s, p.minGroupFraction);
    const best = sc.scan(sc.inTimeOrder(x));
    if (best.empty) { r.error = "候補なし"; return r; }
    r.pUsed = chi2Sf(best.chi2, 1);
    return r;
  }
  const sx = [...x].sort((a, b) => a - b);
  if (p.groupMode === "median") g = groupByCuts(x, [quantile7(sx, 0.5)]);
  else if (p.groupMode === "value") g = groupByCuts(x, [p.groupValue as number]);
  else g = groupByCuts(x, [quantile7(sx, 1 / 3), quantile7(sx, 2 / 3)]);
  if ("error" in g) { r.error = g.error; return r; }
  const lr = logrank(s, g.group, g.k);
  if ("error" in lr) { r.error = lr.error; return r; }
  r.pUsed = lr.p;
  return r;
}

const zScores = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length, s = sd(xs);
  return { m, s, z: xs.map((v) => (v - m) / s) };
};

export async function multivariate(
  ctx: FeatureContext, rows: FeatureResult[], cov: CovariateData, p: R1Params, rng: Rng,
  tick: (label: string, fraction: number) => Promise<void>,
): Promise<MultivariateResult> {
  const out: MultivariateResult = { selected: [], covariates: cov.names, nUsed: 0, eventsUsed: 0, fit: null, terms: [], addedValue: null, cIndexApparent: null, corrected: null, notes: [], error: null };
  const selIdx = rows.map((r, j) => (isSelected(r, p) ? j : -1)).filter((j) => j >= 0);
  out.selected = selIdx.map((j) => rows[j].feature);
  if (!selIdx.length) { out.notes.push("基準を満たす特徴量が無いため、多変量モデルは作りません"); return out; }
  const n = ctx.surv.time.length;
  const featIdx = selIdx.map((j) => ctx.names.indexOf(rows[j].feature));
  const complete = Array.from({ length: n }, (_, i) => i).filter((i) => featIdx.every((j) => ctx.values[j][i] !== null) && cov.values.every((c) => c[i] !== null));
  const s = subSurv(ctx.surv, complete);
  out.nUsed = complete.length;
  out.eventsUsed = s.event.reduce((a, b) => a + b, 0);
  if (complete.length < n) out.notes.push(`欠損のある患者 ${n - complete.length} 人を多変量モデルから除きました（完全症例のみ）`);
  const nVars = selIdx.length + cov.names.length;
  if (out.eventsUsed < 10 * nVars) out.notes.push(`イベント ${out.eventsUsed} 件に対して変数が ${nVars} 個あります（1 変数あたり 10 件未満）。係数は不安定です`);
  const zs = featIdx.map((j) => zScores(complete.map((i) => ctx.values[j][i] as number)));
  const X = complete.map((_, r) => [...zs.map((z) => z.z[r]), ...cov.values.map((c) => c[complete[r]] as number)]);
  const fit = coxFit(s, X);
  if (!fit.ok) { out.error = `多変量 Cox: ${fit.error}`; return out; }
  out.fit = fit.fit;
  const names = [...selIdx.map((j) => rows[j].feature), ...cov.names];
  out.terms = names.map((nm, a) => ({
    name: nm, kind: a < selIdx.length ? "feature" : "covariate", hr: fit.fit.hr[a], lower: fit.fit.hrLower[a], upper: fit.fit.hrUpper[a], p: fit.fit.p[a],
  }));
  out.cIndexApparent = cIndex(s, fit.fit.linearPredictor);
  if (cov.names.length) {
    const clin = coxFit(s, complete.map((i) => cov.values.map((c) => c[i] as number)));
    if (clin.ok) {
      const d = lrDifference(clin.fit, fit.fit, selIdx.length);
      out.addedValue = { clinicalLoglik: clin.fit.loglik, chi2: d.chi2, df: selIdx.length, p: d.p };
    } else out.notes.push(`臨床因子だけのモデルを作れませんでした: ${clin.error}`);
  }
  if (p.bootstrapB > 0) await bootstrapPart(ctx, cov, p, rng, tick, out);
  return out;
}

async function bootstrapPart(
  ctx: FeatureContext, cov: CovariateData, p: R1Params, rng: Rng, tick: (label: string, fraction: number) => Promise<void>, out: MultivariateResult,
): Promise<void> {
  // 補正 C-index は、最終モデルと同じ患者（選んだ特徴量と共変量が揃った患者）で行う。
  // 選択の繰り返しの候補は、その患者の中で欠損の無い特徴量に限る（選んだ特徴量のリスクを全員で計算できるように）。
  const n = ctx.surv.time.length;
  const finalIdx = out.selected.map((nm) => ctx.names.indexOf(nm));
  const cc = Array.from({ length: n }, (_, i) => i).filter((i) => finalIdx.every((j) => ctx.values[j][i] !== null) && cov.values.every((c) => c[i] !== null));
  const s = subSurv(ctx.surv, cc);
  const events = s.event.reduce((a, b) => a + b, 0);
  if (cc.length < 20 || events < 5) {
    out.notes.push(`補正 C-index は、モデルに使った患者が 20 人未満またはイベント 5 件未満のため計算しません（${cc.length} 人・${events} 件）`);
    return;
  }
  const candidates = ctx.names.map((_, j) => j).filter((j) => cc.every((i) => ctx.values[j][i] !== null));
  const names = candidates.map((j) => ctx.names[j]);
  if (candidates.length < ctx.names.length) out.notes.push(`補正 C-index の選択の繰り返しでは、モデルに使った患者で欠損の無い特徴量 ${candidates.length} / ${ctx.names.length} 個を候補にしました`);
  const vals = candidates.map((j) => cc.map((i) => ctx.values[j][i] as number));
  const covs = cov.values.map((col) => cc.map((i) => col[i] as number));
  const res = await bootstrapCorrectedC(s, makeFit(names, vals, covs, s, p), p.bootstrapB, rng, (d, of) => tick("bootstrap", d / of));
  if ("error" in res) { out.notes.push(`補正 C-index: ${res.error}`); return; }
  out.corrected = res;
  if (res.failed) out.notes.push(`ブートストラップ ${res.failed} 回はモデルを作れず除きました`);
  if (p.groupMode === "optimal") out.notes.push("補正 C-index の選択の繰り返しでは、最適しきい値の補正 p の代わりに素の最小 p を使っています（並べ替えは重すぎるため）");
}

/** 選択 → 多変量 Cox を 1 つの手順として当てはめ、元の患者 i のリスクを返す関数を作る。 */
function makeFit(names: string[], vals: number[][], covs: number[][], s: Surv, p: R1Params): FitFn {
  return (idx) => {
    const sub = subSurv(s, idx);
    const sel = screenSync(names, vals, idx, sub, p);
    const z = sel.map((j) => {
      const col = idx.map((i) => vals[j][i]);
      const m = col.reduce((a, b) => a + b, 0) / col.length, sdv = sd(col);
      return sdv > 0 ? { j, m, sd: sdv } : null;
    }).filter((x): x is { j: number; m: number; sd: number } => x !== null);
    const row = (i: number) => [...z.map((f) => (vals[f.j][i] - f.m) / f.sd), ...covs.map((c) => c[i])];
    if (!z.length && !covs.length) return () => 0;
    const r = coxFit(sub, idx.map(row));
    if (!r.ok) return null;
    const beta = r.fit.beta;
    return (i) => row(i).reduce((acc, v, a) => acc + v * beta[a], 0);
  };
}

export interface KmGroup {
  label: string;
  n: number;
  events: number;
  km: ReturnType<typeof kaplanMeier>;
  surv: Surv;
}

/** 1 つの特徴量の群別 KM（画面の図と点列の CSV に使う）。群分けは解析に使ったしきい値のまま。 */
export function kmForFeature(ctx: FeatureContext, row: FeatureResult): KmGroup[] | null {
  if (row.error !== null || !row.cuts.length) return null;
  const j = ctx.names.indexOf(row.feature);
  if (j < 0) return null;
  const ix = ctx.values[j].map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  const g = groupByCuts(ix.map((i) => ctx.values[j][i] as number), row.cuts);
  if ("error" in g) return null;
  const s = subSurv(ctx.surv, ix);
  const fmt = (v: number) => String(Number(v.toPrecision(4)));
  const labels = g.k === 2
    ? [`≤ ${fmt(row.cuts[0])}`, `> ${fmt(row.cuts[0])}`]
    : [`≤ ${fmt(row.cuts[0])}`, `${fmt(row.cuts[0])} – ${fmt(row.cuts[1])}`, `> ${fmt(row.cuts[1])}`];
  return labels.map((label, k) => {
    const members = g.group.map((v, i) => (v === k ? i : -1)).filter((i) => i >= 0);
    const gs = subSurv(s, members);
    return { label, n: members.length, events: gs.event.reduce((a, b) => a + b, 0), km: kaplanMeier(gs), surv: gs };
  });
}
