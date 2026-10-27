/**
 * スキル R5「少数の特徴で予後モデル」の計算本体。
 * 相関で冗長を除く → Cox ブースティングか LASSO-Cox で候補を絞る → 後退選択 → 最終の Cox →
 * Harrell のブートストラップで楽観を補正した C-index（手順全体を毎回やり直す）→ Schoenfeld 残差。
 * 特徴量は、生存の対象の患者で欠損の無いものだけを使う（欠損のある特徴量は理由つきで除く）。
 */
import type { ParamValue } from "./skill";
import type { Rng } from "./rng";
import type { Surv } from "./survival/types";
import { coxFit, type CoxFit } from "./survival/cox";
import { bootstrapCorrectedC, cIndex, type CorrectedC, type FitFn } from "./survival/cindex";
import { qcFeatures, type QcRemoval } from "./pipeline/qc";
import { coxBoost, type BoostResult } from "./model/coxBoost";
import { lassoCv, fitStd, zRow, type LambdaRule, type LassoCvResult } from "./model/lasso";
import { backwardSelect, type BackwardStep } from "./model/backward";
import { schoenfeld, type SchoenfeldResult } from "./model/schoenfeld";

export type ScreenMethod = "boost" | "lasso";

export interface R5Params {
  /** |Spearman| がこれ以上の組の片方を除く。 */
  corrThreshold: number;
  screen: ScreenMethod;
  boostIterations: number;
  boostNu: number;
  lassoFolds: number;
  lassoRule: LambdaRule;
  lassoNLambda: number;
  pIn: number;
  pOut: number;
  /** 残す変数の最大の数（論文の「5 未満」→ 4）。 */
  maxVars: number;
  bootstrapB: number;
}

/** 論文に値があるものだけ既定値を置く。相関のしきい値・ν・λ の選び方は利用者が入れる（null のままでは実行しない）。 */
export const R5_DEFAULTS: Record<string, ParamValue> = {
  r5Corr: null,
  r5Screen: "boost",
  r5BoostIterations: 100,
  r5BoostNu: null,
  r5LassoFolds: 10,
  r5LassoRule: null,
  r5LassoNLambda: 100,
  r5PIn: 0.1,
  r5POut: 0.15,
  r5MaxVars: 4,
  r5BootstrapB: 2000,
};

export function readR5Params(p: Record<string, ParamValue>): R5Params | { error: string } {
  const num = (k: string, lo: number, hi: number, label: string, int = false): number | { error: string } => {
    const v = p[k];
    if (v === null || v === undefined) return { error: `「${label}」を入れてください` };
    if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi || (int && !Number.isInteger(v))) return { error: `「${label}」は ${lo} 以上 ${hi} 以下の${int ? "整数" : "数"}にしてください` };
    return v;
  };
  const screen = p.r5Screen;
  if (screen !== "boost" && screen !== "lasso") return { error: "候補の絞り方は boost か lasso です" };
  const corr = num("r5Corr", 0.5, 1, "相関のしきい値");
  if (typeof corr === "object") return corr;
  const pIn = num("r5PIn", 1e-6, 0.5, "入る p"), pOut = num("r5POut", 1e-6, 0.5, "出る p"), maxVars = num("r5MaxVars", 1, 50, "残す変数の最大の数", true), B = num("r5BootstrapB", 0, 100000, "ブートストラップの回数", true);
  for (const v of [pIn, pOut, maxVars, B]) if (typeof v === "object") return v;
  if ((pIn as number) >= (pOut as number)) return { error: "入る p は出る p より小さくしてください" };
  const base = { corrThreshold: corr, screen: screen as ScreenMethod, pIn: pIn as number, pOut: pOut as number, maxVars: maxVars as number, bootstrapB: B as number, boostIterations: 0, boostNu: 0, lassoFolds: 0, lassoRule: "min" as LambdaRule, lassoNLambda: 0 };
  if (screen === "boost") {
    const it = num("r5BoostIterations", 1, 100000, "ブースティングの反復回数", true), nu = num("r5BoostNu", 1e-6, 1, "学習率 ν");
    for (const v of [it, nu]) if (typeof v === "object") return v;
    return { ...base, boostIterations: it as number, boostNu: nu as number };
  }
  const folds = num("r5LassoFolds", 2, 1000, "LASSO の交差検証の分割数", true), nl = num("r5LassoNLambda", 2, 1000, "λ の点の数", true);
  for (const v of [folds, nl]) if (typeof v === "object") return v;
  const rule = p.r5LassoRule;
  if (rule !== "min" && rule !== "1se") return { error: "λ の選び方（最小か 1SE）を選んでください" };
  return { ...base, lassoFolds: folds as number, lassoNLambda: nl as number, lassoRule: rule };
}

/** 一回分の手順（相関 → 絞り込み → 後退選択 → Cox）の結果。 */
export interface ProcedureResult {
  corrRemoved: QcRemoval[];
  /** 相関の除外の後に残った列。 */
  afterCorr: number[];
  boost: BoostResult | null;
  lasso: LassoCvResult | null;
  /** 絞り込みで残った候補（順位の順）。 */
  candidates: number[];
  steps: BackwardStep[];
  /** 最終的に選んだ列。 */
  selected: number[];
  /** 選んだ列で、行 rows の平均・SD で標準化した Cox（Efron）。0 個なら null。 */
  fit: CoxFit | null;
  means: number[];
  sds: number[];
  /** 起きたが止めなかったこと。 */
  notes: string[];
}

/** 最小の λ の目安: 行数が列数より多ければ 1e-4、そうでなければ 1e-2（glmnet の既定と同じ）。 */
export const lassoRatio = (nRows: number, nCols: number) => (nRows > nCols ? 1e-4 : 1e-2);

/** rows（重複あり）で手順全体を行う。X は欠損の無い値。 */
export function procedure(surv: Surv, X: number[][], names: string[], rows: number[], cols: number[], p: R5Params, rng: Rng): ProcedureResult | { error: string } {
  const notes: string[] = [];
  const sub = X.map((row) => cols.map((j) => row[j]));
  const qc = qcFeatures(sub, rows, cols.map((j) => names[j]), { dropConstant: true, correlationThreshold: p.corrThreshold, exclude: null });
  const afterCorr = qc.keep.map((a) => cols[a]);
  const corrRemoved = qc.removed;
  const out: ProcedureResult = { corrRemoved, afterCorr, boost: null, lasso: null, candidates: [], steps: [], selected: [], fit: null, means: [], sds: [], notes };
  if (!afterCorr.length) return { error: "相関の除外の後に特徴量が残りませんでした" };
  if (p.screen === "boost") {
    const b = coxBoost(surv, X, rows, afterCorr, { iterations: p.boostIterations, nu: p.boostNu });
    if ("error" in b) return { error: `ブースティング: ${b.error}` };
    out.boost = b;
    out.candidates = b.ranking;
  } else {
    const l = lassoCv({ kind: "survival", surv }, X, rows, afterCorr, { nLambda: p.lassoNLambda, ratio: lassoRatio(rows.length, afterCorr.length), folds: p.lassoFolds, rule: p.lassoRule }, rng);
    if ("error" in l) return { error: `LASSO-Cox: ${l.error}` };
    out.lasso = l;
    if (l.path.stopped) notes.push(`LASSO の経路: ${l.path.stopped}`);
    out.candidates = l.std.cols.map((c, a) => ({ c, b: Math.abs(l.coef.beta[a]) })).filter((x) => x.b > 0).sort((x, y) => y.b - x.b || x.c - y.c).map((x) => x.c);
  }
  if (!out.candidates.length) { notes.push("絞り込みで候補が残りませんでした（特徴量を使わないモデル）"); return out; }
  const bw = backwardSelect(surv, X, rows, out.candidates, { pIn: p.pIn, pOut: p.pOut, maxVars: p.maxVars });
  if ("error" in bw) return { error: `後退選択: ${bw.error}` };
  out.steps = bw.steps;
  out.selected = bw.cols;
  if (bw.steps.some((s) => s.action === "cycle")) notes.push("後退選択で同じ組み合わせに戻ったため、そこで止めました");
  const caps = bw.steps.filter((s) => s.action === "cap");
  if (caps.length) notes.push(`残す変数の最大の数 ${p.maxVars} を超えたため、p の大きい順に追加で除きました: ${caps.map((s) => names[s.col]).join(", ")}`);
  if (bw.cols.length) {
    const st = fitStd(X, rows, bw.cols);
    if ("error" in st) return st;
    out.fit = bw.fit;
    out.means = st.means;
    out.sds = st.sds;
  }
  return out;
}

/** 手順の結果から、元の行 i のリスク（線形予測子）を返す関数。 */
export function riskOf(r: ProcedureResult, X: number[][]): (i: number) => number {
  if (!r.fit) return () => 0;
  const beta = r.fit.beta;
  return (i) => r.selected.reduce((a, j, q) => a + ((X[i][j] - r.means[q]) / r.sds[q]) * beta[q], 0);
}

/** 試験専用の漏れ: ブートストラップの各回で選択をやり直さず、全データで選んだ列のまま当てはめ直す。 */
export interface R5Leak {
  selectOnce?: boolean;
}

export interface R5Model {
  names: string[];
  /** 使った特徴量（欠損の無いもの）と、欠損で除いたもの。 */
  used: number[];
  droppedMissing: string[];
  result: ProcedureResult;
  cApparent: number | null;
  corrected: CorrectedC | null;
  schoenfeld: SchoenfeldResult | null;
  /** 選んだ列の HR（SD あたり）・CI・p。 */
  terms: { name: string; beta: number; hr: number; lower: number; upper: number; p: number }[];
  notes: string[];
}

export async function r5Model(
  surv: Surv, values: (number | null)[][], names: string[], p: R5Params, rng: Rng,
  tick: (label: string, fraction: number) => Promise<void> = async () => {}, leak: R5Leak = {},
): Promise<R5Model | { error: string }> {
  const n = surv.time.length;
  const used = names.map((_, j) => j).filter((j) => values[j].every((v) => v !== null && Number.isFinite(v)));
  const droppedMissing = names.filter((_, j) => !used.includes(j));
  if (!used.length) return { error: "欠損の無い特徴量がありません" };
  const X = Array.from({ length: n }, (_, i) => names.map((_, j) => (values[j][i] === null ? NaN : (values[j][i] as number))));
  const all = Array.from({ length: n }, (_, i) => i);
  const res = procedure(surv, X, names, all, used, p, rng);
  if ("error" in res) return res;
  const notes = [...res.notes];
  if (droppedMissing.length) notes.push(`欠損のある特徴量 ${droppedMissing.length} 個を除きました: ${droppedMissing.join(", ")}`);
  const events = surv.event.reduce((a, b) => a + b, 0);
  if (res.selected.length && events < 10 * res.selected.length) notes.push(`イベント ${events} 件に対して変数が ${res.selected.length} 個あります（1 変数あたり 10 件未満）。係数は不安定です`);
  const risk = riskOf(res, X);
  const cApparent = res.fit ? cIndex(surv, all.map(risk)) : null;
  let corrected: CorrectedC | null = null;
  if (p.bootstrapB > 0) {
    const fit: FitFn = leak.selectOnce
      ? (idx) => {
        if (!res.selected.length) return () => 0;
        const st = fitStd(X, idx, res.selected);
        if ("error" in st || st.cols.length !== res.selected.length) return null;
        const f = coxFit({ time: idx.map((i) => surv.time[i]), event: idx.map((i) => surv.event[i]) }, idx.map((i) => zRow(st, X[i])));
        if (!f.ok) return null;
        return (i) => zRow(st, X[i]).reduce((a, v, q) => a + v * f.fit.beta[q], 0);
      }
      : (idx) => {
        const r = procedure(surv, X, names, idx, used, p, rng);
        return "error" in r ? null : riskOf(r, X);
      };
    const c = await bootstrapCorrectedC(surv, fit, p.bootstrapB, rng, (d, of) => tick("bootstrap", d / of));
    if ("error" in c) notes.push(`補正 C-index: ${c.error}`);
    else {
      corrected = c;
      if (c.failed) notes.push(`ブートストラップ ${c.failed} 回は手順の途中で失敗したため除きました`);
    }
  }
  let sch: SchoenfeldResult | null = null;
  const terms: R5Model["terms"] = [];
  if (res.fit) {
    const f = res.fit;
    res.selected.forEach((j, q) => terms.push({ name: names[j], beta: f.beta[q], hr: f.hr[q], lower: f.hrLower[q], upper: f.hrUpper[q], p: f.p[q] }));
    const Z = all.map((i) => res.selected.map((j, q) => (X[i][j] - res.means[q]) / res.sds[q]));
    const s = schoenfeld(surv, Z, f.beta, f.cov);
    if ("error" in s) notes.push(`Schoenfeld 残差: ${s.error}`);
    else {
      sch = s;
      const bad = s.tests.map((t, q) => (t.p < 0.05 ? res.selected[q] : -1)).filter((j) => j >= 0);
      if (bad.length) notes.push(`比例ハザードの検定で p < 0.05 の変数があります: ${bad.map((j) => names[j]).join(", ")}。効果が時間で変わる可能性があります（時間で層別した解析はこのスキルでは行いません）`);
    }
  }
  return { names, used, droppedMissing, result: res, cApparent, corrected, schoenfeld: sch, terms, notes };
}
