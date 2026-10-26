/**
 * レシピ R2「判別」の計算本体。結果ラベル（2 値）に対して、特徴量ごとに ROC・AUC・しきい値と感度・特異度、
 * 基準を満たした上位 k 個のロジスティック回帰での組み合わせ（LOO で検証）、較正・決定曲線、決定木を作る。
 * 画面・recipes.ts に依存しない。
 */
import type { Dataset } from "./dataset";
import type { ParamValue } from "./recipe";
import type { Rng } from "./rng";
import { calibrationBins, calibrationSlope, type CalibrationBin } from "./discrim/calibration";
import { decisionCurve, defaultThresholds, type DcaPoint } from "./discrim/dca";
import { looTopK } from "./discrim/loo";
import { logisticFit } from "./discrim/logistic";
import { aucCi, bestCut, bestDirection, confusion, performance, rocPoints, type Criterion, type Direction, type Performance, type RocPoint } from "./discrim/roc";
import { buildTree, monteCarloCi, treePerformance, type MonteCarloCi, type TreeNode } from "./discrim/tree";
import { normalQuantile } from "./survival/dist";
import { mannWhitney } from "./survival/mannwhitney";
import { benjaminiHochberg } from "./survival/multiplicity";
import type { Multiplicity } from "./r1";

export interface R2Params {
  positiveLabel: string;
  criterion: Criterion;
  multiplicity: Multiplicity;
  alpha: number;
  topK: number;
  tree: boolean;
  treeStopRate: number;
  treeMinNode: number;
  treeMaxDepth: number;
  mcIterations: number;
  calibrationBins: number;
}

export const R2_DEFAULTS: Record<string, ParamValue> = {
  positiveLabel: null,
  criterion: "accuracy",
  multiplicity: "bh",
  alpha: 0.05,
  topK: 2,
  tree: false,
  treeStopRate: 0.15,
  treeMinNode: 5,
  treeMaxDepth: 6,
  mcIterations: 1000,
  calibrationBins: 10,
};

export function readR2Params(p: Record<string, ParamValue>): R2Params | { error: string } {
  const num = (k: string, lo: number, hi: number, int = false): number | { error: string } => {
    const v = p[k];
    if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi || (int && !Number.isInteger(v))) return { error: `設定「${k}」は ${lo} 以上 ${hi} 以下の${int ? "整数" : "数"}にしてください` };
    return v;
  };
  if (typeof p.positiveLabel !== "string" || p.positiveLabel === "") return { error: "陽性とするラベルを選んでください（既定は決めていません）" };
  if (p.criterion !== "accuracy" && p.criterion !== "youden") return { error: "しきい値の基準は accuracy か youden です" };
  if (p.multiplicity !== "bh" && p.multiplicity !== "none") return { error: "補正は bh か none です" };
  const vals = [num("alpha", 1e-6, 0.5), num("topK", 2, 20, true), num("treeStopRate", 0, 0.5), num("treeMinNode", 2, 1000, true), num("treeMaxDepth", 1, 20, true), num("mcIterations", 100, 100000, true), num("calibrationBins", 2, 50, true)];
  for (const v of vals) if (typeof v === "object") return v;
  const [alpha, topK, stop, minNode, maxDepth, mc, bins] = vals as number[];
  return { positiveLabel: p.positiveLabel, criterion: p.criterion, multiplicity: p.multiplicity, alpha, topK, tree: p.tree === true, treeStopRate: stop, treeMinNode: minNode, treeMaxDepth: maxDepth, mcIterations: mc, calibrationBins: bins };
}

export interface BinaryCohort {
  /** Dataset の並びでの患者番号。 */
  patients: number[];
  /** 1 = 陽性ラベル。 */
  y: number[];
  /** [陰性ラベル, 陽性ラベル]。 */
  labels: [string, string];
}

/** 結果ラベルの列から 2 値の群を作る。ラベルが 2 種類でない・陽性が選ばれていない・片方が少なすぎるときは理由を返す。 */
export function binaryCohort(d: Dataset, positive: string): BinaryCohort | { error: string } {
  const oc = d.clinical.find((c) => c.role === "outcome");
  if (!oc) return { error: "結果ラベルの列を割り当ててください" };
  const patients: number[] = [], raw: string[] = [];
  d.ids.forEach((_, i) => { const v = oc.values[i]; if (v !== null && v !== "") { patients.push(i); raw.push(String(v)); } });
  const distinct = [...new Set(raw)].sort();
  if (distinct.length !== 2) return { error: `結果ラベルは 2 種類にしてください（見つかったのは ${distinct.length} 種類: ${distinct.slice(0, 6).join(", ")}）` };
  if (!distinct.includes(positive)) return { error: `陽性ラベル「${positive}」が結果ラベルの値（${distinct.join(", ")}）にありません` };
  const negative = distinct.find((x) => x !== positive)!;
  const y: number[] = raw.map((v) => (v === positive ? 1 : 0));
  const pos = y.reduce((a, b) => a + b, 0);
  if (pos < 5 || y.length - pos < 5) return { error: `各群 5 人以上が要ります（陽性 ${pos} 人・陰性 ${y.length - pos} 人）` };
  return { patients, y, labels: [negative, positive] };
}

export interface FeatureRow {
  feature: string;
  n: number;
  positives: number;
  direction: Direction | null;
  auc: number | null;
  aucLower: number | null;
  aucUpper: number | null;
  mwP: number | null;
  q: number | null;
  cut: number | null;
  perf: Performance | null;
  error: string | null;
}

export interface DiscrimContext {
  y: number[];
  names: string[];
  /** cohort の並び。欠損は null。 */
  values: (number | null)[][];
  labels: [string, string];
}

const emptyRow = (feature: string): FeatureRow => ({ feature, n: 0, positives: 0, direction: null, auc: null, aucLower: null, aucUpper: null, mwP: null, q: null, cut: null, perf: null, error: null });

export function analyzeFeatures(ctx: DiscrimContext, p: R2Params): { rows: FeatureRow[]; comparisons: number } {
  const rows = ctx.names.map((name, j) => {
    const r = emptyRow(name);
    const ix = ctx.values[j].map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
    const score = ix.map((i) => ctx.values[j][i] as number), label = ix.map((i) => ctx.y[i]);
    r.n = ix.length;
    r.positives = label.reduce((a, b) => a + b, 0);
    if (r.positives < 5 || r.n - r.positives < 5) { r.error = `欠損を除くと陽性 ${r.positives} 人・陰性 ${r.n - r.positives} 人で、解析できません`; return r; }
    if (new Set(score).size < 2) { r.error = "値がすべて同じです"; return r; }
    const dir = bestDirection({ score, label });
    r.direction = dir;
    const ci = aucCi({ score, label }, dir);
    if (ci) { r.auc = ci.auc; r.aucLower = ci.lower; r.aucUpper = ci.upper; }
    const mw = mannWhitney(score.filter((_, i) => label[i] === 1), score.filter((_, i) => label[i] === 0));
    if (!("error" in mw)) r.mwP = mw.p;
    const cut = bestCut({ score, label }, dir, p.criterion);
    if (cut) { r.cut = cut.cut; r.perf = performance(cut.confusion); }
    return r;
  });
  const q = benjaminiHochberg(rows.map((r) => r.mwP));
  rows.forEach((r, i) => { r.q = p.multiplicity === "bh" ? q[i] : null; });
  return { rows, comparisons: rows.filter((r) => r.mwP !== null).length };
}

export const meetsCriterion = (r: FeatureRow, p: R2Params): boolean =>
  r.error === null && r.mwP !== null && (p.multiplicity === "bh" ? (r.q as number) < p.alpha : r.mwP < p.alpha);

export interface ComboTerm {
  name: string;
  /** 標準化した特徴量 1 SD あたりのオッズ比。 */
  oddsRatio: number;
  lower: number;
  upper: number;
  p: number;
}

export interface ComboResult {
  features: string[];
  nUsed: number;
  positives: number;
  terms: ComboTerm[];
  apparent: { auc: number; lower: number; upper: number } | null;
  loo: { auc: number; lower: number; upper: number } | null;
  looCohort: number;
  looFailed: number;
  candidates: number;
  selectionCounts: { feature: string; count: number }[];
  rocApparent: RocPoint[];
  rocLoo: RocPoint[];
  calibration: CalibrationBin[];
  calibrationFit: { intercept: number; slope: number } | null;
  dca: DcaPoint[];
  /** LOO の予測確率 0.5 での判定。 */
  looPerformance: Performance | null;
  notes: string[];
  error: string | null;
}

function z(xs: number[]) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length, s = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
  return { m, s, z: xs.map((v) => (v - m) / s) };
}

export async function combination(ctx: DiscrimContext, rows: FeatureRow[], p: R2Params, tick: (label: string, fraction: number) => Promise<void>): Promise<ComboResult> {
  const out: ComboResult = {
    features: [], nUsed: 0, positives: 0, terms: [], apparent: null, loo: null, looCohort: 0, looFailed: 0, candidates: 0, selectionCounts: [],
    rocApparent: [], rocLoo: [], calibration: [], calibrationFit: null, dca: [], looPerformance: null, notes: [], error: null,
  };
  const meets = rows.map((r, j) => ({ r, j })).filter((x) => meetsCriterion(x.r, p)).sort((a, b) => (a.r.mwP as number) - (b.r.mwP as number) || a.j - b.j);
  if (meets.length < 2) { out.notes.push(`基準を満たした特徴量が ${meets.length} 個のため、組み合わせは作りません（2 個以上要ります）`); return out; }
  const chosen = meets.slice(0, p.topK);
  out.features = chosen.map((x) => x.r.feature);
  const n = ctx.y.length;
  const cc = Array.from({ length: n }, (_, i) => i).filter((i) => chosen.every((x) => ctx.values[x.j][i] !== null));
  out.nUsed = cc.length;
  const y = cc.map((i) => ctx.y[i]);
  out.positives = y.reduce((a, b) => a + b, 0);
  if (cc.length < n) out.notes.push(`欠損のある患者 ${n - cc.length} 人を組み合わせから除きました（完全症例のみ）`);
  if (Math.min(out.positives, cc.length - out.positives) < 10 * chosen.length) out.notes.push(`少ない方の群が ${Math.min(out.positives, cc.length - out.positives)} 人で、変数 ${chosen.length} 個に対して 1 変数あたり 10 人未満です。係数は不安定です`);
  const zs = chosen.map((x) => z(cc.map((i) => ctx.values[x.j][i] as number)));
  const Z = cc.map((_, r) => zs.map((s) => s.z[r]));
  const fit = logisticFit(Z, y);
  if (!fit.ok) { out.error = `ロジスティック回帰: ${fit.error}`; return out; }
  const zc = normalQuantile(0.975);
  out.terms = chosen.map((x, a) => ({ name: x.r.feature, oddsRatio: Math.exp(fit.fit.beta[a + 1]), lower: Math.exp(fit.fit.beta[a + 1] - zc * fit.fit.se[a + 1]), upper: Math.exp(fit.fit.beta[a + 1] + zc * fit.fit.se[a + 1]), p: fit.fit.p[a + 1] }));
  const apparentPred = Z.map((r) => fit.fit.predict(r));
  const ai = aucCi({ score: apparentPred, label: y });
  if (ai) out.apparent = { auc: ai.auc, lower: ai.lower, upper: ai.upper };
  out.rocApparent = rocPoints({ score: apparentPred, label: y });

  // LOO: 選択も各折の内側。候補は、完全症例の中で欠損の無い特徴量
  const candIdx = ctx.names.map((_, j) => j).filter((j) => cc.every((i) => ctx.values[j][i] !== null));
  out.candidates = candIdx.length;
  if (candIdx.length < ctx.names.length) out.notes.push(`LOO の選択の候補は、組み合わせに使った患者で欠損の無い特徴量 ${candIdx.length} / ${ctx.names.length} 個です`);
  await tick("LOO", 0);
  const Xc = cc.map((i) => candIdx.map((j) => ctx.values[j][i] as number));
  const loo = looTopK(Xc, y, { k: p.topK, alpha: p.multiplicity === "bh" ? p.alpha : null, selectInside: true });
  await tick("LOO", 1);
  out.looFailed = loo.failed;
  out.selectionCounts = loo.selectionCounts.map((count, a) => ({ feature: ctx.names[candIdx[a]], count })).filter((x) => x.count > 0).sort((a, b) => b.count - a.count);
  const ok = loo.pred.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  if (loo.failed) out.notes.push(`LOO のうち ${loo.failed} 折はモデルを作れず、その患者は LOO の評価から除きました`);
  const lp = ok.map((i) => loo.pred[i] as number), ly = ok.map((i) => y[i]);
  out.looCohort = ok.length;
  const li = aucCi({ score: lp, label: ly });
  if (li) out.loo = { auc: li.auc, lower: li.lower, upper: li.upper };
  out.notes.push("LOO の予測は互いに独立でないため、LOO の AUC の信頼区間は目安です");
  out.rocLoo = rocPoints({ score: lp, label: ly });
  out.calibration = calibrationBins(lp, ly, p.calibrationBins);
  const cs = calibrationSlope(lp, ly);
  if (!("error" in cs)) out.calibrationFit = cs; else out.notes.push(`較正の傾き・切片は計算しませんでした: ${cs.error}`);
  out.dca = decisionCurve(lp, ly, defaultThresholds());
  out.looPerformance = performance(confusion(ly, lp.map((v) => (v >= 0.5 ? 1 : 0))));
  return out;
}

export interface TreeResult {
  root: TreeNode;
  candidates: string[];
  nUsed: number;
  perf: Performance;
  mc: MonteCarloCi;
  notes: string[];
}

export function decisionTree(ctx: DiscrimContext, rows: FeatureRow[], p: R2Params, rng: Rng): TreeResult | { note: string } {
  const cand = rows.map((r, j) => ({ r, j })).filter((x) => meetsCriterion(x.r, p));
  if (!cand.length) return { note: "基準を満たした特徴量が無いため、決定木は作りません" };
  const n = ctx.y.length;
  const cc = Array.from({ length: n }, (_, i) => i).filter((i) => cand.every((x) => ctx.values[x.j][i] !== null));
  const y = cc.map((i) => ctx.y[i]);
  const pos = y.reduce((a, b) => a + b, 0);
  if (pos < 5 || y.length - pos < 5) return { note: `完全症例が少なすぎます（陽性 ${pos} 人・陰性 ${y.length - pos} 人）` };
  const X = cc.map((i) => cand.map((x) => ctx.values[x.j][i] as number));
  const root = buildTree(X, y, { stopRate: p.treeStopRate, minNode: p.treeMinNode, maxDepth: p.treeMaxDepth, candidates: cand.map((_, a) => a) });
  const notes: string[] = [];
  if (cc.length < n) notes.push(`欠損のある患者 ${n - cc.length} 人を決定木から除きました（完全症例のみ）`);
  notes.push("決定木の性能は、木を作った同じ患者での値です（検証していません）。LOO や独立の検証データでの値ではありません");
  return { root: remap(root, cand.map((x) => x.j)), candidates: cand.map((x) => x.r.feature), nUsed: cc.length, perf: treePerformance(root, X, y), mc: monteCarloCi(root, X, y, p.mcIterations, rng), notes };
}

/** 木の中の特徴量番号を、候補の並びから特徴量全体の番号に直す（表示で名前を引くため）。 */
function remap(t: TreeNode, map: number[]): TreeNode {
  return { ...t, split: t.split ? { ...t.split, feature: map[t.split.feature] } : null, positive: t.positive ? remap(t.positive, map) : null, negative: t.negative ? remap(t.negative, map) : null };
}

