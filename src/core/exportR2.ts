/** R2 の結果を CSV にする。計算できなかった値は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { ComboResult, FeatureRow } from "./r2";
import type { TreeNode } from "./discrim/tree";
import { treeNodes } from "./discrim/tree";

const pv = (x: { value: number | null; lower: number | null; upper: number | null } | undefined) => [x?.value ?? null, x?.lower ?? null, x?.upper ?? null];

export function r2ResultsCsv(rows: FeatureRow[]): string {
  const header = ["feature", "n", "positives", "direction", "auc", "auc_lower", "auc_upper", "mannwhitney_p", "q_bh", "cut",
    "sensitivity", "sens_lower", "sens_upper", "specificity", "spec_lower", "spec_upper", "ppv", "ppv_lower", "ppv_upper", "npv", "npv_lower", "npv_upper",
    "accuracy", "acc_lower", "acc_upper", "tp", "fp", "tn", "fn", "error"];
  return toCsv(header, rows.map((r) => [r.feature, r.n, r.positives, r.direction, r.auc, r.aucLower, r.aucUpper, r.mwP, r.q, r.cut,
    ...pv(r.perf?.sensitivity), ...pv(r.perf?.specificity), ...pv(r.perf?.ppv), ...pv(r.perf?.npv), ...pv(r.perf?.accuracy),
    r.perf?.confusion.tp ?? null, r.perf?.confusion.fp ?? null, r.perf?.confusion.tn ?? null, r.perf?.confusion.fn ?? null, r.error]));
}

/** ROC・較正・決定曲線の点列（図を作り直せる元の数値）。 */
export function r2CurvesCsv(c: ComboResult): string {
  const out: (string | number | null)[][] = [];
  c.rocApparent.forEach((p) => out.push(["roc_apparent", p.fpr, p.tpr, null, null]));
  c.rocLoo.forEach((p) => out.push(["roc_loo", p.fpr, p.tpr, null, null]));
  c.calibration.forEach((b) => out.push(["calibration_loo", b.meanPredicted, b.observed.value, b.observed.lower, b.observed.upper]));
  c.dca.forEach((d) => { out.push(["dca_model", d.threshold, d.model, null, null]); out.push(["dca_all", d.threshold, d.all, null, null]); out.push(["dca_none", d.threshold, d.none, null, null]); });
  return toCsv(["curve", "x", "y", "y_lower", "y_upper"], out);
}

export function treeCsv(root: TreeNode, names: string[]): string {
  const nodes = treeNodes(root);
  return toCsv(["node", "depth", "n", "positives", "branch_label", "feature", "direction", "cut", "label", "tp", "fp", "tn", "fn"],
    nodes.map((n, i) => [i, n.depth, n.n, n.pos, n.branchLabel, n.split ? names[n.split.feature] : null, n.split?.direction ?? null, n.split?.cut ?? null, n.label,
      n.split?.confusion.tp ?? null, n.split?.confusion.fp ?? null, n.split?.confusion.tn ?? null, n.split?.confusion.fn ?? null]));
}
