/** P4（パイプライン）の結果を CSV にする。評価できなかった値は空欄（0 や 0.5 にしない）。 */
import { toCsv } from "./csv";
import type { P4State } from "./skills";

/** 外側の分割ごとの結果。 */
export function p4FoldsCsv(st: P4State): string {
  const cv = st.cv;
  if (!cv) return toCsv(["repeat", "fold", "n_train", "n_test", "k", "features", "score", "error"], []);
  return toCsv(["repeat", "fold", "n_train", "n_test", "k", "features", "score", "error"],
    cv.folds.map((f) => [f.repeat + 1, f.fold + 1, f.nTrain, f.nTest, f.k, f.cols.map((j) => st.names[j]).join(" + "), f.score, f.error]));
}

/** 開発集合の症例ごとの、交差検証での予測（反復ごとの列）。 */
export function p4PredictionsCsv(st: P4State): string {
  const cv = st.cv, dev = st.qc?.dev ?? [];
  const c = st.cohort;
  const outcomeHead = c.task.kind === "binary" ? ["outcome"] : ["time", "event"];
  const reps = cv ? cv.pred.map((_, r) => `pred_repeat_${r + 1}`) : [];
  return toCsv(["id", ...outcomeHead, ...reps, ...(cv ? cv.outerAssign.map((_, r) => `fold_repeat_${r + 1}`) : [])], dev.map((i, p) => {
    const out = c.task.kind === "binary" ? [c.labels![c.task.y[i]]] : [c.task.surv.time[i], c.task.surv.event[i]];
    return [c.ids[i], ...out, ...(cv ? cv.pred.map((pr) => pr[p]) : []), ...(cv ? cv.outerAssign.map((a) => a[p] + 1) : [])];
  }));
}

/** 特徴量ごとの、外側の分割で選ばれた回数。 */
export function p4SelectionCsv(st: P4State): string {
  const cv = st.cv;
  return toCsv(["feature", "selected", "of"], cv ? st.names.map((n, j) => [n, cv.selectionCounts[j], cv.folds.length]) : []);
}

/** 品質管理で除いた特徴量と、欠損で除いた症例。 */
export function p4QcCsv(st: P4State): string {
  const q = st.qc;
  if (!q) return toCsv(["kind", "name", "reason"], []);
  return toCsv(["kind", "name", "reason"], [
    ...q.removed.map((r) => ["feature", r.feature, r.reason]),
    ...q.droppedDev.map((id) => ["case", id, "欠損のある開発集合の症例"]),
  ]);
}
