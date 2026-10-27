/** R5 の結果を CSV にする。計算できなかった値は空欄（0 にしない）。 */
import { toCsv } from "./csv";
import type { R5State } from "./skills";

/** 選んだ特徴量の係数（標準化した尺度、SD あたり）と、標準化に使った平均・SD、C-index。 */
export function r5ModelCsv(st: R5State): string {
  const m = st.model;
  if (!m) return toCsv(["feature", "beta_per_sd", "hr_per_sd", "hr_lower", "hr_upper", "p", "mean", "sd"], []);
  const r = m.result;
  const rows: (string | number | null)[][] = m.terms.map((x, q) => [x.name, x.beta, x.hr, x.lower, x.upper, x.p, r.means[q], r.sds[q]]);
  rows.push(["c_index_apparent", m.cApparent, null, null, null, null, null, null]);
  rows.push(["c_index_corrected", m.corrected?.corrected ?? null, null, null, null, null, null, null]);
  rows.push(["optimism", m.corrected?.optimism ?? null, null, null, null, null, null, null]);
  return toCsv(["feature", "beta_per_sd", "hr_per_sd", "hr_lower", "hr_upper", "p", "mean", "sd"], rows);
}

/** 相関で除いた特徴量、欠損で除いた特徴量、後退選択の経過を順に。 */
export function r5StepsCsv(st: R5State): string {
  const m = st.model;
  if (!m) return toCsv(["stage", "feature", "action", "p_or_reason"], []);
  return toCsv(["stage", "feature", "action", "p_or_reason"], [
    ...m.droppedMissing.map((f) => ["missing", f, "removed", "欠損がある"]),
    ...m.result.corrRemoved.map((x) => ["correlation", x.feature, "removed", x.reason]),
    ...m.result.steps.map((s) => ["backward", st.names[s.col], s.action, s.p]),
  ]);
}

/** 絞り込み: ブースティングなら係数と重要度、LASSO なら λ ごとの交差検証の偏差と、選んだ λ での係数。 */
export function r5ScreenCsv(st: R5State): string {
  const r = st.model?.result;
  if (r?.boost) {
    const b = r.boost;
    return toCsv(["feature", "beta_per_sd", "importance", "selected_count"], b.std.cols.map((c, a) => [st.names[c], b.beta[a], b.importance[a], b.selected.filter((x) => x === a).length]));
  }
  if (r?.lasso) {
    const l = r.lasso;
    const head = ["lambda", "cv_deviance", "cv_sd", "nonzero", "chosen", ...l.std.cols.map((c) => st.names[c])];
    return toCsv(head, l.lambdas.map((lam, a) => {
      const pt = l.path.points[a];
      return [lam, l.cvm[a], l.cvsd[a], pt ? pt.coef.beta.filter((x) => x !== 0).length : null, a === l.chosen ? 1 : 0, ...(pt ? pt.coef.beta : l.std.cols.map(() => null))];
    }));
  }
  return toCsv(["feature"], []);
}

/** イベントごとの Schoenfeld 残差（スケールしたもの）と時間の変換。 */
export function r5SchoenfeldCsv(st: R5State): string {
  const m = st.model, s = m?.schoenfeld;
  if (!m || !s) return toCsv(["id", "time", "km_transform"], []);
  const sel = m.result.selected.map((j) => st.names[j]);
  return toCsv(["id", "time", "km_transform", ...sel.map((n) => `resid_${n}`), ...sel.map((n) => `scaled_${n}`)],
    s.rows.map((row, i) => [st.ids[row], s.times[i], s.transformed[i], ...s.resid[i], ...s.scaled[i]]));
}
