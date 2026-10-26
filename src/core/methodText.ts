/**
 * 「方法」の文章。実行の記録（RunRecord）だけから作る（画面の状態を見ない）ので、同じ記録なら同じ文になる。
 * 記録に無いこと（出典・手法の根拠）は書かない。
 */
import type { RunRecord } from "./engine";
import type { Locale, ParamValue, Recipe } from "./recipe";

const short = (h: string | null, locale: Locale) => (h === null ? (locale === "ja" ? "未取得" : "not available") : h.slice(0, 12));
const show = (v: ParamValue) => (Array.isArray(v) ? `[${v.join(", ")}]` : v === null ? "null" : String(v));

export function methodText(record: RunRecord, recipe: Recipe, locale: Locale): string {
  const join = record.steps.find((s) => s.id === "join");
  const delta = record.steps.find((s) => s.id === "delta" && s.status === "ok");
  const n = (k: string) => join?.summary[k] ?? "?";
  const files = record.inputs.map((i) => `${i.name}（${i.rows}${locale === "ja" ? " 行" : " rows"}、SHA-256 ${short(i.sha256, locale)}）`).join(locale === "ja" ? "、" : "; ");
  const changed = record.changedFromDefault.map((k) => `${k}=${show(record.params[k])}`);
  const lines: string[] = [];
  if (locale === "ja") {
    lines.push(`レシピ「${recipe.title.ja}」（${recipe.id}）を、GRAPHY-Next プラグイン research-stats ${record.pluginVersion} で実行した。`);
    lines.push(`入力ファイル: ${files}。`);
    if (delta) lines.push(`2 時点の特徴量から変化率を求めた（順序: ${show(record.params.deltaOrder)}、病変の不一致 ${delta.summary.unmatchedLesions} 個、変化率を作れなかった値 ${delta.summary.impossible} 個）。`);
    if (join) {
      lines.push(`特徴量の表 ${n("featureRows")} 行（${n("patientsInFeatures")} 患者）と臨床の表（${n("patientsInClinical")} 患者）を患者 ID で結合し、${n("joined")} 患者を解析の対象とした。特徴量の表だけにいた患者は ${n("featureOnly")}、臨床の表だけにいた患者は ${n("clinicalOnly")}。`);
      if (join.summary.survivalUsable !== null && join.summary.survivalUsable !== undefined) lines.push(`時間とイベントが揃った患者は ${join.summary.survivalUsable} 人であった。`);
      lines.push("欠損は 0 で補わず、欠損のまま扱った（複数病変の平均は欠損を除いて求めた）。");
    }
    lines.push(`乱数シード: ${record.seed}。${changed.length ? `既定値から変更した設定: ${changed.join("、")}。` : "設定は既定値のままであった。"}`);
    lines.push("研究用であり、診断・治療の判断に用いない。");
  } else {
    lines.push(`The recipe "${recipe.title.en}" (${recipe.id}) was run with the GRAPHY-Next plugin research-stats ${record.pluginVersion}.`);
    lines.push(`Input files: ${files}.`);
    if (delta) lines.push(`Percent change was computed from the two time points (order: ${show(record.params.deltaOrder)}; unmatched lesions ${delta.summary.unmatchedLesions}; values that could not be computed ${delta.summary.impossible}).`);
    if (join) {
      lines.push(`The feature table (${n("featureRows")} rows, ${n("patientsInFeatures")} patients) and the clinical table (${n("patientsInClinical")} patients) were joined by patient ID; ${n("joined")} patients were analysed. Patients only in the feature table: ${n("featureOnly")}; only in the clinical table: ${n("clinicalOnly")}.`);
      if (join.summary.survivalUsable !== null && join.summary.survivalUsable !== undefined) lines.push(`${join.summary.survivalUsable} patients had both time and event.`);
      lines.push("Missing values were not filled with 0 and were kept missing (the mean over several lesions ignores missing values).");
    }
    lines.push(`Random seed: ${record.seed}. ${changed.length ? `Settings changed from the defaults: ${changed.join("; ")}.` : "All settings were the defaults."}`);
    lines.push("For research use only; not for diagnostic or treatment decisions.");
  }
  return `${lines.join("\n")}\n`;
}
