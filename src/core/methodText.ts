/**
 * 「方法」の文章。実行の記録（RunRecord）だけから作る（画面の状態を見ない）ので、同じ記録なら同じ文になる。
 * 記録に無いこと（出典・手法の根拠）は書かない。
 */
import type { RunRecord } from "./engine";
import type { Locale, ParamValue, Recipe } from "./recipe";

const short = (h: string | null, locale: Locale) => (h === null ? (locale === "ja" ? "未取得" : "not available") : h.slice(0, 12));
const show = (v: ParamValue) => (Array.isArray(v) ? `[${v.join(", ")}]` : v === null ? "null" : String(v));

const GROUP_JA: Record<string, string> = { median: "中央値", value: "指定した値", tertile: "三分位", optimal: "最適しきい値（log-rank のχ²が最大になる値。探索的）" };
const GROUP_EN: Record<string, string> = { median: "the median", value: "a specified value", tertile: "tertiles", optimal: "an optimal cut-off (maximising the log-rank statistic; exploratory)" };

/** R1（予後）の手順の記録から、解析の方法の文を作る。記録に無い手順は書かない。 */
function r1Lines(record: RunRecord, locale: Locale): string[] {
  const uni = record.steps.find((s) => s.id === "univariate" && s.status === "ok");
  if (!uni) return [];
  const p = record.params, cohort = record.steps.find((s) => s.id === "cohort"), mv = record.steps.find((s) => s.id === "multivariate" && s.status === "ok");
  const gm = String(p.groupMode), bh = p.multiplicity === "bh";
  const L: string[] = [];
  const num = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : "?");
  if (locale === "ja") {
    L.push(`生存解析の対象は ${cohort?.summary.patients} 人（イベント ${cohort?.summary.events} 件）。特徴量ごとに ${GROUP_JA[gm]} で 2 群（三分位は 3 群）に分け、Kaplan–Meier 法と log-rank 検定、Cox 比例ハザードモデル（同時刻は Efron の近似）で比べた。連続値の Cox は標準化（SD あたり）で行った。`);
    if (gm === "optimal") L.push(`最適しきい値は各群が全体の ${num(p.minGroupFraction)} 以上になる候補から選び、探索による p の過小評価を補うため、時間とイベントの組を並べ替える検定（${p.permutations} 回）で補正した p を用いた。`);
    L.push(bh ? `${uni.summary.comparisons} 個の比較を Benjamini–Hochberg 法で補正し、q < ${num(p.alpha)} を有意とした。` : `多重比較の補正をしない（探索的な解析）。p < ${num(p.alpha)} を目安とした。`);
    if (mv) {
      L.push(`基準を満たした ${mv.summary.selected} 個の特徴量${(record.steps.find((s) => s.id === "cohort")?.summary.covariates ?? 0) ? "と臨床因子" : ""}で多変量 Cox モデルを当てはめた（完全症例 ${mv.summary.nUsed} 人、イベント ${mv.summary.eventsUsed} 件）。`);
      if (mv.summary.cIndexApparent !== null) L.push(`C-index は見かけ ${num(mv.summary.cIndexApparent)}${mv.summary.cIndexCorrected !== null ? `、ブートストラップ（${mv.summary.bootstraps} 回、特徴量の選択も毎回やり直す）で楽観を補正して ${num(mv.summary.cIndexCorrected)}` : ""}。`);
      if (mv.summary.addedValueP !== null) L.push(`臨床因子に特徴量を加えた効果は −2 log 尤度の差（尤度比検定）で p = ${num(mv.summary.addedValueP)}。`);
    }
  } else {
    L.push(`${cohort?.summary.patients} patients (${cohort?.summary.events} events) entered the survival analysis. Each feature was split by ${GROUP_EN[gm]} into two groups (three for tertiles) and compared with Kaplan–Meier curves, the log-rank test and Cox proportional hazards models (Efron's approximation for ties); the continuous Cox models used standardised values (per SD).`);
    if (gm === "optimal") L.push(`The cut-off was chosen among candidates leaving at least ${num(p.minGroupFraction)} of the patients in each group, and the p value was corrected for the search by a permutation test (${p.permutations} permutations of the time-event pairs).`);
    L.push(bh ? `${uni.summary.comparisons} comparisons were corrected with the Benjamini–Hochberg procedure (q < ${num(p.alpha)}).` : `No correction for multiple comparisons was applied (exploratory analysis); p < ${num(p.alpha)} was used as a guide.`);
    if (mv) {
      L.push(`A multivariable Cox model was fitted with the ${mv.summary.selected} selected feature(s)${(cohort?.summary.covariates ?? 0) ? " and the clinical covariates" : ""} (complete cases ${mv.summary.nUsed}, ${mv.summary.eventsUsed} events).`);
      if (mv.summary.cIndexApparent !== null) L.push(`The apparent C-index was ${num(mv.summary.cIndexApparent)}${mv.summary.cIndexCorrected !== null ? `; the optimism-corrected C-index (bootstrap, ${mv.summary.bootstraps} resamples, with the feature selection repeated in each) was ${num(mv.summary.cIndexCorrected)}` : ""}.`);
      if (mv.summary.addedValueP !== null) L.push(`The added value of the features over the clinical covariates was tested by the difference in −2 log likelihood (p = ${num(mv.summary.addedValueP)}).`);
    }
  }
  return L;
}

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
    lines.push(...r1Lines(record, "ja"));
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
    lines.push(...r1Lines(record, "en"));
    lines.push(`Random seed: ${record.seed}. ${changed.length ? `Settings changed from the defaults: ${changed.join("; ")}.` : "All settings were the defaults."}`);
    lines.push("For research use only; not for diagnostic or treatment decisions.");
  }
  return `${lines.join("\n")}\n`;
}
