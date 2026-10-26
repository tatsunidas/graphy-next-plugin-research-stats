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

/** R2（判別）の手順の記録から、解析の方法の文を作る。記録に無い手順は書かない。 */
function r2Lines(record: RunRecord, locale: Locale): string[] {
  const feat = record.steps.find((s) => s.id === "features" && s.status === "ok");
  if (!feat) return [];
  const p = record.params, oc = record.steps.find((s) => s.id === "outcome"), co = record.steps.find((s) => s.id === "combination" && s.status === "ok");
  const tree = record.steps.find((s) => s.id === "tree" && s.status === "ok");
  const bh = p.multiplicity === "bh";
  const num = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : "?");
  const crit = p.criterion === "accuracy" ? { ja: "診断精度が最大", en: "maximum accuracy" } : { ja: "Youden 指数が最大", en: "maximum Youden index" };
  const L: string[] = [];
  if (locale === "ja") {
    L.push(`結果ラベル「${oc?.summary.positiveLabel}」を陽性（${oc?.summary.positive} 人）、「${oc?.summary.negativeLabel}」を陰性（${oc?.summary.negative} 人）とした。特徴量ごとに ROC 曲線と AUC（DeLong 法の 95% 信頼区間。向きは AUC が 0.5 以上になる側）を求め、${crit.ja}となるしきい値での感度・特異度・陽性的中率・陰性的中率・精度を Clopper–Pearson の正確な 95% 信頼区間とともに示した。事象の有無での値の差は Mann–Whitney 検定（正規近似）で調べた。`);
    L.push(bh ? `${feat.summary.comparisons} 個の比較を Benjamini–Hochberg 法で補正し、q < ${num(p.alpha)} を基準とした。` : `多重比較の補正をしない（探索的な解析）。p < ${num(p.alpha)} を目安とした。`);
    if (co) {
      L.push(`基準を満たした上位 ${p.topK} 個（${co.summary.features}）を標準化し、ロジスティック回帰で組み合わせた（完全症例 ${co.summary.nUsed} 人）。AUC は見かけ ${num(co.summary.apparentAuc)}、leave-one-out（特徴量の選択も各折の内側でやり直す）で ${num(co.summary.looAuc)}（評価対象 ${co.summary.looCohort} 人）。`);
    }
    if (tree) L.push(`決定木は、各節点でその枝に残った症例の診断精度が最大になる特徴量としきい値を選び、枝の誤分類率が ${num(p.treeStopRate)} 未満になれば止めた（節点 ${tree.summary.nodes} 個、木を作った同じ患者での精度 ${num(tree.summary.accuracy)}。検証はしていない）。信頼区間は、各節点の感度・特異度の不確かさを木に伝えるモンテカルロ（${p.mcIterations} 回）の 2.5・97.5 パーセンタイルとした。`);
  } else {
    L.push(`"${oc?.summary.positiveLabel}" was the positive outcome (${oc?.summary.positive} patients) and "${oc?.summary.negativeLabel}" the negative (${oc?.summary.negative}). For each feature, the ROC curve and AUC (DeLong 95% CI; the direction giving AUC of at least 0.5) were computed, and sensitivity, specificity, PPV, NPV and accuracy at the cut-off with ${crit.en} are given with exact Clopper–Pearson 95% CIs. Values by outcome were compared with the Mann–Whitney test (normal approximation).`);
    L.push(bh ? `${feat.summary.comparisons} comparisons were corrected with the Benjamini–Hochberg procedure (q < ${num(p.alpha)}).` : `No correction for multiple comparisons was applied (exploratory analysis); p < ${num(p.alpha)} was used as a guide.`);
    if (co) L.push(`The top ${p.topK} features meeting the criterion (${co.summary.features}) were standardised and combined by logistic regression (complete cases ${co.summary.nUsed}). The AUC was ${num(co.summary.apparentAuc)} apparent and ${num(co.summary.looAuc)} by leave-one-out with the feature selection repeated inside each fold (${co.summary.looCohort} patients evaluated).`);
    if (tree) L.push(`The decision tree chose at each node the feature and cut-off with the highest accuracy among the cases left in the branch, and stopped when the misclassification rate of a branch fell below ${num(p.treeStopRate)} (${tree.summary.nodes} nodes; accuracy ${num(tree.summary.accuracy)} in the same patients used to build the tree; not validated). Confidence intervals are the 2.5 and 97.5 percentiles of a Monte Carlo simulation (${p.mcIterations} iterations) propagating the uncertainty of each node's sensitivity and specificity.`);
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
    lines.push(...r1Lines(record, "ja"), ...r2Lines(record, "ja"));
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
    lines.push(...r1Lines(record, "en"), ...r2Lines(record, "en"));
    lines.push(`Random seed: ${record.seed}. ${changed.length ? `Settings changed from the defaults: ${changed.join("; ")}.` : "All settings were the defaults."}`);
    lines.push("For research use only; not for diagnostic or treatment decisions.");
  }
  return `${lines.join("\n")}\n`;
}
