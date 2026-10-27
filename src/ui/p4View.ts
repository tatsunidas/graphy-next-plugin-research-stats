import { rocPoints } from "../core/discrim/roc";
import { calibrationBins } from "../core/discrim/calibration";
import type { P4State } from "../core/skills";
import { h } from "./dom";
import type { Key } from "./i18n";
import { xyChart, type XySeries } from "./xyChart";
import { hyperText } from "../core/pipeline/nestedCv";

type T = (key: Key, vars?: Record<string, string | number>) => string;
const f3 = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(3));

/** 重要度の上位（最大 15 個）の横棒。 */
function importanceChart(items: { n: string; v: number }[], t: T): HTMLElement {
  const top = [...items].sort((a, b) => b.v - a.v).slice(0, 15);
  const max = Math.max(...top.map((x) => x.v), 1e-12);
  return h("div", { "data-testid": "rs-p4-importance" },
    h("div", { class: "rs-big" }, t("p4ImpTitle")),
    h("table", { class: "rs-table" }, h("tbody", {}, ...top.map((x) => h("tr", {},
      h("td", {}, x.n), h("td", {}, x.v.toFixed(3)),
      h("td", { style: "width:50%" }, h("div", { class: "rs-bar", style: `width:${((x.v / max) * 100).toFixed(1)}%` })))))));
}

/** P4 の結果: 人数、外側の分割ごとの評価（表と図）、束ねた ROC・較正、選ばれた回数、最終モデル、テスト集合の評価。 */
export function p4Panel(st: P4State, t: T): HTMLElement {
  const bin = st.cohort.task.kind === "binary";
  const metric = bin ? "AUC" : "C-index";
  const sp = st.split, qc = st.qc, cv = st.cv, fm = st.final, te = st.test;
  const list = (title: string, xs: string[], testId: string) =>
    xs.length ? h("details", { "data-testid": testId }, h("summary", {}, `${title}（${xs.length}）`), h("ul", {}, ...xs.map((x) => h("li", {}, x)))) : null;

  const lockInfo = sp?.lock
    ? h("div", { class: sp.lockIsNew ? "rs-warn" : "rs-note", "data-testid": "rs-p4-lock-info" },
      t("p4LockInfo", { sha: (te?.lock ?? sp.lock).sha256.slice(0, 12), n: sp.test.length, e: (te?.lock ?? sp.lock).evaluations.length }),
      sp.lockIsNew ? h("div", {}, t("p4LockNewNote")) : null,
      h("div", { class: "rs-note" }, t("p4LockLimit")))
    : null;

  const sizes = qc ? h("div", { class: "rs-big", "data-testid": "rs-p4-sizes" }, t("p4Sizes", { dev: qc.dev.length, test: sp?.test.length ?? 0, m: st.names.length, r: (qc.dev.length / st.names.length).toFixed(2) })) : null;

  let cvView: HTMLElement | null = null;
  if (cv) {
    const lasso = st.params.model === "lasso", ml = st.params.ml !== null;
    const table = h("table", { class: "rs-table", "data-testid": "rs-p4-folds" },
      h("thead", {}, h("tr", {}, ...[t("thRepeat"), t("thFold"), t("thTrain"), t("thTestN"), "k", ...(lasso ? ["λ"] : []), ...(ml ? [t("thParams")] : []), metric, t("thFeatures"), t("thReason")].map((x) => h("th", {}, x)))),
      h("tbody", {}, ...cv.folds.map((f) => h("tr", { "data-testid": "rs-p4-fold" },
        h("td", {}, String(f.repeat + 1)), h("td", {}, String(f.fold + 1)), h("td", {}, String(f.nTrain)), h("td", {}, String(f.nTest)),
        h("td", {}, f.k === null ? "—" : String(f.k)), lasso ? h("td", { "data-testid": "rs-p4-fold-lambda" }, f.lambda === null ? "—" : f.lambda.toPrecision(3)) : null,
        ml ? h("td", { "data-testid": "rs-p4-fold-params" }, hyperText(f.params) || "—") : null,
        h("td", { "data-testid": "rs-p4-fold-score" }, f3(f.score)),
        h("td", {}, f.cols.map((j) => st.names[j]).join(" + ")), h("td", {}, f.error ?? "")))));
    const pts = cv.folds.map((f, q) => (f.score === null ? null : [q + 1, f.score])).filter((x): x is number[] => x !== null);
    const dev = st.qc!.dev;
    const charts: (HTMLElement | null)[] = [
      h("div", {}, h("div", { class: "rs-big" }, t("p4FoldChart", { metric })),
        xyChart([{ label: metric, kind: "points", points: pts }, ...(cv.summary.mean !== null ? [{ label: t("p4Mean"), kind: "line" as const, points: [[0.5, cv.summary.mean], [cv.folds.length + 0.5, cv.summary.mean]] }] : [])],
          { xLabel: t("p4FoldX"), yLabel: metric, xRange: [0.5, cv.folds.length + 0.5], yRange: [0, 1], testId: "rs-p4-fold-svg", digits: 1 })),
    ];
    if (bin && st.cohort.task.kind === "binary") {
      const y = st.cohort.task.y;
      const ok = dev.map((i, p) => ({ i, v: cv.pred[0][p] })).filter((x): x is { i: number; v: number } => x.v !== null);
      const l = { score: ok.map((x) => x.v), label: ok.map((x) => y[x.i]) };
      const roc: XySeries[] = [{ label: t("p4RocCv"), kind: "step", points: rocPoints(l).map((p) => [p.fpr, p.tpr]) }];
      if (te && te.roc.length) roc.push({ label: t("p4RocTest"), kind: "step", points: te.roc.map((p) => [p.fpr, p.tpr]) });
      charts.push(h("div", {}, h("div", { class: "rs-big" }, t("p4RocTitle")), xyChart(roc, { xLabel: t("rocX"), yLabel: t("rocY"), xRange: [0, 1], yRange: [0, 1], diagonal: true, testId: "rs-p4-roc-svg" })));
      if (st.params.model === "svm") charts.push(h("div", { class: "rs-note", "data-testid": "rs-p4-cal-na" }, t("p4CalNa")));
      else {
        const cal = calibrationBins(l.score, l.label, st.params.calibrationBins);
        charts.push(h("div", {}, h("div", { class: "rs-big" }, t("p4CalTitle")),
          xyChart([{ label: t("p4RocCv"), kind: "points", points: cal.map((b) => [b.meanPredicted, b.observed.value ?? 0, b.observed.lower ?? 0, b.observed.upper ?? 1]) }],
            { xLabel: t("calX"), yLabel: t("calY"), xRange: [0, 1], yRange: [0, 1], diagonal: true, testId: "rs-p4-cal-svg" })));
      }
    }
    const sel = st.names.map((n, j) => ({ n, c: cv.selectionCounts[j] })).filter((x) => x.c > 0).sort((a, b) => b.c - a.c);
    cvView = h("div", { "data-testid": "rs-p4-cv" },
      h("h3", {}, t("p4CvTitle")),
      h("div", { class: "rs-big", "data-testid": "rs-p4-cv-summary" }, t("p4CvSummary", { metric, mean: f3(cv.summary.mean), min: f3(cv.summary.min), max: f3(cv.summary.max), v: cv.summary.valid, t: cv.summary.total })),
      bin ? h("div", { "data-testid": "rs-p4-pooled" }, t("p4Pooled", { v: cv.pooled.map((x) => f3(x)).join(", ") })) : null,
      h("div", { class: "rs-warn", "data-testid": "rs-p4-noci" }, t("p4NoCi")),
      h("div", { class: "rs-tablewrap" }, table),
      h("div", { class: "rs-charts" }, ...charts),
      sel.length ? h("div", { class: "rs-note", "data-testid": "rs-p4-sel" }, `${t("p4SelTitle", { n: cv.folds.length })}: ${sel.map((x) => `${x.n} (${x.c})`).join(", ")}`) : null,
    );
  }

  const finalView = fm ? h("div", { "data-testid": "rs-p4-final" },
    h("h3", {}, t("p4FinalTitle")),
    fm.model ? h("div", {}, t("p4Final", { k: fm.model.cols.length, features: fm.model.cols.map((j) => st.names[j]).join(" + ") || "—" })) : h("div", { class: "rs-err" }, fm.error ?? ""),
    fm.model && fm.choice.lambda !== null ? h("div", { "data-testid": "rs-p4-final-lambda" }, `λ = ${fm.choice.lambda.toPrecision(3)}`) : null,
    fm.model?.resolved ? h("div", { "data-testid": "rs-p4-final-params" }, `${t("thParams")}: ${hyperText(fm.model.resolved)}`) : null,
    fm.model?.importance ? importanceChart(fm.model.importance.map((v, a) => ({ n: st.names[fm.model!.cols[a]], v })), t) : null,
    h("div", { class: "rs-note", "data-testid": "rs-p4-final-note" }, t(st.params.ml ? "p4FinalNoteMl" : st.params.model === "lasso" ? "p4FinalNoteLasso" : "p4FinalNote"))) : null;

  const testView = te ? h("div", { "data-testid": "rs-p4-test" },
    h("h3", {}, t("p4TestTitle")),
    h("div", { class: "rs-big", "data-testid": "rs-p4-test-value" }, t(bin ? "p4TestAuc" : "p4TestC", { v: f3(te.value), lo: f3(te.lower), hi: f3(te.upper), n: te.n, b: st.params.testBootstrapB })),
    h("div", { class: te.evaluation > 1 ? "rs-warn" : "rs-note", "data-testid": "rs-p4-test-count" }, t("p4TestCount", { n: te.evaluation })),
    list(t("p4TestExcluded"), te.excluded, "rs-p4-test-excluded")) : null;

  return h("div", { "data-testid": "rs-p4" },
    h("h2", {}, t("stepP4")),
    lockInfo, sizes,
    qc ? list(t("p4Removed"), qc.removed.map((r) => `${r.feature}: ${r.reason}`), "rs-p4-removed") : null,
    qc ? list(t("p4Dropped"), qc.droppedDev, "rs-p4-dropped") : null,
    cvView, finalView, testView);
}
