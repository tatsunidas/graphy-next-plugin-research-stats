import { treeNodes, type TreeNode } from "../core/discrim/tree";
import type { Performance } from "../core/discrim/roc";
import { rocPoints } from "../core/discrim/roc";
import type { FeatureRow } from "../core/r2";
import { meetsCriterion } from "../core/r2";
import type { R2State } from "../core/recipes";
import { h } from "./dom";
import type { Key } from "./i18n";
import { fmtP } from "./r1View";
import { xyChart, type XySeries } from "./xyChart";

type T = (key: Key, vars?: Record<string, string | number>) => string;

const f2 = (v: number | null, d = 2) => (v === null ? "—" : v.toFixed(d));
const pct = (p: { value: number | null; lower: number | null; upper: number | null } | undefined) =>
  !p || p.value === null ? "—" : `${(p.value * 100).toFixed(1)}% (${((p.lower ?? 0) * 100).toFixed(1)}–${((p.upper ?? 1) * 100).toFixed(1)})`;
const num = (v: number) => String(Number(v.toPrecision(4)));

export const sortedFeatures = (rows: FeatureRow[]): FeatureRow[] =>
  [...rows].sort((a, b) => (a.mwP ?? Infinity) - (b.mwP ?? Infinity));

function confusionTable(t: T, title: string, c: Performance["confusion"], testId: string): HTMLElement {
  return h("div", { "data-testid": testId },
    h("div", { class: "rs-note" }, title),
    h("table", { class: "rs-cm" },
      h("thead", {}, h("tr", {}, h("th", {}, `${t("cmActual")} \\ ${t("cmPred")}`), h("th", {}, t("cmPos")), h("th", {}, t("cmNeg")))),
      h("tbody", {},
        h("tr", {}, h("th", {}, t("cmPos")), h("td", { "data-cell": "tp" }, String(c.tp)), h("td", { "data-cell": "fn" }, String(c.fn))),
        h("tr", {}, h("th", {}, t("cmNeg")), h("td", { "data-cell": "fp" }, String(c.fp)), h("td", { "data-cell": "tn" }, String(c.tn))))));
}

function treeList(t: T, names: string[], node: TreeNode): HTMLElement {
  const label = (n: TreeNode) => (n.label === 1 ? t("cmPos") : t("cmNeg"));
  const li = h("li", {});
  if (node.split) {
    const op = node.split.direction === "higher" ? ">" : "<";
    li.append(h("b", {}, t("treeNodeText", { name: names[node.split.feature], op, cut: num(node.split.cut) })), ` (n=${node.n})`);
    li.append(h("ul", {},
      h("li", {}, h("span", { class: "rs-tag" }, `yes → ${t("cmPos")}`), h("ul", {}, treeList(t, names, node.positive!))),
      h("li", {}, h("span", { class: "rs-tag" }, `no → ${t("cmNeg")}`), h("ul", {}, treeList(t, names, node.negative!)))));
  } else {
    li.append(t("treeLeaf", { label: label(node), n: node.n, pos: node.pos }));
  }
  return li;
}

export function r2Panel(st: R2State, t: T, selected: string | null, onSelect: (name: string) => void): HTMLElement {
  const rows = sortedFeatures(st.rows);
  const pick = rows.find((r) => r.feature === selected) ?? rows.find((r) => r.error === null) ?? null;
  const failed = rows.filter((r) => r.error !== null).length;
  const sel = rows.filter((r) => meetsCriterion(r, st.params)).length;
  const bh = st.params.multiplicity === "bh";
  const table = h("table", { class: "rs-table", "data-testid": "rs-r2-table" },
    h("thead", {}, h("tr", {}, ...[t("thFeature"), "n", t("thDir"), t("thAuc"), "p", ...(bh ? [t("thQ")] : []), t("thCutoff"), t("thSens"), t("thSpec"), t("thPpv"), t("thNpv"), t("thAcc"), t("thReason")].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...rows.map((r) => h("tr", { class: r === pick ? "rs-sel" : "", "data-testid": "rs-r2-row", "data-feature": r.feature, onclick: () => onSelect(r.feature) },
      h("td", {}, r.feature), h("td", {}, `${r.n}/${r.positives}`), h("td", {}, r.direction === null ? "—" : r.direction === "higher" ? t("dirHigher") : t("dirLower")),
      h("td", { "data-testid": "rs-r2-auc" }, r.auc === null ? "—" : `${f2(r.auc, 3)} (${f2(r.aucLower, 3)}–${f2(r.aucUpper, 3)})`),
      h("td", { "data-testid": "rs-r2-p" }, fmtP(r.mwP)), ...(bh ? [h("td", {}, fmtP(r.q))] : []),
      h("td", { "data-testid": "rs-r2-cut" }, r.cut === null ? "—" : num(r.cut)),
      h("td", {}, pct(r.perf?.sensitivity)), h("td", {}, pct(r.perf?.specificity)), h("td", {}, pct(r.perf?.ppv)), h("td", {}, pct(r.perf?.npv)), h("td", {}, pct(r.perf?.accuracy)),
      h("td", {}, r.error ?? "")))));

  const co = st.combo;
  const roc: XySeries[] = [];
  if (pick && pick.error === null && pick.direction) {
    const j = st.context.names.indexOf(pick.feature);
    const ix = st.context.values[j].map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
    roc.push({ label: t("rocFeature", { name: pick.feature }), kind: "step", points: rocPoints({ score: ix.map((i) => st.context.values[j][i] as number), label: ix.map((i) => st.context.y[i]) }, pick.direction).map((p) => [p.fpr, p.tpr]) });
  }
  if (co && co.rocApparent.length) roc.push({ label: t("rocApparent"), kind: "step", points: co.rocApparent.map((p) => [p.fpr, p.tpr]) });
  if (co && co.rocLoo.length) roc.push({ label: t("rocLoo"), kind: "step", points: co.rocLoo.map((p) => [p.fpr, p.tpr]) });
  const charts = h("div", { class: "rs-charts" },
    roc.length ? h("div", {}, h("div", { class: "rs-big" }, t("rocTitle")), xyChart(roc, { xLabel: t("rocX"), yLabel: t("rocY"), xRange: [0, 1], yRange: [0, 1], diagonal: true, testId: "rs-roc-svg" })) : null,
    co && co.calibration.length ? h("div", {}, h("div", { class: "rs-big" }, t("calTitle")), xyChart([{ label: t("rocLoo"), kind: "points", points: co.calibration.map((b) => [b.meanPredicted, b.observed.value ?? 0, b.observed.lower ?? 0, b.observed.upper ?? 1]) }], { xLabel: t("calX"), yLabel: t("calY"), xRange: [0, 1], yRange: [0, 1], diagonal: true, testId: "rs-cal-svg" })) : null,
    co && co.dca.length ? h("div", {}, h("div", { class: "rs-big" }, t("dcaTitle")), (() => {
      const top = Math.max(0.05, ...co.dca.map((d) => Math.max(d.model, d.all)));
      const lo = Math.min(-0.05, ...co.dca.map((d) => Math.min(d.model, 0)));
      const clipLo = (v: number) => Math.max(lo, v);
      return xyChart([
        { label: t("dcaModel"), kind: "line", points: co.dca.map((d) => [d.threshold, d.model]) },
        { label: t("dcaAll"), kind: "line", points: co.dca.map((d) => [d.threshold, clipLo(d.all)]) },
        { label: t("dcaNone"), kind: "line", points: co.dca.map((d) => [d.threshold, d.none]) },
      ], { xLabel: t("dcaX"), yLabel: t("dcaY"), xRange: [0, 1], yRange: [lo, top], testId: "rs-dca-svg", digits: 2 });
    })()) : null,
  );

  const cms = h("div", { class: "rs-charts" },
    pick?.perf ? confusionTable(t, t("cmFeature", { name: pick.feature, cut: num(pick.cut as number) }), pick.perf.confusion, "rs-cm-feature") : null,
    co?.looPerformance ? confusionTable(t, t("cmLoo"), co.looPerformance.confusion, "rs-cm-loo") : null);

  const combo = h("div", { "data-testid": "rs-combo" },
    h("h3", {}, t("comboTitle")),
    !co || co.error || !co.features.length ? h("div", { class: co?.error ? "rs-err" : "rs-note", "data-testid": "rs-combo-none" }, co?.error ?? t("comboNone")) : h("div", {},
      h("div", { class: "rs-note" }, t("comboUsed", { features: co.features.join(" + "), n: co.nUsed, p: co.positives })),
      h("table", { class: "rs-table", "data-testid": "rs-combo-table" },
        h("thead", {}, h("tr", {}, h("th", {}, t("thTerm")), h("th", {}, t("thOr")), h("th", {}, "p"))),
        h("tbody", {}, ...co.terms.map((x) => h("tr", {}, h("td", {}, x.name), h("td", {}, `${f2(x.oddsRatio)} (${f2(x.lower)}–${f2(x.upper)})`), h("td", {}, fmtP(x.p)))))),
      co.apparent ? h("div", { "data-testid": "rs-auc-apparent" }, t("aucApparent", { v: f2(co.apparent.auc, 3), lo: f2(co.apparent.lower, 3), hi: f2(co.apparent.upper, 3) })) : null,
      co.loo ? h("div", { "data-testid": "rs-auc-loo" }, t("aucLoo", { v: f2(co.loo.auc, 3), lo: f2(co.loo.lower, 3), hi: f2(co.loo.upper, 3), n: co.looCohort })) : null,
      co.calibrationFit ? h("div", { "data-testid": "rs-calfit" }, t("calFit", { a: f2(co.calibrationFit.intercept, 3), b: f2(co.calibrationFit.slope, 3) })) : null,
      co.selectionCounts.length ? h("div", { class: "rs-note", "data-testid": "rs-selcounts" }, `${t("selTitle")}: ${co.selectionCounts.map((x) => `${x.feature} (${x.count})`).join(", ")}`) : null),
    ...(co?.notes ?? []).map((n) => h("div", { class: "rs-warn", "data-testid": "rs-combo-note" }, n)));

  const tr = st.tree;
  const treeView = st.params.tree ? h("div", { "data-testid": "rs-tree" },
    h("h3", {}, t("treeTitle")),
    tr ? h("div", {},
      h("div", { class: "rs-tree", "data-testid": "rs-tree-list" }, h("ul", {}, treeList(t, st.context.names, tr.root))),
      h("div", { "data-testid": "rs-tree-perf" }, t("treePerf", { se: pct(tr.perf.sensitivity), sp: pct(tr.perf.specificity), ac: pct(tr.perf.accuracy) })),
      h("div", { "data-testid": "rs-tree-mc" }, t("treeMc", { it: tr.mc.iterations, se: ci(tr.mc.sensitivity), sp: ci(tr.mc.specificity), ac: ci(tr.mc.accuracy) })),
      h("div", { class: "rs-note" }, `${treeNodes(tr.root).filter((n) => n.split).length} nodes`),
      ...tr.notes.map((n) => h("div", { class: "rs-warn" }, n))) : h("div", { class: "rs-note", "data-testid": "rs-tree-none" }, st.treeNote ?? t("comboNone"))) : null;

  return h("div", { "data-testid": "rs-r2" },
    h("h2", {}, t("stepR2")),
    h("div", { "data-testid": "rs-r2-labels" }, t("r2Labels", { pos: st.cohort.labels[1], np: st.cohort.y.reduce((a, b) => a + b, 0), neg: st.cohort.labels[0], nn: st.cohort.y.length - st.cohort.y.reduce((a, b) => a + b, 0) })),
    h("div", { class: "rs-warn", "data-testid": "rs-r2-notice" }, t("r2Notice1")),
    h("div", { class: "rs-warn" }, t("r2NoticeCal")),
    st.params.multiplicity === "none" ? h("div", { class: "rs-warn", "data-testid": "rs-r2-notice-none" }, t("r1NoticeNone")) : null,
    h("div", { "data-testid": "rs-r2-summary" }, t("r2Summary", { m: st.comparisons, s: sel, f: failed })),
    h("div", { class: "rs-tablewrap" }, table),
    charts, cms, combo, treeView);
}

const ci = (x: { lower: number; upper: number }) => `${(x.lower * 100).toFixed(1)}–${(x.upper * 100).toFixed(1)}%`;
