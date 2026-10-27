import type { R5State } from "../core/skills";
import { h } from "./dom";
import type { Key } from "./i18n";
import { fmtP } from "./r1View";
import { xyChart } from "./xyChart";

type T = (key: Key, vars?: Record<string, string | number>) => string;
const f3 = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(3));

const range = (vals: number[]): [number, number] => {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1;
  return [lo - pad, hi + pad];
};

/** R5 の結果: 人数、選んだ特徴量の HR、補正 C-index、選択の経過、絞り込み（重要度か LASSO の交差検証）、Schoenfeld。 */
export function r5Panel(st: R5State, t: T): HTMLElement | null {
  const m = st.model;
  if (!m) return null;
  const r = m.result;
  const events = st.cohort.surv.event.reduce((a, b) => a + b, 0);
  const act: Record<string, Key> = { remove: "actRemove", add: "actAdd", cap: "actCap", cycle: "actCycle" };

  const model = m.terms.length
    ? h("table", { class: "rs-table", "data-testid": "rs-r5-terms" },
      h("thead", {}, h("tr", {}, ...[t("thFeature"), t("thHrPerSd"), t("thP")].map((x) => h("th", {}, x)))),
      h("tbody", {}, ...m.terms.map((x) => h("tr", { "data-testid": "rs-r5-term" },
        h("td", {}, x.name), h("td", {}, `${x.hr.toFixed(2)} (${x.lower.toFixed(2)}–${x.upper.toFixed(2)})`), h("td", {}, fmtP(x.p))))))
    : h("div", { class: "rs-warn", "data-testid": "rs-r5-none" }, t("r5None"));

  const cview = h("div", { "data-testid": "rs-r5-c" },
    h("div", { class: "rs-big", "data-testid": "rs-r5-capp" }, t("r5CApp", { a: f3(m.cApparent) })),
    m.corrected ? h("div", { class: "rs-big", "data-testid": "rs-r5-ccorr" }, t("r5CCorr", { c: f3(m.corrected.corrected), o: f3(m.corrected.optimism), b: m.corrected.bootstraps, f: m.corrected.failed })) : null);

  const steps = r.steps.length
    ? h("details", { "data-testid": "rs-r5-steps" }, h("summary", {}, `${t("r5Steps")}（${r.steps.length}）`),
      h("table", { class: "rs-table" },
        h("thead", {}, h("tr", {}, ...[t("thAction"), t("thFeature"), t("thP")].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...r.steps.map((s) => h("tr", {}, h("td", {}, t(act[s.action])), h("td", {}, st.names[s.col]), h("td", {}, fmtP(s.p)))))))
    : null;

  let screen: HTMLElement | null = null;
  if (r.boost) {
    const b = r.boost;
    const rows = b.std.cols.map((c, a) => ({ c, imp: b.importance[a], beta: b.beta[a] })).filter((x) => x.beta !== 0).sort((x, y) => y.imp - x.imp);
    screen = h("details", { "data-testid": "rs-r5-importance", open: true }, h("summary", {}, `${t("r5Imp")}（${rows.length}）`),
      h("div", { class: "rs-tablewrap" }, h("table", { class: "rs-table" },
        h("thead", {}, h("tr", {}, ...[t("thFeature"), t("thImp")].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...rows.map((x) => h("tr", {}, h("td", {}, st.names[x.c]), h("td", {}, x.imp.toPrecision(4))))))));
  } else if (r.lasso) {
    const l = r.lasso;
    const pts = l.lambdas.map((lam, a) => (l.cvm[a] === null ? null : [Math.log10(lam), l.cvm[a] as number, (l.cvm[a] as number) - (l.cvsd[a] as number), (l.cvm[a] as number) + (l.cvsd[a] as number)])).filter((x): x is number[] => x !== null);
    const ys = pts.flatMap((p) => [p[2], p[3]]);
    const x = Math.log10(l.lambda);
    screen = h("div", { "data-testid": "rs-r5-lasso" }, h("div", { class: "rs-big" }, t("r5Path")),
      xyChart([{ label: t("r5PathY"), kind: "points", points: pts }, { label: `λ = ${l.lambda.toPrecision(3)}`, kind: "line", points: [[x, Math.min(...ys)], [x, Math.max(...ys)]] }],
        { xLabel: t("r5PathX"), yLabel: t("r5PathY"), xRange: range(pts.map((p) => p[0])), yRange: range(ys), testId: "rs-r5-lasso-svg", digits: 2 }));
  }

  let sch: HTMLElement | null = null;
  if (m.schoenfeld) {
    const s = m.schoenfeld;
    const charts = r.selected.map((j, q) => {
      const ys = s.scaled.map((v) => v[q]);
      return h("div", {}, h("div", { class: "rs-big" }, `${st.names[j]}: p = ${fmtP(s.tests[q].p)}`),
        xyChart([
          { label: t("r5SchY"), kind: "scatter", points: s.times.map((tm, i) => [tm, ys[i]]) },
          { label: t("r5Smooth", { w: s.window }), kind: "line", points: s.times.map((tm, i) => [tm, s.smooth[i][q]]) },
        ], { xLabel: t("r5SchX"), yLabel: t("r5SchY"), xRange: range(s.times), yRange: range(ys), testId: `rs-r5-sch-svg-${q}`, digits: 1 }));
    });
    sch = h("div", { "data-testid": "rs-r5-sch" },
      h("h3", {}, t("r5Sch")),
      h("table", { class: "rs-table", "data-testid": "rs-r5-sch-table" },
        h("thead", {}, h("tr", {}, ...[t("thFeature"), "χ²", t("thP")].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...r.selected.map((j, q) => h("tr", {}, h("td", {}, st.names[j]), h("td", {}, s.tests[q].chi2.toFixed(3)), h("td", {}, fmtP(s.tests[q].p)))))),
      h("div", { "data-testid": "rs-r5-sch-global" }, t("r5SchGlobal", { c: s.global.chi2.toFixed(3), df: s.global.df, p: fmtP(s.global.p) })),
      h("div", { class: "rs-note" }, t("r5SchNote", { w: s.window })),
      h("div", { class: "rs-charts" }, ...charts));
  }

  return h("div", { "data-testid": "rs-r5" },
    h("h2", {}, t("skillR5")),
    h("div", { class: "rs-big", "data-testid": "rs-r5-sizes" }, t("r5Sizes", { n: st.cohort.patients.length, e: events, m: m.used.length })),
    h("h3", {}, t("r5Selected", { k: r.selected.length })),
    model, cview, steps, screen, sch);
}
