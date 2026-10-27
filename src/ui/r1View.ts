import { kmForFeature } from "../core/r1";
import type { FeatureResult } from "../core/r1";
import type { R1State } from "../core/skills";
import { h } from "./dom";
import type { Key } from "./i18n";
import { kmChart } from "./kmChart";

type T = (key: Key, vars?: Record<string, string | number>) => string;

export const fmtP = (p: number | null): string => (p === null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));
const fmtN = (v: number | null, d = 2): string => (v === null ? "—" : v.toFixed(d));
const fmtHr = (hr: number | null, lo: number | null, hi: number | null): string =>
  hr === null || lo === null || hi === null ? "—" : `${hr.toFixed(2)} (${lo.toFixed(2)}–${hi.toFixed(2)})`;

/** 結果の表の並び: 使った p の小さい順、計算できなかったものは末尾。 */
export function sortedRows(rows: FeatureResult[]): FeatureResult[] {
  return [...rows].sort((a, b) => (a.pUsed ?? Infinity) - (b.pUsed ?? Infinity));
}

export function r1Panel(st: R1State, t: T, selected: string | null, onSelect: (name: string) => void): HTMLElement {
  const rows = sortedRows(st.rows);
  const pick = rows.find((r) => r.feature === selected) ?? rows.find((r) => r.error === null) ?? null;
  const failed = rows.filter((r) => r.error !== null).length;
  const sel = rows.filter((r) => r.error === null && r.pUsed !== null && (st.params.multiplicity === "bh" ? (r.q as number) < st.params.alpha : r.pUsed < st.params.alpha)).length;
  const opt = st.params.groupMode === "optimal";
  const notices = [t("r1Notice1"), opt ? t("r1NoticeOpt") : null, st.params.multiplicity === "none" ? t("r1NoticeNone") : null].filter((x): x is string => x !== null);
  const pHead = opt ? t("pOptimal") : t("pLogrank");

  const table = h("table", { class: "rs-table", "data-testid": "rs-r1-table" },
    h("thead", {}, h("tr", {},
      ...[t("thFeature"), t("thN"), t("thGroups"), t("thCut"), pHead, ...(opt ? ["raw p"] : []), ...(st.params.multiplicity === "bh" ? [t("thQ")] : []), t("thHrGroup"), t("thHrSd"), t("thReason")].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...rows.map((r) => h("tr", { class: r === pick ? "rs-sel" : "", "data-testid": "rs-r1-row", "data-feature": r.feature, onclick: () => onSelect(r.feature) },
      h("td", {}, r.feature), h("td", {}, `${r.n}/${r.events}`), h("td", {}, r.groupSizes.join("/") || "—"),
      h("td", {}, r.cuts.map((c) => String(Number(c.toPrecision(4)))).join(", ") || "—"),
      h("td", { "data-testid": "rs-r1-p" }, fmtP(r.pUsed)),
      ...(opt ? [h("td", {}, fmtP(r.rawOptimalP))] : []),
      ...(st.params.multiplicity === "bh" ? [h("td", { "data-testid": "rs-r1-q" }, fmtP(r.q))] : []),
      h("td", {}, fmtHr(r.hrGroup, r.hrGroupLower, r.hrGroupUpper)), h("td", {}, fmtHr(r.hrPerSd, r.hrPerSdLower, r.hrPerSdUpper)),
      h("td", {}, r.error ?? "")))),
  );

  const groups = pick ? kmForFeature(st.context, pick) : null;
  const chart = groups
    ? h("div", {},
      h("div", { class: "rs-big", "data-testid": "rs-km-title" }, t("kmTitle", { name: pick!.feature })),
      kmChart(groups, { time: t("kmTime"), survival: t("kmSurvival"), atRisk: t("kmAtRisk"), ci: t("kmCi") }))
    : null;

  const m = st.model;
  const model = h("div", { "data-testid": "rs-model" },
    h("h3", {}, t("modelTitle")),
    !m || (!m.fit && !m.error) ? h("div", { class: "rs-note", "data-testid": "rs-model-none" }, t("modelNone")) : null,
    m?.error ? h("div", { class: "rs-err" }, m.error) : null,
    m?.fit ? h("div", {},
      h("div", { class: "rs-note" }, t("modelUsed", { n: m.nUsed, e: m.eventsUsed })),
      h("table", { class: "rs-table", "data-testid": "rs-model-table" },
        h("thead", {}, h("tr", {}, h("th", {}, t("thTerm")), h("th", {}, t("thHr")), h("th", {}, "p"))),
        h("tbody", {}, ...m.terms.map((x) => h("tr", {}, h("td", {}, x.name), h("td", {}, fmtHr(x.hr, x.lower, x.upper)), h("td", {}, fmtP(x.p)))))),
      m.cIndexApparent !== null ? h("div", { "data-testid": "rs-cindex" }, t("cApparent", { v: fmtN(m.cIndexApparent, 3) })) : null,
      m.corrected ? h("div", { "data-testid": "rs-cindex-corrected" }, t("cCorrected", { v: fmtN(m.corrected.corrected, 3), o: fmtN(m.corrected.optimism, 3), b: m.corrected.bootstraps })) : null,
      m.addedValue ? h("div", { "data-testid": "rs-added" }, t("addedValue", { df: m.addedValue.df, chi2: fmtN(m.addedValue.chi2, 2), p: fmtP(m.addedValue.p) })) : null,
    ) : null,
    ...(m?.notes ?? []).map((n) => h("div", { class: "rs-warn", "data-testid": "rs-model-note" }, n)),
  );

  return h("div", { "data-testid": "rs-r1" },
    h("h2", {}, t("stepR1")),
    ...notices.map((n) => h("div", { class: "rs-warn", "data-testid": "rs-r1-notice" }, n)),
    h("div", { "data-testid": "rs-r1-summary" }, t("r1Summary", { m: st.comparisons, s: sel, f: failed })),
    h("div", { class: "rs-tablewrap" }, table),
    chart,
    model,
  );
}
