import { blandAltmanFor } from "../core/r3";
import type { ReliabilityRow } from "../core/r3";
import type { IccCategory } from "../core/reliability/icc";
import type { R3State } from "../core/recipes";
import { h } from "./dom";
import type { Key } from "./i18n";
import { xyChart } from "./xyChart";

type T = (key: Key, vars?: Record<string, string | number>) => string;
export interface BaChoice { feature: string | null; a: number; b: number }

const NS = "http://www.w3.org/2000/svg";
const CAT_COLOR: Record<IccCategory, string> = { poor: "var(--cat-poor)", moderate: "var(--cat-mod)", good: "var(--cat-good)", excellent: "var(--cat-exc)" };
const catKey = (c: IccCategory): Key => (c === "poor" ? "catPoor" : c === "moderate" ? "catModerate" : c === "good" ? "catGood" : "catExcellent");
const f2 = (v: number, d = 3) => v.toFixed(d);
const el = (tag: string, attrs: Record<string, string | number> = {}, text?: string): SVGElement => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e as SVGElement;
};

/** ICC の棒グラフ: 棒 = 切った後の ICC、線 = 95% CI、色 = 区分（文字でも書く）、点線 = 0.5・0.75・0.9。 */
export function iccChart(rows: ReliabilityRow[], t: T): SVGElement {
  const items = rows.filter((r) => r.result).sort((a, b) => a.result!.singleTruncated.icc - b.result!.singleTruncated.icc);
  const rowH = 16, L = 190, R = 110, T0 = 24, W = 700, plotW = W - L - R, H = T0 + items.length * rowH + 24;
  const X = (v: number) => L + v * plotW;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "data-testid": "rs-icc-svg", style: `max-width:${W}px;height:auto` });
  svg.append(el("title", {}, t("iccChartTitle")));
  for (const v of [0, 0.5, 0.75, 0.9, 1]) {
    svg.append(el("line", { x1: X(v), x2: X(v), y1: T0 - 4, y2: T0 + items.length * rowH, class: v === 0 || v === 1 ? "rs-axisline" : "rs-grid", "stroke-dasharray": v === 0 || v === 1 ? "" : "3 3" }));
    svg.append(el("text", { x: X(v), y: T0 - 8, "text-anchor": "middle", class: "rs-axis" }, String(v)));
  }
  items.forEach((r, i) => {
    const y = T0 + i * rowH + rowH / 2, s = r.result!, tr = s.singleTruncated, color = CAT_COLOR[s.category];
    svg.append(
      el("text", { x: L - 6, y: y + 4, "text-anchor": "end", class: "rs-axis", "data-testid": "rs-icc-label" }, r.feature.length > 28 ? `${r.feature.slice(0, 27)}…` : r.feature),
      el("rect", { x: X(0), y: y - 4, width: Math.max(1, tr.icc * plotW), height: 8, fill: color, opacity: 0.55 }),
      el("line", { x1: X(tr.lower), x2: X(Math.min(1, tr.upper)), y1: y, y2: y, stroke: "var(--fg)", "stroke-width": 1.4 }),
      el("text", { x: X(1) + 8, y: y + 4, class: "rs-axis" }, `${f2(tr.icc, 2)} ${t(catKey(s.category))}`),
    );
  });
  return svg;
}

export function r3Panel(st: R3State, t: T, ba: BaChoice, onBa: (c: BaChoice) => void): HTMLElement {
  const rep = st.data.report;
  const by = (c: IccCategory) => st.rows.filter((r) => r.result?.category === c).length;
  const failed = st.rows.filter((r) => r.error !== null).length;
  const list = (title: string, xs: string[], testId: string) =>
    xs.length ? h("details", { "data-testid": testId }, h("summary", {}, `${title}（${xs.length}）`), h("ul", {}, ...xs.map((x) => h("li", {}, x)))) : null;

  const table = h("table", { class: "rs-table", "data-testid": "rs-r3-table" },
    h("thead", {}, h("tr", {}, ...[t("thFeature"), "n", t("thDropped"), t("thIcc"), t("thCat"), t("thIccAvg"), t("thReason")].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...st.rows.map((r) => {
      const s = r.result;
      return h("tr", { "data-testid": "rs-r3-row", "data-feature": r.feature },
        h("td", {}, r.feature), h("td", {}, String(r.n)), h("td", {}, String(r.dropped)),
        h("td", { "data-testid": "rs-r3-icc" }, s ? `${f2(s.singleTruncated.icc)} (${f2(s.singleTruncated.lower)}–${f2(s.singleTruncated.upper)})` : "—"),
        h("td", { "data-testid": "rs-r3-cat" }, s ? t(catKey(s.category)) : "—"),
        h("td", {}, s ? `${f2(s.averageTruncated.icc)} (${f2(s.averageTruncated.lower)}–${f2(s.averageTruncated.upper)})` : "—"),
        h("td", {}, r.error ?? s?.note ?? ""));
    })));

  // Bland–Altman
  const feats = st.data.features, k = st.data.conditions.length;
  const feature = ba.feature && feats.includes(ba.feature) ? ba.feature : feats[0];
  const a = Math.min(ba.a, k - 1), b = ba.b === a ? (a + 1) % k : Math.min(ba.b, k - 1);
  const sel = (testId: string, label: string, options: [string, string][], value: string, on: (v: string) => void) => {
    const e = h("select", { "data-testid": testId, onchange: (ev: Event) => on((ev.target as HTMLSelectElement).value) }, ...options.map(([v, l]) => h("option", { value: v }, l)));
    e.value = value;
    return h("div", { class: "rs-row" }, h("label", {}, label), e);
  };
  const res = blandAltmanFor(st.data, feature, a, b);
  const baView = h("div", { "data-testid": "rs-ba" },
    h("h3", {}, t("baTitle")),
    sel("rs-ba-feature", t("baFeature"), feats.map((f) => [f, f]), feature, (v) => onBa({ feature: v, a, b })),
    sel("rs-ba-a", t("baA"), st.data.conditions.map((c, i) => [String(i), c]), String(a), (v) => onBa({ feature, a: Number(v), b })),
    sel("rs-ba-b", t("baB"), st.data.conditions.map((c, i) => [String(i), c]), String(b), (v) => onBa({ feature, a, b: Number(v) })),
    "error" in res ? h("div", { class: "rs-err", "data-testid": "rs-ba-error" }, res.error) : h("div", {},
      h("div", { "data-testid": "rs-ba-bias" }, t("baBias", { v: f2(res.bias), lo: f2(res.biasCi[0]), hi: f2(res.biasCi[1]) })),
      h("div", { "data-testid": "rs-ba-loa" }, t("baLoa", { lo: f2(res.lowerLoa), lol: f2(res.lowerLoaCi[0]), loh: f2(res.lowerLoaCi[1]), hi: f2(res.upperLoa), hil: f2(res.upperLoaCi[0]), hih: f2(res.upperLoaCi[1]), sd: f2(res.sd), n: res.n })),
      (() => {
        const xs = res.points.map((p) => p[0]), ys = res.points.map((p) => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), pad = (x1 - x0) * 0.05 || 1;
        const y0 = Math.min(...ys, res.lowerLoaCi[0]), y1 = Math.max(...ys, res.upperLoaCi[1]), py = (y1 - y0) * 0.05 || 1;
        const xr: [number, number] = [x0 - pad, x1 + pad];
        const line = (v: number) => [[xr[0], v], [xr[1], v]];
        return xyChart([
          { label: `${st.data.conditions[a]} → ${st.data.conditions[b]}`, kind: "scatter", points: res.points },
          { label: `bias ${f2(res.bias, 2)}`, kind: "line", points: line(res.bias) },
          { label: `+1.96 SD ${f2(res.upperLoa, 2)}`, kind: "line", points: line(res.upperLoa) },
          { label: `−1.96 SD ${f2(res.lowerLoa, 2)}`, kind: "line", points: line(res.lowerLoa) },
        ], { xLabel: t("baMean"), yLabel: t("baDiff"), xRange: xr, yRange: [y0 - py, y1 + py], testId: "rs-ba-svg", digits: 2 });
      })()));

  return h("div", { "data-testid": "rs-r3" },
    h("h2", {}, t("stepR3")),
    h("div", { "data-testid": "rs-r3-conditions" }, t("r3Conditions", { k, names: st.data.conditions.join(", "), m: rep.matched, f: st.data.features.length })),
    h("div", { class: "rs-warn", "data-testid": "rs-r3-notice" }, t("r3Notice1")),
    h("div", { class: "rs-warn" }, t("r3Notice2")),
    list(t("r3Unmatched"), rep.unmatched.map((u) => `${u.id}（無い条件: ${u.missingIn.join(", ")}）`), "rs-r3-unmatched"),
    list(t("r3Excluded"), rep.excludedColumns.map((c) => `${c.name}: ${c.reason}`), "rs-r3-excluded"),
    h("div", { "data-testid": "rs-r3-summary" }, t("r3Summary", { poor: by("poor"), moderate: by("moderate"), good: by("good"), excellent: by("excellent"), failed })),
    h("div", { class: "rs-tablewrap" }, table),
    h("div", { class: "rs-big" }, t("iccChartTitle")),
    h("div", { style: "max-height:420px;overflow-y:auto;border:1px solid var(--line);border-radius:6px;padding:4px" }, iccChart(st.rows, t)),
    baView);
}
