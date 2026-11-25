import type { PreInfo } from "../core/pipeline/pre";
import type { PDropEntry } from "../core/pipeline/nestedCv";
import type { AssumptionRow } from "../core/stats/assumptionReport";
import { h } from "./dom";
import type { Key } from "./i18n";
import { fmtP } from "./r1View";

type T = (key: Key, vars?: Record<string, string | number>) => string;
const f3 = (v: number | null) => (v === null ? "" : v.toFixed(3));
const tests = (x: Record<string, number>) => Object.entries(x).map(([k, v]) => `${k}: ${v}`).join(", ");

/** 前処理の結果（除いた特徴量の数と一覧）。何も除かず何も報告しなければ出さない。 */
export function preResultView(info: PreInfo | null, t: T, testId: string, corrected = false): HTMLElement | null {
  if (!info) return null;
  const lang = t("lang") as "ja" | "en";
  return h("div", { "data-testid": testId },
    h("h3", {}, t("preResultTitle")),
    h("div", { "data-testid": `${testId}-count` }, t("preCount", { before: info.before, after: info.after, n: info.before - info.after })),
    info.removed.length && corrected ? h("div", { class: "rs-note", "data-testid": `${testId}-m` }, t("preMNote")) : null,
    info.paired ? h("div", { class: "rs-note", "data-testid": `${testId}-paired` }, t("prePairedTests", { tests: tests(info.paired.tests) || "—" })) : null,
    info.removed.length
      ? h("details", { "data-testid": `${testId}-removed` }, h("summary", {}, `${t("preRemoved")}（${info.removed.length}）`),
        h("ul", {}, ...info.removed.map((r) => h("li", {}, `${r.feature}: ${r.reason[lang]}`))))
      : null);
}

/** 前提の検定の表（報告）。 */
export function assumptionView(a: { alpha: number; rows: AssumptionRow[] } | null, t: T): HTMLElement | null {
  if (!a) return null;
  const sw = (c: AssumptionRow["shapiro"][0]) => (c.p === null ? "" : `${f3(c.stat)} / ${fmtP(c.p)}`);
  const lang = t("lang") as "ja" | "en";
  return h("div", { "data-testid": "rs-assump" },
    h("h3", {}, t("assumpTitle")),
    h("div", { class: "rs-note" }, t("assumpNote", { alpha: a.alpha })),
    h("div", { class: "rs-tablewrap" },
      h("table", { class: "rs-table", "data-testid": "rs-assump-table" },
        h("thead", {}, h("tr", {}, ...[t("thFeature"), t("thN0"), t("thN1"), t("thSw0"), t("thSw1"), t("thLevene"), t("thTestChosen"), t("thTestP"), t("thReason")].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...a.rows.map((r) => h("tr", { "data-testid": "rs-assump-row", "data-feature": r.feature },
          h("td", {}, r.feature), h("td", {}, String(r.n[0])), h("td", {}, String(r.n[1])), h("td", { title: r.shapiro[0].reason ?? "" }, sw(r.shapiro[0])), h("td", { title: r.shapiro[1].reason ?? "" }, sw(r.shapiro[1])),
          h("td", { title: r.levene.reason ?? "" }, r.levene.p === null ? "" : fmtP(r.levene.p)), h("td", { "data-testid": "rs-assump-test" }, r.test ?? ""), h("td", {}, r.p === null ? "" : fmtP(r.p)),
          h("td", {}, r.why?.[lang] ?? r.reason ?? "")))))));
}

/** P4: 対応なしの p 値による除外の、外側の各訓練・最終モデルでの結果。 */
export function pdropView(log: readonly PDropEntry[] | undefined, t: T): HTMLElement | null {
  if (!log) return null;
  const outer = log.filter((e) => /^r\d+f\d+$/.test(e.scope)), inner = log.filter((e) => e.scope.includes(".i")), fin = log.find((e) => e.scope === "final");
  const nums = outer.map((e) => e.dropped.length);
  const mean = (xs: number[]) => (xs.length ? (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(1) : "—");
  const common = outer.length ? outer[0].dropped.map((d) => d.feature).filter((f) => outer.every((e) => e.dropped.some((d) => d.feature === f))) : [];
  const row = (e: PDropEntry) => h("tr", { "data-testid": "rs-p4-pdrop-row" },
    h("td", {}, e.scope), h("td", {}, String(e.nTrain)), h("td", {}, String(e.tested)), h("td", {}, String(e.kept)), h("td", {}, String(e.dropped.length)), h("td", {}, tests(e.tests)),
    h("td", {}, e.dropped.map((d) => d.feature).join(", ")));
  return h("div", { "data-testid": "rs-p4-pdrop" },
    h("h3", {}, t("pdropTitle")),
    h("div", { "data-testid": "rs-p4-pdrop-summary" }, t("pdropSummary", {
      n: outer.length, mean: mean(nums), min: nums.length ? Math.min(...nums) : "—", max: nums.length ? Math.max(...nums) : "—",
      ni: inner.length, meani: mean(inner.map((e) => e.dropped.length)), fin: fin ? fin.dropped.length : "—",
    })),
    common.length ? h("div", { class: "rs-note", "data-testid": "rs-p4-pdrop-common" }, `${t("pdropAll")}（${common.length}）: ${common.join(", ")}`) : null,
    h("div", { class: "rs-note" }, t("pdropNote")),
    h("div", { class: "rs-tablewrap" },
      h("table", { class: "rs-table" },
        h("thead", {}, h("tr", {}, ...[t("thScope"), "n", t("thTested"), t("thKept"), t("thDroppedCols"), t("thTests"), t("thFeatures")].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...[...outer, ...(fin ? [fin] : [])].map(row)))));
}
