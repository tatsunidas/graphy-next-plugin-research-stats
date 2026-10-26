import { CsvError, parseCsv } from "../core/csv";
import { sha256Hex } from "../core/digest";
import { datasetCsv } from "../core/exportDataset";
import { defaultFeatureColumns, type Dataset, type JoinReport, type TableSource } from "../core/dataset";
import { runRecipe, type RunRecord } from "../core/engine";
import { kmPointsCsv, r1ResultsCsv } from "../core/exportR1";
import { methodText } from "../core/methodText";
import type { Locale, ParamValue, Recipe } from "../core/recipe";
import { DATASET_CHECK, R1_PROGNOSIS, type R1State } from "../core/recipes";
import { r1Panel } from "./r1View";
import { suggestColumns } from "../core/suggest";
import { download, h } from "./dom";
import { makeT, type Key } from "./i18n";
import { CSS } from "./style";

declare const __PLUGIN_VERSION__: string;
const VERSION = typeof __PLUGIN_VERSION__ === "string" ? __PLUGIN_VERSION__ : "dev";

interface State {
  features: TableSource[];
  clinical: TableSource | null;
  baseline: TableSource[];
  followup: TableSource[];
  errors: string[]; // ファイルの読み込みエラー
  params: Record<string, ParamValue>;
  seed: number;
  noSha: boolean;
  recipe: Recipe;
  progress: { label: string; fraction: number } | null;
  cancelled: boolean;
  selectedFeature: string | null;
  result: { record: RunRecord; dataset: Dataset | null; report: JoinReport | null; r1: R1State | null } | null;
}

async function readSource(file: File): Promise<TableSource> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const sha256 = await sha256Hex(bytes);
  return { name: file.name, sha256, table: parseCsv(new TextDecoder("utf-8").decode(bytes)) };
}

/** 研究統計のウィンドウの中身を組み立てる。 */
export function mountWizard(root: HTMLElement, locale: Locale): void {
  const t = makeT(locale);
  const s: State = { features: [], clinical: null, baseline: [], followup: [], errors: [], params: { ...R1_PROGNOSIS.defaults }, seed: 1, noSha: false, recipe: DATASET_CHECK, progress: null, cancelled: false, selectedFeature: null, result: null };
  root.replaceChildren();
  const wrap = h("div", { class: "rs-root", "data-testid": "rs-root" });
  root.append(h("style", {}, CSS), wrap);

  const render = () => {
    const items: (Node | null)[] = [
      h("div", { class: "rs-banner", "data-testid": "rs-banner" }, t("banner")),
      h("h2", {}, t("step1")),
      fileRow("rs-file-features", t("featFiles"), true, async (fs) => { s.features = await load(fs); }),
      fileRow("rs-file-clinical", t("clinFile"), false, async (fs) => { s.clinical = (await load(fs))[0] ?? null; }),
      fileRow("rs-file-baseline", t("baseFiles"), true, async (fs) => { s.baseline = await load(fs); }),
      fileRow("rs-file-followup", t("followFiles"), true, async (fs) => { s.followup = await load(fs); }),
      ...s.errors.map((e) => h("div", { class: "rs-err", "data-testid": "rs-read-error" }, e)),
      ...loadedNotes(),
      s.noSha ? h("div", { class: "rs-note" }, t("sha")) : null,
      h("h2", {}, t("step2")),
      mapping(),
      recipeControls(),
      h("div", { class: "rs-row" },
        h("label", {}, t("seed")),
        h("input", { type: "number", min: 0, step: 1, value: s.seed, "data-testid": "rs-seed", onchange: (e: Event) => { s.seed = Math.max(0, Math.floor(Number((e.target as HTMLInputElement).value) || 0)); } }),
      ),
      h("div", { class: "rs-row" },
        h("button", { class: "rs-primary", "data-testid": "rs-run", disabled: !(s.features.length && s.clinical) || s.progress !== null, onclick: run }, s.recipe.id === R1_PROGNOSIS.id ? t("runR1") : t("run")),
        s.progress !== null ? h("button", { "data-testid": "rs-cancel", onclick: () => { s.cancelled = true; } }, t("cancel")) : null,
        !(s.features.length && s.clinical) ? h("span", { class: "rs-note" }, t("pickFirst")) : null,
      ),
      s.progress !== null
        ? h("div", { "data-testid": "rs-progress" },
          h("div", { class: "rs-note" }, t("running", { label: s.progress.label, pct: Math.round(s.progress.fraction * 100) })),
          h("div", { class: "rs-progress" }, h("div", { style: `width:${Math.round(s.progress.fraction * 100)}%` })))
        : null,
      s.result ? resultView(s.result) : null,
    ];
    wrap.replaceChildren(...items.filter((x): x is Node => x !== null));
  };

  async function load(files: File[]): Promise<TableSource[]> {
    const out: TableSource[] = [];
    s.errors = s.errors.filter((e) => !files.some((f) => e.startsWith(`${f.name}:`)));
    for (const f of files) {
      try {
        const src = await readSource(f);
        if (src.sha256 === null) s.noSha = true;
        out.push(src);
      } catch (e) {
        const reason = e instanceof CsvError ? e.message : String(e);
        s.errors.push(`${f.name}: ${t("readError", { name: f.name, reason })}`);
      }
    }
    s.result = null;
    // 読み直したら列の割り当ての候補も作り直す
    return out;
  }

  const loadedNotes = () =>
    [...s.features, ...(s.clinical ? [s.clinical] : []), ...s.baseline, ...s.followup].map((x) =>
      h("div", { class: "rs-note", "data-testid": "rs-loaded" }, t("loaded", { name: x.name, rows: x.table.rows.length, cols: x.table.header.length })));

  function fileRow(testId: string, label: string, multiple: boolean, onFiles: (fs: File[]) => Promise<void>): HTMLElement {
    return h("div", { class: "rs-row" },
      h("label", {}, label),
      h("input", {
        type: "file", accept: ".csv,text/csv", multiple, "data-testid": testId,
        onchange: async (e: Event) => {
          const fs = Array.from((e.target as HTMLInputElement).files ?? []);
          await onFiles(fs);
          applySuggestions();
          render();
        },
      }),
    );
  }

  function applySuggestions() {
    const sug = suggestColumns(s.features[0]?.table ?? null, s.clinical?.table ?? null);
    s.params = {
      ...s.params,
      featureIdColumn: sug.featureId || null,
      patientColumn: sug.patient,
      clinicalIdColumn: sug.clinicalId || null,
      timeColumn: sug.time,
      eventColumn: sug.event,
      outcomeColumn: sug.outcome,
      covariates: [],
    };
  }

  function select(testId: string, label: string, options: string[], value: string | null, allowNone: string | null, onPick: (v: string | null) => void): HTMLElement {
    const sel = h("select", { "data-testid": testId, onchange: (e: Event) => onPick((e.target as HTMLSelectElement).value || null) },
      allowNone !== null ? h("option", { value: "" }, allowNone) : null,
      ...options.map((o) => h("option", { value: o, selected: o === value }, o)),
    );
    sel.value = value ?? "";
    return h("div", { class: "rs-row" }, h("label", {}, label), sel);
  }

  function mapping(): HTMLElement {
    const fh = s.features[0]?.table.header ?? [];
    const ch = s.clinical?.table.header ?? [];
    const p = s.params;
    const set = (k: string) => (v: string | null) => { s.params = { ...s.params, [k]: v }; s.result = null; render(); };
    const cols = s.features.length
      ? defaultFeatureColumns({ header: fh, rows: s.features.flatMap((x) => x.table.rows) }, [p.featureIdColumn as string | null, p.patientColumn as string | null])
      : [];
    const cov = (p.covariates as string[]) ?? [];
    const used = [p.clinicalIdColumn, p.timeColumn, p.eventColumn, p.outcomeColumn].filter(Boolean) as string[];
    return h("div", {},
      select("rs-feature-id", t("featureId"), fh, p.featureIdColumn as string | null, null, set("featureIdColumn")),
      select("rs-patient", t("patient"), fh.filter((x) => x !== p.featureIdColumn), p.patientColumn as string | null, t("patientNone"), set("patientColumn")),
      select("rs-clinical-id", t("clinicalId"), ch, p.clinicalIdColumn as string | null, null, set("clinicalIdColumn")),
      select("rs-time", t("time"), ch.filter((x) => x !== p.clinicalIdColumn), p.timeColumn as string | null, t("none"), set("timeColumn")),
      select("rs-event", t("event"), ch.filter((x) => x !== p.clinicalIdColumn), p.eventColumn as string | null, t("none"), set("eventColumn")),
      select("rs-outcome", t("outcome"), ch.filter((x) => x !== p.clinicalIdColumn), p.outcomeColumn as string | null, t("none"), set("outcomeColumn")),
      h("div", { class: "rs-row" },
        h("label", {}, t("covariates")),
        h("div", { class: "rs-chips", "data-testid": "rs-covariates" },
          ...ch.filter((x) => !used.includes(x)).map((c) =>
            h("label", {}, h("input", { type: "checkbox", checked: cov.includes(c), onchange: (e: Event) => {
              const on = (e.target as HTMLInputElement).checked;
              s.params = { ...s.params, covariates: on ? [...cov, c] : cov.filter((x) => x !== c) };
              s.result = null;
            } }), ` ${c}`))),
      ),
      s.baseline.length && s.followup.length
        ? select("rs-delta-order", t("deltaOrder"), ["lesion-then-mean", "mean-then-change"], p.deltaOrder as string, null, set("deltaOrder"))
        : null,
      s.features.length ? h("div", { class: "rs-note", "data-testid": "rs-feature-count" }, t("featureCount", { n: cols.length })) : null,
    );
  }

  function numberRow(testId: string, label: string, key: string, opts: { step?: number; min?: number; max?: number; emptyIsNull?: boolean }): HTMLElement {
    const v = s.params[key];
    return h("div", { class: "rs-row" },
      h("label", {}, label),
      h("input", {
        type: "number", step: opts.step ?? "any", min: opts.min, max: opts.max, value: v === null || v === undefined ? "" : String(v), "data-testid": testId,
        onchange: (e: Event) => {
          const raw = (e.target as HTMLInputElement).value.trim();
          s.params = { ...s.params, [key]: raw === "" ? (opts.emptyIsNull ? null : NaN) : Number(raw) };
          s.result = null;
          render();
        },
      }));
  }

  function recipeControls(): HTMLElement {
    const p = s.params;
    const pick = (id: string) => { s.recipe = id === R1_PROGNOSIS.id ? R1_PROGNOSIS : DATASET_CHECK; s.result = null; render(); };
    const sel = h("select", { "data-testid": "rs-recipe", onchange: (e: Event) => pick((e.target as HTMLSelectElement).value) },
      h("option", { value: DATASET_CHECK.id }, t("recipeCheck")), h("option", { value: R1_PROGNOSIS.id }, t("recipeR1")));
    sel.value = s.recipe.id;
    const out: Node[] = [h("div", { class: "rs-row" }, h("label", {}, t("recipe")), sel)];
    if (s.recipe.id !== R1_PROGNOSIS.id) return h("div", {}, ...out);
    const choice = (testId: string, label: string, key: string, options: [string, string][]) => {
      const el = h("select", { "data-testid": testId, onchange: (e: Event) => { s.params = { ...s.params, [key]: (e.target as HTMLSelectElement).value }; s.result = null; render(); } },
        ...options.map(([v, l]) => h("option", { value: v }, l)));
      el.value = String(p[key]);
      return h("div", { class: "rs-row" }, h("label", {}, label), el);
    };
    out.push(
      choice("rs-group-mode", t("groupMode"), "groupMode", [["median", t("gMedian")], ["value", t("gValue")], ["tertile", t("gTertile")], ["optimal", t("gOptimal")]]),
      p.groupMode === "value" ? numberRow("rs-group-value", t("groupValue"), "groupValue", { emptyIsNull: true }) : (null as unknown as HTMLElement),
      choice("rs-multiplicity", t("multiplicity"), "multiplicity", [["bh", t("mBh")], ["none", t("mNone")]]),
      numberRow("rs-alpha", t("alpha"), "alpha", { step: 0.01, min: 0.0001, max: 0.5 }),
      p.groupMode === "optimal" ? numberRow("rs-min-fraction", t("minFraction"), "minGroupFraction", { step: 0.05, min: 0.01, max: 0.4 }) : (null as unknown as HTMLElement),
      p.groupMode === "optimal" ? numberRow("rs-permutations", t("permutations"), "permutations", { step: 100, min: 19 }) : (null as unknown as HTMLElement),
      numberRow("rs-rmst-tau", t("rmstTau"), "rmstTau", { emptyIsNull: true, min: 0 }),
      h("div", { class: "rs-row" }, h("label", {}, t("mw")), h("input", { type: "checkbox", checked: p.mannWhitney === true, "data-testid": "rs-mw", onchange: (e: Event) => { s.params = { ...s.params, mannWhitney: (e.target as HTMLInputElement).checked }; s.result = null; } })),
      numberRow("rs-bootstrap", t("bootstrapB"), "bootstrapB", { step: 100, min: 0 }),
    );
    if (p.groupMode === "optimal") {
      const m = defaultFeatureColumns({ header: s.features[0]?.table.header ?? [], rows: s.features.flatMap((x) => x.table.rows) }, [p.featureIdColumn as string | null, p.patientColumn as string | null]).length;
      if (m) out.push(h("div", { class: "rs-note", "data-testid": "rs-heavy" }, t("heavy", { m, b: Number(p.permutations) })));
    }
    return h("div", {}, ...out.filter((x) => x !== null));
  }

  async function run() {
    if (!s.clinical || !s.features.length || s.progress !== null) return;
    const recipe = s.recipe;
    const params: Record<string, ParamValue> = { ...s.params };
    for (const k of Object.keys(params)) if (!(k in recipe.defaults)) delete params[k];
    s.cancelled = false;
    s.progress = { label: "", fraction: 0 };
    s.result = null;
    render();
    try {
      const { record, state } = await runRecipe(
        recipe,
        { features: s.features, clinical: s.clinical, baseline: s.baseline, followup: s.followup },
        params,
        {
          seed: s.seed, pluginVersion: VERSION, now: () => new Date().toISOString(),
          isCancelled: () => s.cancelled,
          onProgress: (_id, label, fraction) => { s.progress = { label, fraction }; updateProgress(); },
        },
      );
      const r1 = (state.r1 as R1State | undefined) ?? null;
      s.selectedFeature = null;
      s.result = { record, dataset: (state.dataset as Dataset | undefined) ?? null, report: (state.report as JoinReport | undefined) ?? null, r1 };
    } catch (e) {
      s.errors.push(String(e instanceof Error ? e.message : e));
    }
    s.progress = null;
    render();
  }

  /** 進み具合だけを書き換える（全体を描き直すと、実行中に画面が固まる・選択が飛ぶ）。 */
  function updateProgress() {
    const box = wrap.querySelector('[data-testid="rs-progress"]');
    if (!box || !s.progress) return;
    const pct = Math.round(s.progress.fraction * 100);
    const note = box.querySelector(".rs-note");
    const bar = box.querySelector(".rs-progress > div") as HTMLElement | null;
    if (note) note.textContent = t("running", { label: s.progress.label, pct });
    if (bar) bar.style.width = `${pct}%`;
  }

  function resultView(r: NonNullable<State["result"]>): HTMLElement {
    const failed = r.record.steps.flatMap((x) => x.errors);
    const warns = r.record.steps.flatMap((x) => x.warnings);
    const rep = r.report;
    const list = (title: string, xs: string[], testId: string) =>
      xs.length ? h("details", { "data-testid": testId }, h("summary", {}, `${title}（${xs.length}）`), h("ul", {}, ...xs.map((x) => h("li", {}, x)))) : null;
    return h("div", { "data-testid": "rs-result" },
      h("h2", {}, t("step3")),
      failed.length ? h("div", { class: "rs-err", "data-testid": "rs-errors" }, h("b", {}, t("errors")), h("ul", {}, ...failed.map((e) => h("li", {}, e)))) : null,
      warns.length ? h("div", { class: "rs-warn", "data-testid": "rs-warnings" }, h("b", {}, t("warnings")), h("ul", {}, ...warns.map((e) => h("li", {}, e)))) : null,
      rep ? h("div", {},
        h("div", { class: "rs-big", "data-testid": "rs-joined" }, t("joined", { n: rep.joined })),
        h("div", { "data-testid": "rs-counts" }, t("counts", { rows: rep.featureRows, pf: rep.patientsInFeatures, pc: rep.patientsInClinical, fo: rep.featureOnly.length, co: rep.clinicalOnly.length })),
        rep.survivalUsable !== null ? h("div", { "data-testid": "rs-survival" }, t("survival", { n: rep.survivalUsable })) : null,
        rep.lesionsPerPatient ? h("div", { class: "rs-note" }, t("lesions", rep.lesionsPerPatient)) : null,
        list(t("featureOnly"), rep.featureOnly, "rs-feature-only"),
        list(t("clinicalOnly"), rep.clinicalOnly, "rs-clinical-only"),
        rep.missing.length
          ? list(t("missing"), rep.missing.map((m) => t("missingRow", { name: m.name, count: m.count, of: m.of })), "rs-missing")
          : h("div", { class: "rs-note", "data-testid": "rs-no-missing" }, t("noMissing")),
        list(t("exclusions"), rep.exclusions.map((x) => `${x.id}: ${x.reason}`), "rs-exclusions"),
      ) : null,
      r.r1 && r.r1.rows.length ? r1Panel(r.r1, t, s.selectedFeature, (name) => { s.selectedFeature = name; render(); }) : null,
      h("h2", {}, t("step4")),
      h("div", { class: "rs-row" },
        h("button", { "data-testid": "rs-save-data", disabled: !r.dataset, onclick: () => r.dataset && download(datasetCsv(r.dataset, (s.params.clinicalIdColumn as string) ?? "ID"), "research-stats-dataset.csv", "text/csv;charset=utf-8") }, t("saveData")),
        r.r1 && r.r1.rows.length ? h("button", { "data-testid": "rs-save-results", onclick: () => download(r1ResultsCsv(r.r1!.rows), "research-stats-r1-results.csv", "text/csv;charset=utf-8") }, t("saveResults")) : null,
        r.r1 && r.r1.rows.length ? h("button", { "data-testid": "rs-save-km", onclick: () => download(kmPointsCsv(r.r1!.context, r.r1!.rows), "research-stats-km-points.csv", "text/csv;charset=utf-8") }, t("saveKm")) : null,
        h("button", { "data-testid": "rs-save-method", onclick: () => download(methodText(r.record, s.recipe, locale), "research-stats-method.txt", "text/plain;charset=utf-8") }, t("saveMethod")),
        h("button", { "data-testid": "rs-save-record", onclick: () => download(`${JSON.stringify(r.record, null, 2)}\n`, "research-stats-record.json", "application/json") }, t("saveRecord")),
      ),
    );
  }

  render();
}

export type { Key };
