import { CsvError, parseCsv } from "../core/csv";
import { sha256Hex } from "../core/digest";
import { datasetCsv } from "../core/exportDataset";
import { defaultFeatureColumns, type Dataset, type JoinReport, type TableSource } from "../core/dataset";
import { runSkill, type RunRecord } from "../core/engine";
import { kmPointsCsv, r1ResultsCsv } from "../core/exportR1";
import { methodText } from "../core/methodText";
import type { Locale, ParamValue, Skill } from "../core/skill";
import { DATASET_CHECK, P4_PIPELINE, R1_PROGNOSIS, R2_DISCRIMINATION, R3_REPRODUCIBILITY, R5_MODEL, type P4State, type R1State, type R2State, type R3State, type R5State } from "../core/skills";
import { R5_DEFAULTS } from "../core/r5";
import { r5ModelCsv, r5SchoenfeldCsv, r5ScreenCsv, r5StepsCsv } from "../core/exportR5";
import { r5Panel } from "./r5View";
import { P4_DEFAULTS } from "../core/p4";
import { p4FoldsCsv, p4PredictionsCsv, p4QcCsv, p4SelectionCsv } from "../core/exportP4";
import { p4Panel } from "./p4View";
import { R3_DEFAULTS } from "../core/r3";
import { blandAltmanCsv, exclusionCsv, r3ResultsCsv } from "../core/exportR3";
import { blandAltmanFor } from "../core/r3";
import { r3Panel, type BaChoice } from "./r3View";
import { R1_DEFAULTS } from "../core/r1";
import { R2_DEFAULTS } from "../core/r2";
import { r2CurvesCsv, r2ResultsCsv, treeCsv } from "../core/exportR2";
import { r2Panel } from "./r2View";
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
  /** P4: 品質管理で除く特徴量の一覧と、テスト集合の固定ファイル。 */
  exclusion: TableSource | null;
  lock: { name: string; text: string } | null;
  errors: string[]; // ファイルの読み込みエラー
  params: Record<string, ParamValue>;
  seed: number;
  noSha: boolean;
  skill: Skill;
  progress: { label: string; fraction: number } | null;
  cancelled: boolean;
  selectedFeature: string | null;
  ba: BaChoice;
  result: { record: RunRecord; dataset: Dataset | null; report: JoinReport | null; r1: R1State | null; r2: R2State | null; r3: R3State | null; p4: P4State | null; r5: R5State | null } | null;
}

async function readSource(file: File): Promise<TableSource> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const sha256 = await sha256Hex(bytes);
  return { name: file.name, sha256, table: parseCsv(new TextDecoder("utf-8").decode(bytes)) };
}

/** 研究統計のウィンドウの中身を組み立てる。 */
export function mountWizard(root: HTMLElement, locale: Locale): void {
  const t = makeT(locale);
  const s: State = { features: [], clinical: null, baseline: [], followup: [], exclusion: null, lock: null, errors: [], params: { ...DATASET_CHECK.defaults, ...R1_DEFAULTS, ...R2_DEFAULTS, ...R3_DEFAULTS, ...P4_DEFAULTS, ...R5_DEFAULTS }, seed: 1, noSha: false, skill: DATASET_CHECK, progress: null, cancelled: false, selectedFeature: null, ba: { feature: null, a: 0, b: 1 }, result: null };
  root.replaceChildren();
  const wrap = h("div", { class: "rs-root", "data-testid": "rs-root" });
  root.append(h("style", {}, CSS), wrap);

  const render = () => {
    const isR3 = s.skill.id === R3_REPRODUCIBILITY.id, isP4 = s.skill.id === P4_PIPELINE.id;
    const canRun = isR3 ? s.features.length >= 2 : !!(s.features.length && s.clinical);
    const items: (Node | null)[] = [
      h("div", { class: "rs-banner", "data-testid": "rs-banner" }, t("banner")),
      h("h2", {}, t("step1")),
      fileRow("rs-file-features", isR3 ? t("condFiles") : t("featFiles"), true, async (fs) => { s.features = await load(fs); }),
      isR3 ? null : fileRow("rs-file-clinical", t("clinFile"), false, async (fs) => { s.clinical = (await load(fs))[0] ?? null; }),
      isR3 ? null : fileRow("rs-file-baseline", t("baseFiles"), true, async (fs) => { s.baseline = await load(fs); }),
      isR3 ? null : fileRow("rs-file-followup", t("followFiles"), true, async (fs) => { s.followup = await load(fs); }),
      isP4 ? fileRow("rs-file-exclusion", t("exclFile"), false, async (fs) => { s.exclusion = (await load(fs))[0] ?? null; }) : null,
      isP4 && s.params.lockMode === "use" ? fileRow("rs-file-lock", t("lockFile"), false, async (fs) => {
        const f = fs[0];
        s.lock = f ? { name: f.name, text: await f.text() } : null;
        s.result = null;
      }, ".json,application/json") : null,
      isP4 && s.lock && s.params.lockMode === "use" ? h("div", { class: "rs-note", "data-testid": "rs-lock-loaded" }, s.lock.name) : null,
      ...s.errors.map((e) => h("div", { class: "rs-err", "data-testid": "rs-read-error" }, e)),
      ...loadedNotes(),
      s.noSha ? h("div", { class: "rs-note" }, t("sha")) : null,
      h("h2", {}, t("step2")),
      isR3 ? r3Mapping() : mapping(),
      skillControls(),
      h("div", { class: "rs-row" },
        h("label", {}, t("seed")),
        h("input", { type: "number", min: 0, step: 1, value: s.seed, "data-testid": "rs-seed", onchange: (e: Event) => { s.seed = Math.max(0, Math.floor(Number((e.target as HTMLInputElement).value) || 0)); } }),
      ),
      h("div", { class: "rs-row" },
        h("button", { class: "rs-primary", "data-testid": "rs-run", disabled: !canRun || s.progress !== null, onclick: () => run(false) }, s.skill.id === DATASET_CHECK.id ? t("run") : t("runR1")),
        s.progress !== null ? h("button", { "data-testid": "rs-cancel", onclick: () => { s.cancelled = true; } }, t("cancel")) : null,
        !canRun ? h("span", { class: "rs-note" }, t("pickFirst")) : null,
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

  function fileRow(testId: string, label: string, multiple: boolean, onFiles: (fs: File[]) => Promise<void>, accept = ".csv,text/csv"): HTMLElement {
    return h("div", { class: "rs-row" },
      h("label", {}, label),
      h("input", {
        type: "file", accept, multiple, "data-testid": testId,
        onchange: async (e: Event) => {
          const fs = Array.from((e.target as HTMLInputElement).files ?? []);
          await onFiles(fs);
          if (accept.includes(".csv") && testId !== "rs-file-exclusion") applySuggestions();
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
      reliabilityIdColumn: sug.featureId || null,
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

  /** 結果ラベルの列にある値（陽性とするラベルの選択肢）。 */
  function outcomeValues(): string[] {
    const col = s.params.outcomeColumn as string | null;
    const tb = s.clinical?.table;
    if (!col || !tb) return [];
    const j = tb.header.indexOf(col);
    if (j < 0) return [];
    return [...new Set(tb.rows.map((r) => String(r[j] ?? "").trim()).filter((x) => x !== ""))].sort();
  }

  function r2Controls(): HTMLElement[] {
    const p = s.params;
    const labels = outcomeValues();
    const pos = h("select", { "data-testid": "rs-positive", onchange: (e: Event) => { s.params = { ...s.params, positiveLabel: (e.target as HTMLSelectElement).value || null }; s.result = null; } },
      h("option", { value: "" }, t("positiveNone")), ...labels.map((l) => h("option", { value: l }, l)));
    pos.value = typeof p.positiveLabel === "string" && labels.includes(p.positiveLabel) ? p.positiveLabel : "";
    if (pos.value === "" && p.positiveLabel !== null) s.params = { ...s.params, positiveLabel: null };
    const choice = (testId: string, label: string, key: string, options: [string, string][]) => {
      const el = h("select", { "data-testid": testId, onchange: (e: Event) => { s.params = { ...s.params, [key]: (e.target as HTMLSelectElement).value }; s.result = null; render(); } }, ...options.map(([v, l]) => h("option", { value: v }, l)));
      el.value = String(p[key]);
      return h("div", { class: "rs-row" }, h("label", {}, label), el);
    };
    const out: HTMLElement[] = [
      h("div", { class: "rs-row" }, h("label", {}, t("positiveLabel")), pos),
      choice("rs-criterion", t("criterion"), "criterion", [["accuracy", t("cAccuracy")], ["youden", t("cYouden")]]),
      choice("rs-multiplicity", t("multiplicity"), "multiplicity", [["bh", t("mBh")], ["none", t("mNone")]]),
      numberRow("rs-alpha", t("alpha"), "alpha", { step: 0.01, min: 0.0001, max: 0.5 }),
      numberRow("rs-topk", t("topK"), "topK", { step: 1, min: 2, max: 20 }),
      numberRow("rs-calbins", t("calBins"), "calibrationBins", { step: 1, min: 2, max: 50 }),
      h("div", { class: "rs-row" }, h("label", {}, t("treeOn")), h("input", { type: "checkbox", checked: p.tree === true, "data-testid": "rs-tree-on", onchange: (e: Event) => { s.params = { ...s.params, tree: (e.target as HTMLInputElement).checked }; s.result = null; render(); } })),
    ];
    if (p.tree === true) out.push(
      numberRow("rs-tree-stop", t("treeStop"), "treeStopRate", { step: 0.01, min: 0, max: 0.5 }),
      numberRow("rs-tree-min", t("treeMin"), "treeMinNode", { step: 1, min: 2 }),
      numberRow("rs-tree-depth", t("treeDepth"), "treeMaxDepth", { step: 1, min: 1, max: 20 }),
      numberRow("rs-mc", t("mcIter"), "mcIterations", { step: 100, min: 100 }),
    );
    return out;
  }

  function r3Mapping(): HTMLElement {
    const fh = s.features[0]?.table.header ?? [];
    return h("div", {}, select("rs-r3-id", t("condId"), fh, s.params.reliabilityIdColumn as string | null, null, (v) => { s.params = { ...s.params, reliabilityIdColumn: v }; s.result = null; render(); }));
  }

  function r3Controls(): HTMLElement[] {
    const el = h("select", { "data-testid": "rs-min-category", onchange: (e: Event) => { s.params = { ...s.params, minCategory: (e.target as HTMLSelectElement).value || null }; s.result = null; } },
      h("option", { value: "" }, t("mcNone")), ...["moderate", "good", "excellent"].map((c) => h("option", { value: c }, c)));
    el.value = typeof s.params.minCategory === "string" ? s.params.minCategory : "";
    return [h("div", { class: "rs-row" }, h("label", {}, t("minCategory")), el)];
  }

  function p4Controls(): HTMLElement[] {
    const p = s.params;
    const choice = (testId: string, label: string, key: string, options: [string, string][], toValue: (v: string) => ParamValue = (v) => v) => {
      const el = h("select", { "data-testid": testId, onchange: (e: Event) => { const v = (e.target as HTMLSelectElement).value; s.params = { ...s.params, [key]: v === "" ? null : toValue(v) }; s.result = null; render(); } },
        h("option", { value: "" }, t("choose")), ...options.map(([v, l]) => h("option", { value: v }, l)));
      el.value = p[key] === null || p[key] === undefined ? "" : String(p[key]);
      return h("div", { class: "rs-row" }, h("label", {}, label), el);
    };
    const out: (HTMLElement | null)[] = [
      choice("rs-p4-kind", t("p4Kind"), "outcomeKind", [["binary", t("kBinary")], ["survival", t("kSurvival")]]),
    ];
    if (p.outcomeKind === "binary") {
      const labels = outcomeValues();
      const pos = h("select", { "data-testid": "rs-positive", onchange: (e: Event) => { s.params = { ...s.params, positiveLabel: (e.target as HTMLSelectElement).value || null }; s.result = null; } },
        h("option", { value: "" }, t("positiveNone")), ...labels.map((l) => h("option", { value: l }, l)));
      pos.value = typeof p.positiveLabel === "string" && labels.includes(p.positiveLabel) ? p.positiveLabel : "";
      out.push(h("div", { class: "rs-row" }, h("label", {}, t("positiveLabel")), pos));
    }
    if (p.outcomeKind === "survival") {
      out.push(choice("rs-p4-strat", t("p4Strat"), "stratifySurvival", [["true", t("stratYes")], ["false", t("stratNo")]], (v) => v === "true"));
      out.push(numberRow("rs-p4-boot", t("p4Boot"), "testBootstrapB", { step: 100, min: 100 }));
    }
    out.push(
      choice("rs-p4-lock", t("p4Lock"), "lockMode", [["create", t("lockCreate")], ["use", t("lockUse")], ["none", t("lockNone")]]),
      p.lockMode === "create" ? numberRow("rs-p4-frac", t("p4Frac"), "testFraction", { step: 0.05, min: 0.05, max: 0.95, emptyIsNull: true }) : null,
      numberRow("rs-p4-outer", t("p4Outer"), "outerK", { step: 1, min: 2, emptyIsNull: true }),
      numberRow("rs-p4-inner", t("p4Inner"), "innerK", { step: 1, min: 2, emptyIsNull: true }),
      numberRow("rs-p4-repeats", t("p4Repeats"), "repeats", { step: 1, min: 1, max: 100, emptyIsNull: true }),
      h("div", { class: "rs-row" }, h("label", {}, t("p4K")), h("input", {
        type: "text", value: typeof p.kCandidates === "string" ? p.kCandidates : "", placeholder: "1, 2, 4", "data-testid": "rs-p4-k",
        onchange: (e: Event) => { const v = (e.target as HTMLInputElement).value.trim(); s.params = { ...s.params, kCandidates: v === "" ? null : v }; s.result = null; },
      })),
      h("div", { class: "rs-note", "data-testid": "rs-p4-examples" }, t("p4Examples")),
      h("div", { class: "rs-row" }, h("label", {}, t("p4Const")), h("input", { type: "checkbox", checked: p.dropConstant === true, "data-testid": "rs-p4-const", onchange: (e: Event) => { s.params = { ...s.params, dropConstant: (e.target as HTMLInputElement).checked }; s.result = null; } })),
      numberRow("rs-p4-corr", t("p4Corr"), "correlationThreshold", { step: 0.01, min: 0.5, max: 1, emptyIsNull: true }),
      choice("rs-p4-missing", t("p4Missing"), "missing", [["complete", t("mComplete")], ["impute", t("mImpute")]]),
    );
    return out.filter((x): x is HTMLElement => x !== null);
  }

  function r5Controls(): HTMLElement[] {
    const p = s.params;
    const choice = (testId: string, label: string, key: string, options: [string, string][], allowNone: boolean) => {
      const el = h("select", { "data-testid": testId, onchange: (e: Event) => { const v = (e.target as HTMLSelectElement).value; s.params = { ...s.params, [key]: v === "" ? null : v }; s.result = null; render(); } },
        allowNone ? h("option", { value: "" }, t("choose")) : null, ...options.map(([v, l]) => h("option", { value: v }, l)));
      el.value = p[key] === null || p[key] === undefined ? "" : String(p[key]);
      return h("div", { class: "rs-row" }, h("label", {}, label), el);
    };
    const boost = p.r5Screen === "boost";
    const out: (HTMLElement | null)[] = [
      numberRow("rs-r5-corr", t("r5Corr"), "r5Corr", { step: 0.01, min: 0.5, max: 1, emptyIsNull: true }),
      choice("rs-r5-screen", t("r5Screen"), "r5Screen", [["boost", t("sBoost")], ["lasso", t("sLasso")]], false),
      boost ? numberRow("rs-r5-iter", t("r5BoostIter"), "r5BoostIterations", { step: 10, min: 1 }) : null,
      boost ? numberRow("rs-r5-nu", t("r5Nu"), "r5BoostNu", { step: 0.01, min: 0.000001, max: 1, emptyIsNull: true }) : null,
      boost ? null : numberRow("rs-r5-folds", t("r5LassoFolds"), "r5LassoFolds", { step: 1, min: 2 }),
      boost ? null : choice("rs-r5-rule", t("r5LassoRule"), "r5LassoRule", [["min", t("ruleMin")], ["1se", t("rule1se")]], true),
      boost ? null : numberRow("rs-r5-nlambda", t("r5NLambda"), "r5LassoNLambda", { step: 10, min: 2 }),
      numberRow("rs-r5-pin", t("r5PIn"), "r5PIn", { step: 0.01, min: 0.000001, max: 0.5 }),
      numberRow("rs-r5-pout", t("r5POut"), "r5POut", { step: 0.01, min: 0.000001, max: 0.5 }),
      numberRow("rs-r5-max", t("r5MaxVars"), "r5MaxVars", { step: 1, min: 1 }),
      numberRow("rs-r5-boot", t("r5Boot"), "r5BootstrapB", { step: 100, min: 0 }),
      h("div", { class: "rs-note", "data-testid": "rs-r5-examples" }, t("r5Examples")),
      Number(p.r5BootstrapB) > 0 ? h("div", { class: "rs-note", "data-testid": "rs-r5-heavy" }, t("r5Heavy")) : null,
    ];
    return out.filter((x): x is HTMLElement => x !== null);
  }

  function skillControls(): HTMLElement {
    const p = s.params;
    const pick = (id: string) => { s.skill = [DATASET_CHECK, R1_PROGNOSIS, R2_DISCRIMINATION, R3_REPRODUCIBILITY, P4_PIPELINE, R5_MODEL].find((r) => r.id === id) ?? DATASET_CHECK; s.result = null; s.selectedFeature = null; render(); };
    const sel = h("select", { "data-testid": "rs-skill", onchange: (e: Event) => pick((e.target as HTMLSelectElement).value) },
      h("option", { value: DATASET_CHECK.id }, t("skillCheck")), h("option", { value: R1_PROGNOSIS.id }, t("skillR1")), h("option", { value: R2_DISCRIMINATION.id }, t("skillR2")), h("option", { value: R3_REPRODUCIBILITY.id }, t("skillR3")), h("option", { value: P4_PIPELINE.id }, t("skillP4")), h("option", { value: R5_MODEL.id }, t("skillR5")));
    sel.value = s.skill.id;
    const out: Node[] = [h("div", { class: "rs-row" }, h("label", {}, t("skill")), sel)];
    if (s.skill.id === R3_REPRODUCIBILITY.id) return h("div", {}, ...out, ...r3Controls());
    if (s.skill.id === P4_PIPELINE.id) return h("div", {}, ...out, ...p4Controls());
    if (s.skill.id === R5_MODEL.id) return h("div", {}, ...out, ...r5Controls());
    if (s.skill.id === R2_DISCRIMINATION.id) return h("div", {}, ...out, ...r2Controls());
    if (s.skill.id !== R1_PROGNOSIS.id) return h("div", {}, ...out);
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

  async function run(evaluate = false) {
    const needsClinical = s.skill.id !== R3_REPRODUCIBILITY.id;
    const isP4 = s.skill.id === P4_PIPELINE.id;
    if ((needsClinical && !s.clinical) || !s.features.length || s.progress !== null) return;
    const skill = s.skill;
    const params: Record<string, ParamValue> = { ...s.params };
    for (const k of Object.keys(params)) if (!(k in skill.defaults)) delete params[k];
    if (isP4) params.evaluateTest = evaluate;
    s.cancelled = false;
    s.progress = { label: "", fraction: 0 };
    s.result = null;
    render();
    try {
      const { record, state } = await runSkill(
        skill,
        {
          features: s.features, clinical: needsClinical ? (s.clinical ?? undefined) : undefined, baseline: s.baseline, followup: s.followup,
          exclusion: isP4 ? (s.exclusion ?? undefined) : undefined, lock: isP4 && s.params.lockMode === "use" ? (s.lock ?? undefined) : undefined,
        },
        params,
        {
          seed: s.seed, pluginVersion: VERSION, now: () => new Date().toISOString(),
          isCancelled: () => s.cancelled,
          onProgress: (_id, label, fraction) => { s.progress = { label, fraction }; updateProgress(); },
        },
      );
      const r1 = (state.r1 as R1State | undefined) ?? null;
      const r2 = (state.r2 as R2State | undefined) ?? null;
      const r3 = (state.r3 as R3State | undefined) ?? null;
      const p4 = (state.p4 as P4State | undefined) ?? null;
      const r5 = (state.r5 as R5State | undefined) ?? null;
      s.selectedFeature = null;
      s.result = { record, dataset: (state.dataset as Dataset | undefined) ?? null, report: (state.report as JoinReport | undefined) ?? null, r1, r2, r3, p4, r5 };
      // 評価したら、回数を 1 増やした固定ファイルを今の固定ファイルにして保存する（次の評価は 2 回目として数える）
      if (p4?.test) {
        s.lock = { name: s.lock?.name ?? "research-stats-test-lock.json", text: `${JSON.stringify(p4.test.lock, null, 2)}\n` };
        download(s.lock.text, "research-stats-test-lock.json", "application/json");
      }
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
      r.record.skillId === R3_REPRODUCIBILITY.id ? null : h("h2", {}, t("step3")),
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
      r.p4 ? p4Panel(r.p4, t) : null,
      r.r5 ? r5Panel(r.r5, t) : null,
      r.p4 && r.p4.final?.model && r.p4.params.lockMode === "use" && !r.p4.test && r.record.status === "ok"
        ? h("div", { "data-testid": "rs-p4-eval-box" },
          h("button", { class: "rs-primary", "data-testid": "rs-p4-evaluate", disabled: s.progress !== null, onclick: () => run(true) }, t("p4Evaluate")),
          h("div", { class: "rs-note" }, t("p4EvaluateNote")))
        : null,
      r.r3 && r.r3.rows.length ? r3Panel(r.r3, t, s.ba, (c) => { s.ba = c; render(); }) : null,
      r.r2 && r.r2.rows.length ? r2Panel(r.r2, t, s.selectedFeature, (name) => { s.selectedFeature = name; render(); }) : null,
      r.r1 && r.r1.rows.length ? r1Panel(r.r1, t, s.selectedFeature, (name) => { s.selectedFeature = name; render(); }) : null,
      h("h2", {}, t("step4")),
      h("div", { class: "rs-row" },
        r.record.skillId === R3_REPRODUCIBILITY.id ? null : h("button", { "data-testid": "rs-save-data", disabled: !r.dataset, onclick: () => r.dataset && download(datasetCsv(r.dataset, (s.params.clinicalIdColumn as string) ?? "ID"), "research-stats-dataset.csv", "text/csv;charset=utf-8") }, t("saveData")),
        r.r1 && r.r1.rows.length ? h("button", { "data-testid": "rs-save-results", onclick: () => download(r1ResultsCsv(r.r1!.rows), "research-stats-r1-results.csv", "text/csv;charset=utf-8") }, t("saveResults")) : null,
        r.r1 && r.r1.rows.length ? h("button", { "data-testid": "rs-save-km", onclick: () => download(kmPointsCsv(r.r1!.context, r.r1!.rows), "research-stats-km-points.csv", "text/csv;charset=utf-8") }, t("saveKm")) : null,
        r.r3 && r.r3.rows.length ? h("button", { "data-testid": "rs-save-r3-results", onclick: () => download(r3ResultsCsv(r.r3!.rows), "research-stats-r3-icc.csv", "text/csv;charset=utf-8") }, t("saveR3Results")) : null,
        r.r3 && r.r3.params.minCategory ? h("button", { "data-testid": "rs-save-exclusion", onclick: () => download(exclusionCsv(r.r3!.candidates, r.r3!.params.minCategory), "research-stats-r3-exclusion-candidates.csv", "text/csv;charset=utf-8") }, t("saveExclusion")) : null,
        r.r3 && r.r3.rows.length ? h("button", { "data-testid": "rs-save-ba", onclick: () => {
          const d = r.r3!.data, f = s.ba.feature && d.features.includes(s.ba.feature) ? s.ba.feature : d.features[0], k = d.conditions.length;
          const a = Math.min(s.ba.a, k - 1), b = s.ba.b === a ? (a + 1) % k : Math.min(s.ba.b, k - 1);
          const res = blandAltmanFor(d, f, a, b);
          if (!("error" in res)) download(blandAltmanCsv(res, f, d.conditions[a], d.conditions[b]), "research-stats-r3-bland-altman.csv", "text/csv;charset=utf-8");
        } }, t("saveBa")) : null,
        r.p4?.split?.lock ? h("button", { "data-testid": "rs-save-lock", onclick: () => download(`${JSON.stringify(r.p4!.test?.lock ?? r.p4!.split!.lock, null, 2)}\n`, "research-stats-test-lock.json", "application/json") }, t("saveLock")) : null,
        r.p4?.cv ? h("button", { "data-testid": "rs-save-p4-folds", onclick: () => download(p4FoldsCsv(r.p4!), "research-stats-p4-folds.csv", "text/csv;charset=utf-8") }, t("saveP4Folds")) : null,
        r.p4?.cv ? h("button", { "data-testid": "rs-save-p4-pred", onclick: () => download(p4PredictionsCsv(r.p4!), "research-stats-p4-predictions.csv", "text/csv;charset=utf-8") }, t("saveP4Pred")) : null,
        r.p4?.cv ? h("button", { "data-testid": "rs-save-p4-sel", onclick: () => download(p4SelectionCsv(r.p4!), "research-stats-p4-selection.csv", "text/csv;charset=utf-8") }, t("saveP4Sel")) : null,
        r.p4?.qc ? h("button", { "data-testid": "rs-save-p4-qc", onclick: () => download(p4QcCsv(r.p4!), "research-stats-p4-qc.csv", "text/csv;charset=utf-8") }, t("saveP4Qc")) : null,
        r.r5?.model ? h("button", { "data-testid": "rs-save-r5-model", onclick: () => download(r5ModelCsv(r.r5!), "research-stats-r5-model.csv", "text/csv;charset=utf-8") }, t("saveR5Model")) : null,
        r.r5?.model ? h("button", { "data-testid": "rs-save-r5-steps", onclick: () => download(r5StepsCsv(r.r5!), "research-stats-r5-steps.csv", "text/csv;charset=utf-8") }, t("saveR5Steps")) : null,
        r.r5?.model ? h("button", { "data-testid": "rs-save-r5-screen", onclick: () => download(r5ScreenCsv(r.r5!), "research-stats-r5-screening.csv", "text/csv;charset=utf-8") }, t("saveR5Screen")) : null,
        r.r5?.model?.schoenfeld ? h("button", { "data-testid": "rs-save-r5-sch", onclick: () => download(r5SchoenfeldCsv(r.r5!), "research-stats-r5-schoenfeld.csv", "text/csv;charset=utf-8") }, t("saveR5Sch")) : null,
        r.r2 && r.r2.rows.length ? h("button", { "data-testid": "rs-save-r2-results", onclick: () => download(r2ResultsCsv(r.r2!.rows), "research-stats-r2-results.csv", "text/csv;charset=utf-8") }, t("saveR2Results")) : null,
        r.r2?.combo && r.r2.combo.rocLoo.length ? h("button", { "data-testid": "rs-save-curves", onclick: () => download(r2CurvesCsv(r.r2!.combo!), "research-stats-r2-curves.csv", "text/csv;charset=utf-8") }, t("saveCurves")) : null,
        r.r2?.tree ? h("button", { "data-testid": "rs-save-tree", onclick: () => download(treeCsv(r.r2!.tree!.root, r.r2!.context.names), "research-stats-r2-tree.csv", "text/csv;charset=utf-8") }, t("saveTree")) : null,
        h("button", { "data-testid": "rs-save-method", onclick: () => download(methodText(r.record, s.skill, locale), "research-stats-method.txt", "text/plain;charset=utf-8") }, t("saveMethod")),
        h("button", { "data-testid": "rs-save-record", onclick: () => download(`${JSON.stringify(r.record, null, 2)}\n`, "research-stats-record.json", "application/json") }, t("saveRecord")),
      ),
    );
  }

  render();
}

export type { Key };
