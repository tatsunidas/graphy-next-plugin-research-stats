// @vitest-environment jsdom
import { Blob as NodeBlob } from "node:buffer";
import { beforeEach, describe, expect, it } from "vitest";
import { mountWizard } from "../src/ui/wizard";

// 臨床 40 人。時間・イベント・結果ラベル（2 値）・共変量を持つ。
const N = 40;
const ids = Array.from({ length: N }, (_, i) => `P${String(i + 1).padStart(3, "0")}`);
const clinicalCsv = (key: string, idList: string[]) =>
  `${key},OS_months,Event,Outcome,age\n` + idList.map((id, i) => `${id},${3 + ((i * 7) % 29)},${i % 3 === 0 ? 0 : 1},${i % 2 ? "resp" : "nonresp"},${40 + (i % 30)}`).join("\n") + "\n";
// 病変ごとの行（患者あたり 2 行）。特徴量は患者ごとにずらした値。
const featureCsv = (patientKey: string, idList: string[]) =>
  `CaseLabel,${patientKey},f1,f2\n` + idList.flatMap((id, i) => [0, 1].map((k) => `${id}_L${k},${id},${(1 + ((i * 13 + k * 5) % 17)) / 3},${(2 + ((i * 7 + k) % 11)) / 7}`)).join("\n") + "\n";

let root: HTMLElement;
const q = (id: string) => root.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
const all = (id: string) => Array.from(root.querySelectorAll(`[data-testid="${id}"]`));
const tick = () => new Promise((r) => setTimeout(r, 0));
async function until(cond: () => boolean, what: string, ms = 8000) {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error(`timeout: ${what}`);
    await tick();
  }
}
async function pickFile(testId: string, name: string, text: string) {
  const input = q(testId) as HTMLInputElement;
  const file = new File([text], name, { type: "text/csv" });
  // jsdom の File には arrayBuffer・text が無い（実機のブラウザにはある）ので、Node の Blob で補う
  const nb = new NodeBlob([text]);
  Object.defineProperties(file, { arrayBuffer: { value: () => nb.arrayBuffer() }, text: { value: () => nb.text() } });
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  input.dispatchEvent(new Event("change"));
  await until(() => all("rs-loaded").some((e) => e.textContent?.includes(name)), `${name} listed`);
}
async function choose(testId: string, value: string) {
  const sel = q(testId) as HTMLSelectElement;
  sel.value = value;
  sel.dispatchEvent(new Event("change"));
  await tick();
}

beforeEach(() => {
  document.body.replaceChildren();
  root = document.createElement("div");
  document.body.append(root);
  mountWizard(root, "ja");
});

describe("wizard: files", () => {
  it("keeps the chosen file name, rows and columns in the list after re-rendering, and removes one at a time", async () => {
    expect(q("rs-file-features")).toBeTruthy();
    await pickFile("rs-file-features", "a.csv", featureCsv("PatientID", ids.slice(0, 5)));
    await pickFile("rs-file-features", "b.csv", featureCsv("PatientID", ids.slice(5, 8)));
    const text = () => all("rs-loaded").map((e) => e.textContent ?? "");
    expect(text()).toHaveLength(2);
    expect(text()[0]).toContain("a.csv");
    expect(text()[0]).toContain("10 行");
    expect(text()[0]).toContain("4 列");
    expect(text()[1]).toContain("b.csv");
    // 別の操作で描き直しても残る
    await pickFile("rs-file-clinical", "c.csv", clinicalCsv("PatientID", ids));
    expect(text().some((x) => x.includes("a.csv")) && text().some((x) => x.includes("c.csv"))).toBe(true);
    // 1 件だけ外す
    (q("rs-file-features-remove-0") as HTMLButtonElement).click();
    await tick();
    expect(text().some((x) => x.includes("a.csv"))).toBe(false);
    expect(text().some((x) => x.includes("b.csv"))).toBe(true);
  });
});

describe("wizard: dataset check gates the next sections", () => {
  it("disables the skill and run sections when nothing joins, and enables them when the IDs match", async () => {
    expect(q("rs-reason-check")).toBeTruthy(); // ファイルが無い間の理由
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids.map((x) => x.replace("P", "Q"))));
    await until(() => !!q("rs-joined"), "check result");
    expect(q("rs-joined")!.textContent).toContain("0 人");
    expect(q("rs-reason-skill")).toBeTruthy();
    expect((q("rs-skill") as HTMLSelectElement).disabled).toBe(true); // 通るまで解析の種類は選べない
    expect((q("rs-run") as HTMLButtonElement).disabled).toBe(true);
    expect(q("rs-check-pass")).toBeNull();
    expect(q("rs-note-no-overlap")).toBeTruthy();
    expect(q("rs-time")).toBeNull();

    // 臨床の表を、ID が合うものに差し替える
    await pickFile("rs-file-clinical", "clin2.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(q("rs-joined")!.textContent).toContain(`${N} 人`);
    expect(q("rs-reason-skill")).toBeNull();
    expect((q("rs-skill") as HTMLSelectElement).disabled).toBe(false);
  });

  it("proposes the clinical key by value overlap, not by name (SubjectCode here)", async () => {
    const feat = `CaseLabel,PatientID,SubjectCode,f1\n` + ids.flatMap((id, i) => [0, 1].map((k) => `${id}_L${k},${id},S${i},${i / 3 + 0.5 + k}`)).join("\n") + "\n";
    await pickFile("rs-file-features", "feat.csv", feat);
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("SubjectCode", ids.map((_, i) => `S${i}`)));
    await until(() => !!q("rs-joined"), "check result");
    expect((q("rs-clinical-id") as HTMLSelectElement).value).toBe("SubjectCode");
    expect((q("rs-patient") as HTMLSelectElement).value).toBe("SubjectCode");
    expect((q("rs-feature-id") as HTMLSelectElement).value).toBe("CaseLabel");
    expect(q("rs-joined")!.textContent).toContain(`${N} 人`);
  });
});

describe("wizard: target variables follow the skill", () => {
  it("shows time/event for R1 and the outcome label for R2, from the clinical headers, and runs with the defaults", async () => {
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    await choose("rs-skill", "r1-prognosis");
    expect(q("rs-time")).toBeTruthy();
    expect(q("rs-event")).toBeTruthy();
    expect(q("rs-outcome")).toBeNull();
    expect(Array.from((q("rs-time") as HTMLSelectElement).options).map((o) => o.value)).toEqual(["", "OS_months", "Event", "Outcome", "age"]);
    expect((q("rs-time") as HTMLSelectElement).value).toBe("OS_months"); // 既定の推測のまま
    expect(q("rs-time-qc")!.textContent).toContain("正の数 40");
    expect(q("rs-event-qc")!.textContent).toContain("0: 14・1: 26");
    expect(q("rs-params")).toBeTruthy();
    expect((q("rs-params") as HTMLDetailsElement).open).toBe(false);

    await choose("rs-skill", "r2-discrimination");
    expect(q("rs-time")).toBeNull();
    expect(q("rs-outcome")).toBeTruthy();
    expect((q("rs-outcome") as HTMLSelectElement).value).toBe("Outcome");
    expect(q("rs-outcome-qc")!.textContent).toContain("異なる値 2 個");
    expect(q("rs-positive")).toBeTruthy();

    // 既定のまま（時間・イベントの選択を触らず）R1 を実行する
    await choose("rs-skill", "r1-prognosis");
    (q("rs-seed") as HTMLInputElement).value = "3";
    expect((q("rs-run") as HTMLButtonElement).disabled).toBe(false);
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
    expect(q("rs-result")).toBeTruthy();
  }, 70000);

  it("does not pass an unused role column to the analysis (R2 with an invalid time column still runs)", async () => {
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids));
    // 時間の列に 0 を入れた臨床の表。R2 は時間を使わない
    const clin = clinicalCsv("PatientID", ids).replace(/^(P001),\d+/m, "$1,0");
    await pickFile("rs-file-clinical", "clin.csv", clin);
    await until(() => !!q("rs-check-pass"), "check pass");
    await choose("rs-skill", "r2-discrimination");
    await choose("rs-positive", "resp");
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")?.textContent).toBeUndefined();
    expect(q("rs-result")).toBeTruthy();
  }, 70000);
});

describe("wizard: reproducibility needs no clinical table", () => {
  it("shows only the row-id column in the check section and runs with two condition files", async () => {
    await choose("rs-skill", "r3-reproducibility");
    await pickFile("rs-file-features", "condA.csv", featureCsv("PatientID", ids.slice(0, 12)));
    expect(q("rs-r3-id")).toBeTruthy();
    expect(q("rs-clinical-id")).toBeNull();
    expect((q("rs-run") as HTMLButtonElement).disabled).toBe(true); // 条件が 1 つ
    await pickFile("rs-file-features", "condB.csv", featureCsv("PatientID", ids.slice(0, 12)));
    expect((q("rs-run") as HTMLButtonElement).disabled).toBe(false);
  });
});


// 2D の単位を選んだ一括抽出の表: 症例（患者）ごとに WHOLE_3D・MAX_SECTION_2D・SLICE_2D の 3 行
const UNITS3: [string, string, string][] = [["WHOLE_3D", "", ""], ["MAX_SECTION_2D", "ACQUISITION", "AXIAL"], ["SLICE_2D", "RESLICED", "SAGITTAL"]];
const feature2dCsv = (extra: { head?: string[]; cells?: (i: number, k: number) => (string | number)[] } = {}) =>
  ["CaseLabel,PatientID,SubjectCode,Unit,PlaneSource,Plane,RowKey,f1,f2" + (extra.head ? `,${extra.head.join(",")}` : ""),
    ...ids.flatMap((id, i) => UNITS3.map(([u, ps, pl], k) =>
      [`C${String(i + 1).padStart(3, "0")}`, id, `S${i}`, u, ps, pl, `C${i + 1}|${u}|${ps}|${pl}|${k}`, ((1 + ((i * 13 + k * 5) % 17)) / 3).toFixed(4), ((2 + ((i * 7 + k) % 11)) / 7).toFixed(4), ...(extra.cells?.(i, k) ?? [])].join(",")))].join("\n") + "\n";
const check = (el: HTMLElement | null) => (el as HTMLInputElement).checked;
async function toggle(testId: string, on: boolean) {
  const el = q(testId) as HTMLInputElement;
  el.checked = on;
  el.dispatchEvent(new Event("change"));
  await tick();
}

describe("wizard: 2D rows (unit of analysis)", () => {
  it("default: WHOLE_3D only, a unique RowKey is the row ID, no duplicate-ID stop and nothing is averaged", async () => {
    await pickFile("rs-file-features", "roi-features-batch.csv", feature2dCsv());
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(q("rs-units")).toBeTruthy();
    expect(check(q("rs-unit-Unit-WHOLE_3D"))).toBe(true);
    expect(check(q("rs-unit-Unit-MAX_SECTION_2D"))).toBe(false);
    expect(check(q("rs-unit-Unit-SLICE_2D"))).toBe(false);
    expect(q("rs-unit-Unit-WHOLE_3D")!.parentElement!.textContent).toContain(`(${N})`); // 件数
    // Plane・PlaneSource は全部の値を選ぶ
    expect(all("rs-unit-Plane").length).toBe(1);
    expect(Array.from(q("rs-unit-Plane")!.querySelectorAll("input")).every((c) => (c as HTMLInputElement).checked)).toBe(true);
    expect((q("rs-feature-id") as HTMLSelectElement).value).toBe("RowKey");
    expect((q("rs-patient") as HTMLSelectElement).value).toBe("PatientID");
    expect(q("rs-joined")!.textContent).toContain(`${N} 人`);
    expect(q("rs-check-errors")).toBeNull();
    expect(q("rs-filtered")!.textContent).toContain(String(2 * N));
    expect(q("rs-note-averaged")).toBeNull();
    expect((q("rs-skill") as HTMLSelectElement).disabled).toBe(false);
  });

  it("selecting more units warns that rows are averaged per patient; selecting none stops with an error", async () => {
    await pickFile("rs-file-features", "roi-features-batch.csv", feature2dCsv());
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    await toggle("rs-unit-Unit-MAX_SECTION_2D", true);
    await until(() => !!q("rs-note-averaged"), "averaged note");
    expect(q("rs-note-averaged")!.textContent).toContain(`${2 * N} 行を患者ごとに平均します`);
    expect(q("rs-note-averaged")!.textContent).toContain("Unit・Plane が混ざった平均は解釈に注意");
    expect(q("rs-check-pass")).not.toBeNull();
    await toggle("rs-unit-Unit-MAX_SECTION_2D", false);
    await toggle("rs-unit-Unit-WHOLE_3D", false);
    await until(() => !!q("rs-check-errors"), "error");
    expect(q("rs-check-errors")!.textContent).toContain("1 行も残りません");
    expect(q("rs-check-pass")).toBeNull();
    expect((q("rs-skill") as HTMLSelectElement).disabled).toBe(true);
  });

  it("a table without Unit/PlaneSource/Plane shows no unit section", async () => {
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(q("rs-units")).toBeNull();
    expect(q("rs-leak-cols")).toBeNull();
    expect(q("rs-leak-target")).toBeNull();
  });
});

describe("wizard: data leakage warnings", () => {
  it("flags an ID-like column in use, lists the already-excluded ones separately, and excludes on request", async () => {
    const feat = featureCsv("PatientID", ids).split("\n").filter(Boolean).map((line, r) => (r === 0 ? `${line},RowNo,Provenance_N` : `${line},${r},3`)).join("\n") + "\n";
    await pickFile("rs-file-features", "feat.csv", feat);
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(q("rs-leak-cols-warn")!.textContent).toContain("RowNo");
    expect(q("rs-leak-cols-warn")!.textContent).not.toContain("Provenance_N");
    expect(q("rs-leak-auto")!.textContent).toContain("Provenance_N"); // 除外済み
    expect(q("rs-leak-auto")!.textContent).not.toContain("RowNo");
    const before = q("rs-feature-count")!.textContent!;
    (q("rs-leak-exclude-RowNo") as HTMLButtonElement).click();
    await until(() => !q("rs-leak-cols-warn"), "warning gone");
    expect(q("rs-leak-user-excluded")!.textContent).toContain("RowNo");
    expect(q("rs-feature-count")!.textContent).not.toBe(before);
    // 戻すと、また警告になる
    (q("rs-leak-restore-RowNo") as HTMLButtonElement).click();
    await until(() => !!q("rs-leak-cols-warn"), "warning back");
  });

  it("warns when a feature completely separates the outcome (default role choice), and can exclude it", async () => {
    const feat = featureCsv("PatientID", ids).split("\n").filter(Boolean).map((line, r) => {
      if (r === 0) return `${line},sepf`;
      const i = Math.floor((r - 1) / 2);
      return `${line},${(i % 2 ? 10 : 1) + (i % 5) / 10}`; // Outcome: i % 2 ? resp : nonresp
    }).join("\n") + "\n";
    await pickFile("rs-file-features", "feat.csv", feat);
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    await choose("rs-skill", "r2-discrimination"); // 目的変数は既定の推測のまま
    expect((q("rs-outcome") as HTMLSelectElement).value).toBe("Outcome");
    expect(q("rs-leak-target")!.textContent).toContain("sepf");
    expect(q("rs-leak-target")!.textContent).toContain("結果を完全に分ける");
    expect(q("rs-leak-target")!.textContent).not.toContain("f1 ↔");
    (q("rs-leak-target-exclude-sepf") as HTMLButtonElement).click();
    await until(() => !q("rs-leak-target"), "target warning gone");
    await until(() => !!q("rs-check-pass"), "check pass again");
  });
});

// ---------------------------------------------------------------------------------------------
// 前処理の区画（段 B3b）。Skill ごとに手順の出方が変わる
// ---------------------------------------------------------------------------------------------
async function setNumber(testId: string, v: string) {
  const el = q(testId) as HTMLInputElement;
  el.value = v;
  el.dispatchEvent(new Event("change"));
  await tick();
}
const has = (id: string) => q(id) !== null;
// 特徴量: 差のある f1・f2 に加え、定数 fc と、f1 と順位が同じ fd（f1 の 2 乗）
const preFeatureCsv = (idList: string[]) =>
  `CaseLabel,PatientID,f1,f2,fc,fd\n` + idList.flatMap((id, i) => [0, 1].map((k) => {
    const f1 = (1 + ((i * 13 + k * 5) % 17)) / 3 + (i % 2 ? 3 : 0);
    return `${id}_L${k},${id},${f1.toFixed(4)},${((2 + ((i * 7 + k) % 11)) / 7).toFixed(4)},2,${(f1 * f1).toFixed(4)}`;
  })).join("\n") + "\n";

describe("wizard: preprocessing section", () => {
  async function ready() {
    await pickFile("rs-file-features", "feat.csv", preFeatureCsv(ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
  }

  it("shows the steps that exist for each skill and none that do not; R3 has no section; the run section is numbered 5", async () => {
    await ready();
    expect(has("rs-sec-pre")).toBe(false); // 解析の種類を選ぶまで無い
    const nums = () => Array.from(root.querySelectorAll(".rs-num")).map((e) => e.textContent);
    // R1: 分散 0・相関だけ（対応なし・報告・標準化は出ない）
    await choose("rs-skill", "r1-prognosis");
    expect(has("rs-sec-pre")).toBe(true);
    expect(nums()).toEqual(["1", "2", "3", "4", "5"]);
    expect(q("rs-sec-run")!.querySelector(".rs-num")!.textContent).toBe("5");
    for (const id of ["rs-pre-const", "rs-pre-corr"]) expect(has(id), id).toBe(true);
    for (const id of ["rs-pre-assump-on", "rs-pre-unpaired-on", "rs-pre-paired-on", "rs-pre-std", "rs-p4-const", "rs-r5-corr"]) expect(has(id), id).toBe(false);
    // ⓘ のツールチップ
    expect(q("rs-pre-const-tip")!.getAttribute("title")!.length).toBeGreaterThan(10);
    // 既定はすべてオフ
    expect(check(q("rs-pre-const"))).toBe(false);
    expect((q("rs-pre-corr") as HTMLInputElement).value).toBe("");
    // R2: ＋正規性・等分散性の報告
    await choose("rs-skill", "r2-discrimination");
    for (const id of ["rs-pre-const", "rs-pre-corr", "rs-pre-assump-on"]) expect(has(id), id).toBe(true);
    for (const id of ["rs-pre-unpaired-on", "rs-pre-paired-on", "rs-pre-std"]) expect(has(id), id).toBe(false);
    // R3: 区画が無い
    await choose("rs-skill", "r3-reproducibility");
    expect(has("rs-sec-pre")).toBe(false);
  });

  it("P4: binary outcome adds the unpaired p-value filter and the standardisation display; survival does not; the old data-testids are kept", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    // 結果の型は臨床の表から推定して入る
    expect(["binary", "survival"]).toContain((q("rs-p4-kind") as HTMLSelectElement).value);
    await choose("rs-p4-kind", "survival");
    for (const id of ["rs-p4-const", "rs-p4-corr", "rs-pre-std"]) expect(has(id), id).toBe(true);
    for (const id of ["rs-pre-unpaired-on", "rs-pre-assump-on", "rs-pre-const", "rs-pre-corr"]) expect(has(id), id).toBe(false);
    await choose("rs-p4-kind", "binary");
    for (const id of ["rs-p4-const", "rs-p4-corr", "rs-pre-std", "rs-pre-unpaired-on", "rs-pre-assump-on"]) expect(has(id), id).toBe(true);
    expect(has("rs-pre-paired-on")).toBe(false); // 前と後の表が無い
    // 標準化は「必要か」を表示するだけ（モデルで決まる）
    expect(q("rs-pre-std")!.getAttribute("data-need")).toBe("true"); // 上位 k 個（ロジスティック）
    await choose("rs-p4-model", "rf");
    expect(q("rs-pre-std")!.getAttribute("data-need")).toBe("false");
    await choose("rs-p4-model", "lasso");
    expect(q("rs-pre-std")!.getAttribute("data-need")).toBe("true");
    // 対応なしの除外: オンにすると α・しきい値が出る（既定 0.05）
    await toggle("rs-pre-unpaired-on", true);
    expect((q("rs-pre-unpaired-alpha") as HTMLInputElement).value).toBe("0.05");
    expect((q("rs-pre-unpaired-thr") as HTMLInputElement).value).toBe("0.05");
    expect(q("rs-pre-unpaired-cost")!.textContent).toContain("訓練の行の組 36 回"); // 既定の (5×1+1)×(5+1)
    await setNumber("rs-p4-outer", ""); 
    expect(q("rs-pre-unpaired-cost")!.textContent).toContain("入れる"); // 分割数が空の間は目安を出さない
    await setNumber("rs-p4-outer", "4"); await setNumber("rs-p4-inner", "3"); await setNumber("rs-p4-repeats", "1");
    expect(q("rs-pre-unpaired-cost")!.textContent).toContain("訓練の行の組 20 回"); // (4×1+1)×(3+1)
  });

  it("P4: the model list follows the outcome type (common models first), and a model that no longer fits is reset with a note", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    const opts = () => Array.from((q("rs-p4-model") as HTMLSelectElement).options).map((o) => o.value);
    await choose("rs-p4-kind", "binary");
    expect(opts()).toEqual(["topk", "lasso", "rf", "svm", "gbm"]);
    await choose("rs-p4-model", "rf");
    expect(q("rs-p4-model-reset")).toBeNull();
    await choose("rs-p4-kind", "survival");
    expect(opts()).toEqual(["topk", "lasso", "enet", "ridge", "coxboost", "rsf"]); // 生存では RF・SVM・GBM は使えず、生存のモデルが出る
    expect((q("rs-p4-model") as HTMLSelectElement).value).toBe("topk");
    expect(q("rs-p4-model-reset")!.textContent).toContain("ランダムフォレスト");
    // モデルごとの設定欄
    expect(has("rs-p4-enet-alpha")).toBe(false);
    await choose("rs-p4-model", "enet");
    expect(has("rs-p4-enet-alpha")).toBe(true);
    expect(has("rs-p4-lasso-rule")).toBe(true);
    await choose("rs-p4-model", "ridge");
    expect(has("rs-p4-enet-alpha")).toBe(false);
    expect(has("rs-p4-lasso-rule")).toBe(true);
    await choose("rs-p4-model", "rsf");
    expect(has("rs-p4-rsf-trees") && has("rs-p4-rsf-mtry") && has("rs-p4-rsf-leaf")).toBe(true);
    expect(has("rs-p4-cox-iter")).toBe(false);
    await choose("rs-p4-model", "coxboost");
    expect(has("rs-p4-cox-iter")).toBe(true);
    expect((q("rs-p4-cox-nu") as HTMLInputElement).value).toBe("0.1"); // 学習率の既定
    expect(has("rs-p4-lasso-rule")).toBe(false);
    // 二値に切り替えると、生存のモデルは使えなくなり topk に戻る（注記つき）
    await choose("rs-p4-kind", "binary");
    expect((q("rs-p4-model") as HTMLSelectElement).value).toBe("topk");
    expect(q("rs-p4-model-reset")!.textContent).toContain("Cox ブースティング");
    await choose("rs-p4-kind", "survival");
    // 負例: 二値に戻しても、選び直すまで勝手に RF には戻らない
    await choose("rs-p4-kind", "binary");
    expect((q("rs-p4-model") as HTMLSelectElement).value).toBe("topk");
  });

  it("the correlation threshold default follows the skill: 0.9 for P4, blank for R1 (the key is shared between skills)", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    await choose("rs-p4-kind", "survival");
    expect((q("rs-p4-corr") as HTMLInputElement).value).toBe("0.9");
    await choose("rs-skill", "r1-prognosis");
    await until(() => !has("rs-p4-corr"), "R1 shown");
    expect((q("rs-pre-corr") as HTMLInputElement).value).toBe(""); // 負例: P4 の 0.9 が R1 の前処理に混ざって、相関除外が勝手に有効にならない
    await choose("rs-skill", "p4-pipeline");
    await choose("rs-p4-kind", "survival");
    expect((q("rs-p4-corr") as HTMLInputElement).value).toBe("0.9"); // P4 に戻すと既定に戻る
  });

  it("R5: the correlation threshold moved here with its data-testid; zero variance is fixed; standardisation is shown; no p-value filter without before/after tables", async () => {
    await ready();
    await choose("rs-skill", "r5-model");
    for (const id of ["rs-r5-corr", "rs-pre-const-fixed", "rs-pre-std"]) expect(has(id), id).toBe(true);
    for (const id of ["rs-pre-const", "rs-pre-corr", "rs-pre-unpaired-on", "rs-pre-paired-on", "rs-pre-assump-on"]) expect(has(id), id).toBe(false);
    expect(q("rs-sec-pre")!.contains(q("rs-r5-corr"))).toBe(true);
  });

  it("the paired filter appears only after both the before and after tables are loaded", async () => {
    await ready();
    await choose("rs-skill", "r1-prognosis");
    expect(has("rs-pre-paired-on")).toBe(false);
    await pickFile("rs-file-baseline", "base.csv", "CaseLabel,PatientID,f1,f2\nA_L0,P001,1,2\n");
    expect(has("rs-pre-paired-on")).toBe(false); // 片方だけでは出ない
    await pickFile("rs-file-followup", "follow.csv", "CaseLabel,PatientID,f1,f2\nA_L0,P001,2,3\n");
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(has("rs-pre-paired-on")).toBe(true);
    await toggle("rs-pre-paired-on", true);
    expect(has("rs-pre-paired-alpha")).toBe(true);
    expect(has("rs-pre-paired-thr")).toBe(true);
  });

  it("default flow: R2 runs with every preprocessing step left off and shows no preprocessing result", async () => {
    await ready();
    await choose("rs-skill", "r2-discrimination");
    await choose("rs-positive", "resp");
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
    expect(has("rs-pre-result")).toBe(false);
    expect(has("rs-assump")).toBe(false);
  }, 70000);

  it("R2 with zero-variance, correlation and the assumption report: shows what was removed and the assumption table, and offers the CSV", async () => {
    await ready();
    await choose("rs-skill", "r2-discrimination");
    await choose("rs-positive", "resp");
    await toggle("rs-pre-const", true);
    await setNumber("rs-pre-corr", "0.95");
    await toggle("rs-pre-assump-on", true);
    await setNumber("rs-pre-assump-alpha", "0.05");
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
    expect(q("rs-pre-result-count")!.textContent).toContain("4 → 2"); // fc（定数）と fd（f1 の 2 乗。順位は同じ）が除かれる
    expect(q("rs-pre-result-removed")!.textContent).toContain("fc");
    expect(q("rs-pre-result-removed")!.textContent).toContain("fd");
    expect(q("rs-pre-result-m")!.textContent).toContain("m");
    expect(all("rs-assump-row").length).toBe(2);
    expect(has("rs-save-assumptions")).toBe(true);
  }, 70000);

  it("P4 with the unpaired p-value filter runs from the default controls and lists the removals per training set", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    await choose("rs-p4-kind", "binary");
    await choose("rs-positive", "resp");
    await choose("rs-p4-lock", "none");
    await setNumber("rs-p4-outer", "4"); await setNumber("rs-p4-inner", "3"); await setNumber("rs-p4-repeats", "1");
    (q("rs-p4-k") as HTMLInputElement).value = "1, 2";
    q("rs-p4-k")!.dispatchEvent(new Event("change"));
    await choose("rs-p4-missing", "complete");
    await toggle("rs-p4-const", true); // 定数 fc を先に除く（検定できない列で全部落ちるのを避ける）
    await toggle("rs-pre-unpaired-on", true);
    await setNumber("rs-pre-unpaired-alpha", "0.05");
    await setNumber("rs-pre-unpaired-thr", "0.2");
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 90000);
    expect(q("rs-errors")?.textContent).toBeUndefined();
    expect(q("rs-p4-pdrop")).toBeTruthy();
    expect(all("rs-p4-pdrop-row").length).toBe(5); // 外側 4 ＋ 最終モデル
    expect(q("rs-p4-pdrop-summary")!.textContent).toContain("外側の訓練 4 回");
    expect(has("rs-save-p4-pdrop")).toBe(true);
    expect(has("rs-save-assumptions")).toBe(false);
    expect(q("rs-p4-leak-rows")!.getAttribute("data-level")).toBe("notice"); // 定数の除外が全行を読む（注意）。p 値による除外は訓練側だけ
  }, 100000);
});

// ---------------------------------------------------------------------------------------------
// 行の扱い（同じ患者に複数行があるとき）: 既定は平均、「行のまま」は P4 だけ
// ---------------------------------------------------------------------------------------------
describe("wizard: how to treat several rows per patient", () => {
  async function ready() {
    await pickFile("rs-file-features", "feat.csv", preFeatureCsv(ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
  }
  const asisOption = () => Array.from((q("rs-rows-mode") as HTMLSelectElement).options).find((o) => o.value === "asis")!;

  it("appears when a patient has several rows; 'rows as is' is disabled with a reason for R1, R2 and R5, and for no skill yet", async () => {
    await ready();
    expect(q("rs-rows-mode")).toBeTruthy();
    expect((q("rs-rows-mode") as HTMLSelectElement).value).toBe("average"); // 既定は平均
    expect(asisOption().disabled).toBe(true);
    expect(q("rs-rows-mode-reason")!.textContent).toContain("P4");
    for (const [skill, word] of [["r1-prognosis", "R1"], ["r2-discrimination", "R2"], ["r5-model", "R5"]] as const) {
      await choose("rs-skill", skill);
      expect(asisOption().disabled, skill).toBe(true);
      const reason = q("rs-rows-mode-reason")!.textContent!;
      expect(reason, skill).toContain(`${word} では選べません`);
      expect(reason, skill).toContain("独立でない");
    }
    expect(q("rs-rows-mode-reason")!.textContent).toContain("別の大きな変更"); // R5 の理由
    await choose("rs-skill", "p4-pipeline");
    expect(asisOption().disabled).toBe(false);
    expect(q("rs-rows-mode-reason")).toBeNull();
  });

  it("a table with one row per patient shows no row-mode choice", async () => {
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids).split("\n").filter((l, r) => r === 0 || l.includes("_L0,")).join("\n") + "\n");
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    expect(q("rs-rows-mode")).toBeNull();
  });

  it("P4: choosing 'rows as is' re-runs the check (rows and patients shown), keeps the analysis section, and runs from the default controls; it is recorded and the split check is OK", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    await choose("rs-rows-mode", "asis");
    await until(() => !!q("rs-note-rows-asis"), "rows-as-is note");
    expect(q("rs-note-rows-asis")!.textContent).toContain(`${2 * N} 行・${N} 人`);
    expect(q("rs-note-averaged")).toBeNull();
    expect(q("rs-joined")!.textContent).toContain(`${N} 人`);
    expect((q("rs-rows-mode") as HTMLSelectElement).value).toBe("asis");
    expect(q("rs-pre-rows-asis")!.textContent).toContain("行を単位に計算"); // p 値による除外などの注意
    await choose("rs-p4-kind", "binary");
    await choose("rs-positive", "resp");
    await choose("rs-p4-lock", "none");
    await setNumber("rs-p4-outer", "4"); await setNumber("rs-p4-inner", "3"); await setNumber("rs-p4-repeats", "1");
    (q("rs-p4-k") as HTMLInputElement).value = "1, 2";
    q("rs-p4-k")!.dispatchEvent(new Event("change"));
    await choose("rs-p4-missing", "complete");
    await toggle("rs-p4-const", true);
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 90000);
    expect(q("rs-errors")?.textContent).toBeUndefined();
    expect(q("rs-p4-rows-asis")!.textContent).toContain(`${2 * N} 行`);
    expect(q("rs-p4-leak-split")!.className).toContain("rs-note"); // OK（NG なら rs-err）
    expect(q("rs-p4-leak-split")!.textContent).not.toContain("分かれています");
    expect(all("rs-p4-fold").length).toBe(4);
  }, 100000);

  it("switching to R1 after choosing 'rows as is' returns to the per-patient mean (nothing is passed to R1)", async () => {
    await ready();
    await choose("rs-skill", "p4-pipeline");
    await choose("rs-rows-mode", "asis");
    await until(() => !!q("rs-note-rows-asis"), "rows-as-is note");
    await choose("rs-skill", "r1-prognosis");
    await until(() => !q("rs-note-rows-asis"), "back to the mean");
    expect((q("rs-rows-mode") as HTMLSelectElement).value).toBe("average");
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
  }, 70000);
});

describe("wizard: P4 runs on the defaults, and the survival stratification is in plain sight", () => {
  async function p4() {
    await pickFile("rs-file-features", "feat.csv", featureCsv("PatientID", ids));
    await pickFile("rs-file-clinical", "clin.csv", clinicalCsv("PatientID", ids));
    await until(() => !!q("rs-check-pass"), "check pass");
    await choose("rs-skill", "p4-pipeline");
  }
  const val = (id: string) => (q(id) as HTMLInputElement).value;

  it("survival: nothing is touched; the stratify choice is outside the detail settings and set to yes", async () => {
    await p4();
    expect(val("rs-p4-kind")).toBe("survival"); // 時間とイベントの列があるので生存
    const strat = q("rs-p4-strat") as HTMLSelectElement;
    expect(strat.value).toBe("true");
    expect(strat.closest("details")).toBeNull();
    expect(Array.from(strat.options).map((o) => o.value)).toEqual(["true", "false"]); // 「選んでください」は無い
    expect([val("rs-p4-lock"), val("rs-p4-outer"), val("rs-p4-inner"), val("rs-p4-repeats"), val("rs-p4-k"), val("rs-p4-missing")]).toEqual(["none", "5", "5", "1", "1, 2, 4, 8", "impute"]);
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
    expect(q("rs-result")).toBeTruthy();
  }, 120000);

  it("binary: the positive label is filled in and the run goes through", async () => {
    await p4();
    await choose("rs-p4-kind", "binary");
    expect(q("rs-p4-strat")).toBeNull(); // 二値は常に層化するので欄を出さない
    expect(val("rs-positive")).toBe("resp"); // 20 対 20 の同数は並びが後の方
    expect(Array.from((q("rs-positive") as HTMLSelectElement).options).map((o) => o.value)).toEqual(["nonresp", "resp"]);
    (q("rs-run") as HTMLButtonElement).click();
    await until(() => !!q("rs-result") || !!q("rs-errors"), "result", 60000);
    expect(q("rs-errors")).toBeNull();
    // 選び直した値は、描き直しても推定で上書きしない
    await choose("rs-positive", "nonresp");
    await choose("rs-p4-model", "lasso");
    expect(val("rs-positive")).toBe("nonresp");
    expect(val("rs-p4-lasso-rule")).toBe("1se");
  }, 120000);
});
