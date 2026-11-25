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

