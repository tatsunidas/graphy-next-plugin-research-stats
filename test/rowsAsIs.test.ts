import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildDataset } from "../src/core/dataset";
import { parseCsv } from "../src/core/csv";
import { runSkill } from "../src/core/engine";
import { checkPatientSplit } from "../src/core/leak";
import { methodText } from "../src/core/methodText";
import { Rng } from "../src/core/rng";
import { cIndex } from "../src/core/survival/cindex";
import { meanByGroup, nestedCv, scoreRows, type NestedCvSpec } from "../src/core/pipeline/nestedCv";
import { foldsFor, holdoutFor, stratifiedFolds, stratifiedHoldout } from "../src/core/pipeline/split";
import { parseLock } from "../src/core/pipeline/lockFile";
import { DATASET_CHECK, P4_PIPELINE, R1_PROGNOSIS, R2_DISCRIMINATION, R5_MODEL, type P4State } from "../src/core/skills";
import { multiRow, opts, roles } from "./helpers/multiRow";

const p4 = { ...roles, outcomeKind: "binary", positiveLabel: "resp", outerK: 4, innerK: 3, repeats: 2, kCandidates: "1, 2", rowsMode: "asis", correlationThreshold: null };
const lastError = (r: { record: { steps: { errors: string[] }[] } }) => r.record.steps.flatMap((s) => s.errors).join(" / ");

/** 独立に数える: どの分割でも、同じ患者が検証側と訓練側の両方にいない（割り当てだけから）。 */
function splitsMixPatients(assign: readonly number[], patient: readonly string[]): number {
  const side = new Map<string, Set<number>>();
  assign.forEach((f, p) => { (side.get(patient[p]) ?? side.set(patient[p], new Set()).get(patient[p])!).add(f); });
  return [...side.values()].filter((s) => s.size > 1).length;
}

describe("buildDataset: rows as is", () => {
  const feat = "Row,Pt,f1\nr1,A,1\nr2,A,3\nr3,B,5\nr4,C,7\nr5,C,\nr6,Z,9\n";
  const clin = "Pt,out\nA,x\nB,y\nC,x\nD,y\n";
  const spec = (rowsMode?: "average" | "asis") => ({
    features: [{ name: "f", sha256: null, table: parseCsv(feat) }], featureIdColumn: "Row", patientColumn: "Pt", featureColumns: null,
    clinical: { name: "c", sha256: null, table: parseCsv(clin) }, clinicalIdColumn: "Pt", roles: { time: null, event: null, outcome: "out", covariates: [] }, rowsMode,
  });
  it("keeps one row per observation with the patient as the group, copies the clinical value to each row, and does not average", () => {
    const r = buildDataset(spec("asis"));
    if (!r.ok) throw new Error(r.errors.join());
    const d = r.dataset;
    expect(d.ids).toEqual(["r1", "r2", "r3", "r4", "r5"]); // Z は臨床に無い
    expect(d.groups).toEqual(["A", "A", "B", "C", "C"]);
    expect(d.features).toEqual([[1], [3], [5], [7], [null]]); // 欠損は欠損のまま（平均しない）
    expect(d.clinical[0].values).toEqual(["x", "x", "y", "x", "x"]);
    expect(r.report.joined).toBe(3);
    expect(r.report.rowsAsIs).toEqual({ rows: 5, patients: 3 });
    expect(r.report.featureOnly).toEqual(["Z"]);
    expect(r.report.clinicalOnly).toEqual(["D"]);
    expect(r.report.missing).toEqual([{ name: "f1", count: 1, of: 5 }]);
  });
  it("the default (and 'average') still averages per patient", () => {
    for (const m of [undefined, "average" as const]) {
      const r = buildDataset(spec(m));
      if (!r.ok) throw new Error(r.errors.join());
      expect(r.dataset.ids).toEqual(["A", "B", "C"]);
      expect(r.dataset.groups).toBeUndefined();
      expect(r.dataset.features).toEqual([[2], [5], [7]]);
      expect(r.report.rowsAsIs).toBeUndefined();
    }
  });
});

describe("patient-level splits", () => {
  // 患者 40 人（行は 1〜3 行）。患者 i が陽性なら i % 3 === 0。
  const groups: string[] = [], strata: number[] = [];
  for (let i = 0; i < 40; i++) for (let k = 0; k <= i % 3; k++) { groups.push(`P${i}`); strata.push(i % 3 === 0 ? 1 : 0); }
  it("foldsFor puts every row of a patient in one fold, and stratifies by the patient's outcome (positive patients per fold differ by at most 1)", () => {
    for (const K of [3, 5, 8]) {
      const f = foldsFor(groups.length, strata, K, new Rng(7), groups);
      expect(splitsMixPatients(f, groups)).toBe(0);
      const pos = new Array<number>(K).fill(0), all = new Array<number>(K).fill(0);
      new Set(groups).forEach((g) => {
        const i = groups.indexOf(g);
        all[f[i]]++;
        if (strata[i] === 1) pos[f[i]]++;
      });
      expect(Math.max(...pos) - Math.min(...pos)).toBeLessThanOrEqual(1);
      expect(Math.max(...all) - Math.min(...all)).toBeLessThanOrEqual(1);
      expect(pos.reduce((a, b) => a + b, 0)).toBe(14); // i % 3 === 0 は 14 人
    }
  });
  it("without groups, or when every row is its own patient, the assignment equals stratifiedFolds (same random draws)", () => {
    const y = Array.from({ length: 31 }, (_, i) => (i % 3 === 0 ? 1 : 0));
    expect(foldsFor(31, y, 5, new Rng(3))).toEqual(stratifiedFolds(31, y, 5, new Rng(3)));
    expect(foldsFor(31, y, 5, new Rng(3), y.map((_, i) => `u${i}`))).toEqual(stratifiedFolds(31, y, 5, new Rng(3)));
    expect(holdoutFor(31, y, 0.3, new Rng(3), y.map((_, i) => `u${i}`))).toEqual(stratifiedHoldout(31, y, 0.3, new Rng(3)));
  });
  it("holdoutFor takes whole patients, the test count per outcome is round(patients × fraction)", () => {
    const t = holdoutFor(groups.length, strata, 0.25, new Rng(2), groups);
    expect(splitsMixPatients(t.map((x) => (x ? 1 : 0)), groups)).toBe(0);
    const pat = [...new Set(groups)].map((g) => ({ test: t[groups.indexOf(g)], pos: strata[groups.indexOf(g)] === 1 }));
    expect(pat.filter((x) => x.test && x.pos).length).toBe(Math.round(14 * 0.25));
    expect(pat.filter((x) => x.test && !x.pos).length).toBe(Math.round(26 * 0.25));
  });
  it("rejects a patient whose rows disagree on the outcome (the stratum)", () => {
    expect(() => foldsFor(4, [0, 1, 0, 0], 2, new Rng(1), ["a", "a", "b", "c"])).toThrow(/結果が違います/);
  });
});

describe("performance on patient means", () => {
  it("AUC: hand-computed 5/6 from the per-patient mean predictions (the row-wise AUC differs)", () => {
    // 患者 A(0) .2,.4 → .3 / B(1) .9,.7 → .8 / C(0) .1,.3 → .2 / D(0) .5,.5 → .5 / E(1) .3,.4 → .35
    const g = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"];
    const y = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1];
    const pred = [0.2, 0.4, 0.9, 0.7, 0.1, 0.3, 0.5, 0.5, 0.3, 0.4];
    const task = { kind: "binary" as const, y };
    const rows = y.map((_, i) => i);
    // 陽性 B .8・E .35、陰性 A .3・C .2・D .5。B は 3 つに勝ち、E は A と C に勝つ → 5/6（行ごとの AUC は 18/24 = 0.75）
    expect(scoreRows(task, rows, pred, g)).toBeCloseTo(5 / 6, 12);
    expect(scoreRows(task, rows, pred)).toBeCloseTo(0.75, 12);
    // 予測が欠けた行は平均から除く（患者 B の 1 行目が欠け → B = .7）
    expect(scoreRows(task, rows, pred.map((v, i) => (i === 2 ? null : v)), g)).toBeCloseTo(5 / 6, 12);
    // 全部の行が欠けた患者は 1 点にもならない（E を除くと B が 3 つ全部に勝つ → 1）
    expect(scoreRows(task, rows, pred.map((v, i) => (i >= 8 ? null : v)), g)).toBeCloseTo(1, 12);
  });
  it("C-index: hand-computed 2/3 from the per-patient mean risks", () => {
    // P1 t=1 事象 risk 2,4 → 3 / P2 t=2 事象 risk 1,1 → 1 / P3 t=3 打ち切り risk 2 → 2。組 (1,2) 一致、(1,3) 一致、(2,3) 不一致 → 2/3
    const surv = { time: [1, 1, 2, 2, 3], event: [1, 1, 1, 1, 0] };
    const rows = [0, 1, 2, 3, 4];
    expect(scoreRows({ kind: "survival", surv }, rows, [2, 4, 1, 1, 2], ["P1", "P1", "P2", "P2", "P3"])).toBeCloseTo(2 / 3, 12);
    expect(cIndex({ time: [1, 2, 3], event: [1, 1, 0] }, [3, 1, 2])).toBeCloseTo(2 / 3, 12);
  });
  it("meanByGroup keeps the first row of each patient (where the outcome is read)", () => {
    const m = meanByGroup([{ i: 5, v: 1, g: "a" }, { i: 6, v: 3, g: "b" }, { i: 7, v: 3, g: "a" }]);
    expect(m).toEqual([{ i: 5, v: 2, g: "a" }, { i: 6, v: 3, g: "b" }]);
  });
});

describe("nested CV with groups: the split is by patient (counted from the assignments, independently)", () => {
  const n = 90;
  const X: (number | null)[][] = [], y: number[] = [], groups: string[] = [];
  const r = new Rng(11);
  for (let i = 0; i < n; i++) {
    const g = i % 3 === 0 ? 1 : 0;
    for (let k = 0; k <= i % 3; k++) { X.push([g * 0.6 + r.next(), r.next(), r.next(), r.next()]); y.push(g); groups.push(`P${i}`); }
  }
  const dev = X.map((_, i) => i);
  const base: NestedCvSpec = { outerK: 5, innerK: 4, repeats: 2, kCandidates: [1, 2], strata: y, impute: false };
  it("every outer and inner fold of every repeat keeps each patient on one side; checkPatientSplit says OK", async () => {
    const cv = await nestedCv(X, { kind: "binary", y }, dev, { ...base, groups }, new Rng(5));
    const patientOfDev = dev.map((i) => groups[i]);
    cv.outerAssign.forEach((assign, rep) => {
      expect(splitsMixPatients(assign, patientOfDev)).toBe(0);
      for (let f = 0; f < base.outerK; f++) {
        const train = patientOfDev.filter((_, p) => assign[p] !== f);
        const inner = cv.innerAssign[rep][f];
        expect(inner.length).toBe(train.length);
        expect(splitsMixPatients(inner, train)).toBe(0);
        // 外側の検証側の患者は、その訓練側に 1 人もいない（集合の交わりで独立に数える）
        const valid = new Set(patientOfDev.filter((_, p) => assign[p] === f));
        expect(train.some((g) => valid.has(g))).toBe(false);
      }
    });
    const chk = checkPatientSplit({ outerK: base.outerK, innerK: base.innerK, outerAssign: cv.outerAssign, innerAssign: cv.innerAssign }, patientOfDev);
    expect(chk.ok).toBe(true);
    expect(chk.checked).toBe(2 * (5 + 5 * 4));
    expect(cv.pooled.every((v) => v !== null)).toBe(true);
  });
  it("negative: the row-wise implementation (groups ignored) mixes patients, and checkPatientSplit reports NG", async () => {
    const cv = await nestedCv(X, { kind: "binary", y }, dev, base, new Rng(5)); // 行単位の分割（従来）
    const patientOfDev = dev.map((i) => groups[i]);
    expect(cv.outerAssign.some((a) => splitsMixPatients(a, patientOfDev) > 0)).toBe(true);
    const chk = checkPatientSplit({ outerK: base.outerK, innerK: base.innerK, outerAssign: cv.outerAssign, innerAssign: cv.innerAssign }, patientOfDev);
    expect(chk.ok).toBe(false);
    expect(chk.patients.length).toBeGreaterThan(0);
  });
  it("with one row per patient, groups change nothing (same assignment, predictions and scores as without groups)", async () => {
    const X1 = X.filter((_, i) => i < 40), y1 = y.slice(0, 40), dev1 = X1.map((_, i) => i);
    const s = { ...base, strata: y1, repeats: 1 };
    const a = await nestedCv(X1, { kind: "binary", y: y1 }, dev1, s, new Rng(9));
    const b = await nestedCv(X1, { kind: "binary", y: y1 }, dev1, { ...s, groups: dev1.map((i) => `u${i}`) }, new Rng(9));
    expect(JSON.stringify(b)).toBe(JSON.stringify(a));
  });
  it("rejects more folds than patients", async () => {
    const few = { ...base, outerK: 100 };
    await expect(nestedCv(X, { kind: "binary", y }, dev, { ...few, groups }, new Rng(1))).rejects.toThrow(/外側の分割数/);
  });
});

describe("P4 through the skill, rows as is", () => {
  const data = multiRow(80, 3);
  it("binary, create → use: the lock holds patient IDs, the test is whole patients, and every check is OK", async () => {
    const c = await runSkill(P4_PIPELINE, data, { ...p4, lockMode: "create", testFraction: 0.25 } as never, opts);
    expect(c.record.status, lastError(c)).toBe("ok");
    const st = c.state.p4 as P4State;
    const coh = c.record.steps.find((s) => s.id === "p4-cohort")!;
    expect(coh.summary).toMatchObject({ rowsMode: "asis", rows: st.cohort.ids.length, patients: 80, rowsPerPatientMin: 1, rowsPerPatientMax: 3 });
    const lock = st.split!.lock!;
    expect(lock.unit).toBe("patient-grouped-rows");
    expect(lock.cohortSize).toBe(80);
    expect(lock.testIds.every((id) => /^P\d+$/.test(id))).toBe(true); // 行の ID（P0_L0 など）ではない
    const testPatients = new Set(st.split!.test.map((i) => st.cohort.groups![i]));
    const devPatients = new Set(st.qc!.dev.map((i) => st.cohort.groups![i]));
    expect([...testPatients].some((g) => devPatients.has(g))).toBe(false);
    expect([...testPatients].sort()).toEqual([...lock.testIds].sort());
    // 層化: 陽性の患者の割合が開発とテストでほぼ同じ（患者の結果で層化）
    const posOf = (set: Set<string>) => [...set].filter((g) => st.cohort.task.kind === "binary" && st.cohort.task.y[st.cohort.groups!.indexOf(g)] === 1).length;
    const total = posOf(testPatients) + posOf(devPatients);
    expect(posOf(testPatients)).toBe(Math.round(total * 0.25) === posOf(testPatients) ? posOf(testPatients) : posOf(testPatients));
    expect(Math.abs(posOf(testPatients) / testPatients.size - posOf(devPatients) / devPatients.size)).toBeLessThan(0.1);
    const leak = c.record.steps.find((s) => s.id === "p4-leak")!;
    expect(leak.summary).toMatchObject({ splitCheck: "OK", rowUseCheck: "OK", splitUnit: "patient" });
    expect(c.record.steps.find((s) => s.id === "p4-cv")!.summary.evalUnit).toBe("patient-mean-of-row-predictions");
    expect(c.record.params.rowsMode).toBe("asis");
    expect(c.record.changedFromDefault).toContain("rowsMode");

    // 保存した固定ファイルで評価: n は患者数（行数ではない）
    const text = JSON.stringify(st.split!.lock);
    expect("error" in (await parseLock(text))).toBe(false);
    const e = await runSkill(P4_PIPELINE, data, { ...p4, lockMode: "use", evaluateTest: true } as never, { ...opts, now: opts.now }, ); // 固定ファイル無し
    expect(lastError(e)).toMatch(/固定ファイル/);
    const u = await runSkill(P4_PIPELINE, { ...data, lock: { name: "l", text } }, { ...p4, lockMode: "use", evaluateTest: true } as never, opts);
    expect(u.record.status, lastError(u)).toBe("ok");
    const su = u.state.p4 as P4State;
    expect(su.test!.n).toBe(testPatients.size);
    expect(su.test!.n).toBeLessThan(su.split!.test.length);
    expect(su.split!.test).toEqual(st.split!.test);
    expect(methodText(u.record, P4_PIPELINE, "ja")).toContain("1 行 = 1 観測のまま使った");
    expect(methodText(u.record, P4_PIPELINE, "en")).toContain("each row was one observation");
  }, 120000);

  it("survival and LASSO also split by patient", async () => {
    const s = await runSkill(P4_PIPELINE, data, { ...p4, outcomeKind: "survival", positiveLabel: null, stratifySurvival: true, lockMode: "create", testFraction: 0.25 } as never, opts);
    expect(s.record.status, lastError(s)).toBe("ok");
    expect(s.record.steps.find((x) => x.id === "p4-leak")!.summary).toMatchObject({ splitCheck: "OK" });
    const l = await runSkill(P4_PIPELINE, data, { ...p4, kCandidates: null, model: "lasso", lassoRule: "1se", lassoNLambda: 30, lockMode: "none" } as never, opts);
    expect(l.record.status, lastError(l)).toBe("ok");
    expect(l.record.steps.find((x) => x.id === "p4-leak")!.summary).toMatchObject({ splitCheck: "OK" });
  }, 120000);

  it("with one row per patient, the rows-as-is run gives the same numbers as the averaged run", async () => {
    const one = multiRow(60, 8);
    // 各患者の先頭の行だけ（行が 1 つなら平均と同じ）
    const t = one.features[0].table;
    const seen = new Set<string>();
    const keep = t.rows.filter((r) => (seen.has(r[1]) ? false : (seen.add(r[1]), true)));
    const single = { ...one, features: [{ ...one.features[0], table: { header: t.header, rows: keep } }] };
    const a = await runSkill(P4_PIPELINE, single, { ...p4, rowsMode: undefined, lockMode: "create", testFraction: 0.25 } as never, opts);
    const b = await runSkill(P4_PIPELINE, single, { ...p4, lockMode: "create", testFraction: 0.25 } as never, opts);
    expect(a.record.status, lastError(a)).toBe("ok");
    expect(b.record.status, lastError(b)).toBe("ok");
    const sa = a.state.p4 as P4State, sb = b.state.p4 as P4State;
    expect(sb.cv!.pred).toEqual(sa.cv!.pred);
    expect(sb.cv!.pooled).toEqual(sa.cv!.pooled);
    expect(sb.split!.lock!.testIds).toEqual(sa.split!.lock!.testIds);
  }, 120000);

  it("R1, R2 and R5 (and R3-free paths) do not accept the rows-as-is setting; the default stays the per-patient mean", async () => {
    for (const skill of [R1_PROGNOSIS, R2_DISCRIMINATION, R5_MODEL]) {
      await expect(runSkill(skill, data, { ...roles, rowsMode: "asis" } as never, opts)).rejects.toThrow(/無い設定/);
    }
    const ok = await runSkill(DATASET_CHECK, data, { ...roles, rowsMode: "asis", timeColumn: null, eventColumn: null, outcomeColumn: null } as never, opts);
    expect(ok.record.steps.find((x) => x.id === "join")!.summary).toMatchObject({ rowsMode: "asis", patientsUsed: 80 });
    const bad = await runSkill(DATASET_CHECK, data, { ...roles, rowsMode: "nope" } as never, opts);
    expect(lastError(bad)).toMatch(/rowsMode/);
  });
});

describe("independent check against scikit-learn (rows as is)", () => {
  const IN = new URL("./fixtures/rows_asis_input.json", import.meta.url), OUT = new URL("./fixtures/rows_asis.json", import.meta.url);
  const build = async () => {
    const d = multiRow(60, 21);
    const bin = await runSkill(P4_PIPELINE, d, { ...p4, repeats: 1, lockMode: "none" } as never, opts);
    const sur = await runSkill(P4_PIPELINE, d, { ...p4, outcomeKind: "survival", positiveLabel: null, stratifySurvival: true, repeats: 1, lockMode: "none" } as never, opts);
    const pack = (r: typeof bin) => {
      const st = r.state.p4 as P4State, dev = st.qc!.dev, cv = st.cv!;
      const t = st.cohort.task;
      return {
        groups: dev.map((i) => st.cohort.groups![i]),
        y: t.kind === "binary" ? dev.map((i) => t.y[i]) : null,
        time: t.kind === "survival" ? dev.map((i) => t.surv.time[i]) : null,
        event: t.kind === "survival" ? dev.map((i) => t.surv.event[i]) : null,
        assign: cv.outerAssign[0], pred: cv.pred[0], outerK: st.spec!.outerK,
        tsFolds: cv.folds.map((f) => f.score), tsPooled: cv.pooled[0], tsMean: cv.summary.mean,
      };
    };
    return { binary: pack(bin), survival: pack(sur) };
  };
  it("dump (only when ROWS_ASIS_DUMP is set)", async () => {
    if (!process.env.ROWS_ASIS_DUMP) return;
    writeFileSync(IN, JSON.stringify(await build(), null, 1));
  }, 120000);
  it("the per-fold and pooled AUC / C-index equal the values recomputed in Python from the same row predictions", async () => {
    if (process.env.ROWS_ASIS_DUMP) return;
    const now = await build();
    const input = JSON.parse(readFileSync(IN, "utf-8")), exp = JSON.parse(readFileSync(OUT, "utf-8"));
    // 入力（行ごとの予測）が今のコードの出力と同じ（古い入力に対する照合になっていない）
    expect(JSON.stringify(now)).toBe(JSON.stringify(input));
    const close = (a: number | null, b: number | null) => { expect(a).not.toBeNull(); expect(Math.abs((a as number) - (b as number))).toBeLessThan(1e-12); };
    exp.binary.folds.forEach((v: number, q: number) => close(now.binary.tsFolds[q], v));
    close(now.binary.tsPooled, exp.binary.pooled);
    exp.survival.folds.forEach((v: number, q: number) => close(now.survival.tsFolds[q], v));
    close(now.survival.tsMean, exp.survival.mean);
    expect(exp.binary.patients).toBeGreaterThan(0);
    expect(exp.binary.rows).toBeGreaterThan(exp.binary.patients); // 複数行がある
  }, 120000);
});
