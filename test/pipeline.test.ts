import { describe, expect, it } from "vitest";
import fx from "./fixtures/pipeline.json";
import { Rng } from "../src/core/rng";
import { stratifiedFolds, stratifiedHoldout } from "../src/core/pipeline/split";
import { qcFeatures, spearman } from "../src/core/pipeline/qc";
import { fitStandardizer, imputeByTrain } from "../src/core/pipeline/preprocess";
import { finalModel, nestedCv, validateSpec, type NestedCvSpec, type Task } from "../src/core/pipeline/nestedCv";

const close = (a: number | null, b: number | null, rel: number, abs = 0) => {
  if (b === null) { expect(a).toBeNull(); return; }
  expect(a).not.toBeNull();
  expect(Math.abs((a as number) - b)).toBeLessThanOrEqual(abs + rel * Math.abs(b));
};

type Case = (typeof fx.cases)[number];
const taskOf = (c: Case): Task => c.task.kind === "binary"
  ? { kind: "binary", y: c.task.y as number[] }
  : { kind: "survival", surv: { time: c.task.time as number[], event: c.task.event as number[] } };
const specOf = (c: Case): NestedCvSpec => ({ ...c.spec, strata: c.spec.strata as number[] | null });

describe("splits vs an independent implementation", () => {
  it("stratified folds, plain folds and hold-out give the same assignment for the same seed", () => {
    const s = fx.splits, rng = new Rng(s.seed);
    expect(stratifiedFolds(23, s.strata, 5, rng)).toEqual(s.folds5);
    expect(stratifiedFolds(23, null, 4, rng)).toEqual(s.foldsPlain4);
    expect(stratifiedHoldout(23, s.strata, 0.3, rng)).toEqual(s.holdout30);
  });
  it("fold sizes and per-stratum counts differ by at most 1, each case is in exactly one fold", () => {
    for (const seed of [1, 2, 3]) {
      const strata = Array.from({ length: 47 }, (_, i) => (i * 7) % 3 === 0 ? 1 : 0);
      const f = stratifiedFolds(47, strata, 5, new Rng(seed));
      expect(f.every((v) => v >= 0 && v < 5)).toBe(true);
      const size = [0, 1, 2, 3, 4].map((k) => f.filter((v) => v === k).length);
      expect(Math.max(...size) - Math.min(...size)).toBeLessThanOrEqual(1);
      for (const s of [0, 1]) {
        const c = [0, 1, 2, 3, 4].map((k) => f.filter((v, i) => v === k && strata[i] === s).length);
        expect(Math.max(...c) - Math.min(...c)).toBeLessThanOrEqual(1);
      }
    }
  });
  it("same seed reproduces, another seed changes the split", () => {
    const a = stratifiedFolds(40, null, 5, new Rng(9)), b = stratifiedFolds(40, null, 5, new Rng(9)), c = stratifiedFolds(40, null, 5, new Rng(10));
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });
  it("rejects impossible fold counts and fractions", () => {
    expect(() => stratifiedFolds(5, null, 6, new Rng(1))).toThrow();
    expect(() => stratifiedFolds(5, null, 1, new Rng(1))).toThrow();
    expect(() => stratifiedHoldout(5, null, 1, new Rng(1))).toThrow();
  });
});

describe("quality control", () => {
  it("Spearman with pairwise-complete rows vs scipy", () => {
    close(spearman(fx.spearman.x, fx.spearman.y), fx.spearman.rho, 1e-12, 1e-14);
  });
  it("drops the exclusion list, constant columns and correlated columns, keeping the column with fewer missing values", () => {
    const X = [[1, 5, 2, 9, 2.1], [2, 5, 4, 8, 4.2], [3, 5, 6, 7, null], [4, 5, 8, 6, 8.3], [5, 5, 1, 5, 1.0], [100, 7, 0, 0, 0]];
    const names = ["a", "const", "b", "anti", "b2"];
    const r = qcFeatures(X as (number | null)[][], [0, 1, 2, 3, 4], names, { dropConstant: true, correlationThreshold: 0.9, exclude: ["anti", "nope"] });
    expect(r.unknownExcluded).toEqual(["nope"]);
    expect(r.keep.map((j) => names[j])).toEqual(["a", "b"]);
    expect(r.removed.map((x) => x.feature).sort()).toEqual(["anti", "b2", "const"]);
    // 行 5（テスト集合）は見ない: その行だけ値が違う "const" も、開発集合の行では 1 種類
    expect(r.removed.find((x) => x.feature === "const")!.reason).toContain("1 種類");
  });
  it("does nothing when every option is off", () => {
    const r = qcFeatures([[1, 1], [2, 1]], [0, 1], ["x", "y"], { dropConstant: false, correlationThreshold: null, exclude: null });
    expect(r.keep).toEqual([0, 1]);
  });
});

describe("preprocessing uses the training rows only", () => {
  it("median, mean and SD come from the training rows; changing a held-out value changes nothing", () => {
    const X: (number | null)[][] = [[1], [null], [3], [10], [1000]];
    const train = [0, 1, 2, 3];
    const a = imputeByTrain(X, train, true);
    expect(a.medians[0]).toBe(3);
    expect(a.X[1][0]).toBe(3);
    const s = fitStandardizer(a.X, train, [0]);
    if ("error" in s) throw new Error(s.error);
    close(s.means[0], (1 + 3 + 3 + 10) / 4, 1e-15);
    const X2 = X.map((r) => [...r]); X2[4][0] = -5;
    const s2 = fitStandardizer(imputeByTrain(X2, train, true).X, train, [0]);
    expect(s2).toEqual(s);
  });
  it("without imputation a column with a missing training value is not usable", () => {
    expect(imputeByTrain([[1], [null], [3]], [0, 1], false).usable[0]).toBe(false);
    expect(imputeByTrain([[1], [null], [3]], [0, 2], false).usable[0]).toBe(true);
  });
});

describe("nested cross-validation vs an independent implementation (statsmodels / scikit-learn / scipy)", () => {
  for (const c of fx.cases) {
    it(c.name, async () => {
      const X = c.X as (number | null)[][], task = taskOf(c), spec = specOf(c), e = c.expected;
      const rng = new Rng(c.seed);
      const lab = task.kind === "binary" ? task.y : task.surv.event;
      expect(stratifiedHoldout(X.length, lab, 0.25, rng)).toEqual(c.test);
      const r = await nestedCv(X, task, c.dev, spec, rng);
      expect(r.outerAssign).toEqual(e.outerAssign);
      expect(r.innerAssign).toEqual(e.innerAssign);
      r.folds.forEach((f, q) => {
        const x = e.folds[q];
        expect(f.k).toBe(x.k);
        expect(f.cols).toEqual(x.cols);
        f.innerScores.forEach((s, a) => close(s, x.innerScores[a] as number | null, 1e-6, 1e-9));
        close(f.score, x.score as number | null, 1e-6, 1e-9);
      });
      r.pooled.forEach((p, a) => close(p, e.pooled[a] as number | null, 1e-6, 1e-9));
      const fm = finalModel(X, task, c.dev, spec, rng);
      expect(fm.error).toBeNull();
      expect(fm.choice.folds).toEqual(e.final.folds);
      expect(fm.choice.k).toBe(e.final.k);
      expect(fm.model!.cols).toEqual(e.final.cols);
      fm.model!.means.forEach((v, a) => close(v, e.final.means[a], 1e-12, 1e-12));
      fm.model!.sds.forEach((v, a) => close(v, e.final.sds[a], 1e-10));
      fm.model!.medians.forEach((v, a) => close(v, e.final.medians[a] as number | null, 0, 0));
      const testRows = c.test.map((t, i) => (t ? i : -1)).filter((i) => i >= 0);
      testRows.forEach((i, q) => close(fm.model!.predict(i), e.final.testPred[q] as number | null, 1e-5, 1e-7));
    });
  }
});

/** ランダムな特徴量・ランダムなラベル（どの特徴量にも情報が無い）。 */
function noise(seed: number, n: number, m: number) {
  const rng = new Rng(seed);
  const g = () => { const u = 1 - rng.next(), v = rng.next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const X = Array.from({ length: n }, () => Array.from({ length: m }, g));
  const y = Array.from({ length: n }, (_, i) => (i < n / 2 ? 1 : 0));
  return { X, y };
}

describe("detector: random labels and random features", () => {
  const spec = (y: number[]): NestedCvSpec => ({ outerK: 5, innerK: 5, repeats: 1, kCandidates: [1, 2, 4], strata: y, impute: false });
  it("nested AUC is near 0.5, ranking on all data first is clearly above 0.5", async () => {
    const honest: number[] = [], leaky: number[] = [];
    for (const seed of [1, 2, 3, 4]) {
      const { X, y } = noise(seed, 60, 400);
      const dev = X.map((_, i) => i);
      const h = await nestedCv(X, { kind: "binary", y }, dev, spec(y), new Rng(seed));
      const l = await nestedCv(X, { kind: "binary", y }, dev, spec(y), new Rng(seed), undefined, { rankOnAll: true });
      honest.push(h.pooled[0]!);
      leaky.push(l.pooled[0]!);
    }
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    expect(Math.abs(mean(honest) - 0.5)).toBeLessThan(0.1);
    expect(mean(leaky)).toBeGreaterThan(0.7);
    expect(Math.min(...leaky)).toBeGreaterThan(Math.max(...honest));
  }, 60000);
  it("positive control: a real signal gives AUC well above 0.5", async () => {
    const { X } = noise(5, 60, 50), e = noise(55, 60, 1).X;
    // ラベルにノイズを入れる（完全に分離するとロジスティック回帰が作れない）
    const y = X.map((r, i) => (r[7] + 0.8 * e[i][0] > 0 ? 1 : 0));
    const r = await nestedCv(X, { kind: "binary", y }, X.map((_, i) => i), spec(y), new Rng(5));
    expect(r.folds.every((f) => f.error === null)).toBe(true);
    expect(r.pooled[0]!).toBeGreaterThan(0.75);
    expect(r.selectionCounts[7]).toBe(5);
  });
  it("test rows are never used: changing their values and labels does not change the cross-validation", async () => {
    const { X, y } = noise(6, 70, 30);
    const dev = X.map((_, i) => i).filter((i) => i % 7 !== 0);
    const s = spec(dev.map((i) => y[i]));
    const a = await nestedCv(X, { kind: "binary", y }, dev, s, new Rng(6));
    const X2 = X.map((r, i) => (i % 7 === 0 ? r.map((v) => v * 100 + 3) : r)), y2 = y.map((v, i) => (i % 7 === 0 ? 1 - v : v));
    const b = await nestedCv(X2, { kind: "binary", y: y2 }, dev, s, new Rng(6));
    expect(b.folds).toEqual(a.folds);
    expect(b.pred).toEqual(a.pred);
  });
  it("rejects specs that cannot run", () => {
    const base: NestedCvSpec = { outerK: 5, innerK: 4, repeats: 1, kCandidates: [1], strata: null, impute: false };
    expect(validateSpec(base, 30)).toBeNull();
    expect(validateSpec({ ...base, outerK: 31 }, 30)).not.toBeNull();
    expect(validateSpec({ ...base, innerK: 25 }, 30)).not.toBeNull();
    expect(validateSpec({ ...base, kCandidates: [] }, 30)).not.toBeNull();
    expect(validateSpec({ ...base, kCandidates: [2, 2] }, 30)).not.toBeNull();
    expect(validateSpec({ ...base, repeats: 0 }, 30)).not.toBeNull();
  });
});
