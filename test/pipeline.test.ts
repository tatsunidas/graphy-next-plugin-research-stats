import { describe, expect, it } from "vitest";
import fx from "./fixtures/pipeline.json";
import { Rng } from "../src/core/rng";
import { stratifiedFolds, stratifiedHoldout } from "../src/core/pipeline/split";
import { qcFeatures, spearman } from "../src/core/pipeline/qc";
import { fitStandardizer, imputeByTrain } from "../src/core/pipeline/preprocess";
import { finalModel, nestedCv, validateSpec, type ModelSpec, type NestedCvSpec, type Task } from "../src/core/pipeline/nestedCv";
import { scaleGamma, svmFit } from "../src/core/model/svm";
import { lassoAt, zRow } from "../src/core/model/lasso";

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

describe("nested cross-validation with LASSO", () => {
  const spec = (y: number[] | null, rule: "min" | "1se" = "min"): NestedCvSpec => ({ outerK: 5, innerK: 5, repeats: 1, kCandidates: [], strata: y, impute: false, model: { kind: "lasso", nLambda: 30, rule } });
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  it("binary: nested AUC is near 0.5 on random data", async () => {
    const auc: number[] = [];
    for (const seed of [1, 2, 3, 4]) {
      const { X, y } = noise(seed, 60, 400);
      const r = await nestedCv(X, { kind: "binary", y }, X.map((_, i) => i), spec(y), new Rng(seed));
      expect(r.folds.every((f) => f.error === null && f.lambda !== null)).toBe(true);
      auc.push(r.pooled[0]!);
    }
    expect(Math.abs(mean(auc) - 0.5)).toBeLessThan(0.1);
  }, 120000);
  it("survival: nested C-index is near 0.5 on random data", async () => {
    const cs: number[] = [];
    for (const seed of [11, 12, 13]) {
      const { X } = noise(seed, 80, 100);
      const rng = new Rng(seed + 100);
      const time = X.map(() => -Math.log(1 - rng.next())), event = X.map(() => (rng.next() < 0.7 ? 1 : 0));
      const r = await nestedCv(X, { kind: "survival", surv: { time, event } }, X.map((_, i) => i), spec(null), new Rng(seed));
      cs.push(r.summary.mean!);
    }
    expect(Math.abs(mean(cs) - 0.5)).toBeLessThan(0.1);
  }, 120000);
  it("positive control: the signal feature is chosen in every fold and AUC is high", async () => {
    const { X } = noise(5, 60, 50), e = noise(55, 60, 1).X;
    const y = X.map((r, i) => (r[7] + 0.8 * e[i][0] > 0 ? 1 : 0));
    const r = await nestedCv(X, { kind: "binary", y }, X.map((_, i) => i), spec(y, "1se"), new Rng(5));
    expect(r.pooled[0]!).toBeGreaterThan(0.75);
    expect(r.selectionCounts[7]).toBe(5);
    const fm = finalModel(X, { kind: "binary", y }, X.map((_, i) => i), spec(y, "1se"), new Rng(5));
    expect(fm.error).toBeNull();
    expect(fm.choice.lambda).toBeGreaterThan(0);
    expect(fm.model!.cols).toContain(7);
    expect(fm.model!.predict(0)).toBeGreaterThan(0);
  }, 60000);
  it("test rows are never used", async () => {
    const { X, y } = noise(6, 70, 30);
    const dev = X.map((_, i) => i).filter((i) => i % 7 !== 0);
    const s = spec(dev.map((i) => y[i]));
    const a = await nestedCv(X, { kind: "binary", y }, dev, s, new Rng(6));
    const X2 = X.map((r, i) => (i % 7 === 0 ? r.map((v) => v * 100 + 3) : r)), y2 = y.map((v, i) => (i % 7 === 0 ? 1 - v : v));
    const b = await nestedCv(X2, { kind: "binary", y: y2 }, dev, s, new Rng(6));
    expect(b.folds).toEqual(a.folds);
    expect(b.pred).toEqual(a.pred);
  }, 60000);
  it("rejects bad LASSO settings but does not need k candidates", () => {
    const base: NestedCvSpec = { outerK: 5, innerK: 4, repeats: 1, kCandidates: [], strata: null, impute: false, model: { kind: "lasso", nLambda: 20, rule: "min" } };
    expect(validateSpec(base, 30)).toBeNull();
    expect(validateSpec({ ...base, model: { kind: "lasso", nLambda: 1, rule: "min" } }, 30)).not.toBeNull();
  });
});

describe("nested cross-validation with LASSO: preprocessing and predictions", () => {
  const spec = (impute: boolean): NestedCvSpec => ({ outerK: 4, innerK: 4, repeats: 1, kCandidates: [], strata: null, impute, model: { kind: "lasso", nLambda: 20, rule: "min" } });
  /** 尺度と位置をずらした特徴量（標準化しないと予測が変わるように）。 */
  function scaled(seed: number, n: number, m: number) {
    const { X } = noise(seed, n, m), e = noise(seed + 50, n, 1).X;
    const y = X.map((r, i) => (r[0] - 0.8 * r[1] + 0.7 * e[i][0] > 0 ? 1 : 0));
    const Xs = X.map((r) => r.map((v, j) => v * (1 + 7 * j) + 3 * j - 5));
    return { X: Xs as (number | null)[][], y };
  }
  it("with imputation, the medians come from the training rows only (test rows changed: same result)", async () => {
    const { X, y } = scaled(31, 80, 8);
    X.forEach((r, i) => { if (i % 5 === 0) r[2] = null; });
    const dev = X.map((_, i) => i).filter((i) => i % 4 !== 0);
    const a = await nestedCv(X, { kind: "binary", y }, dev, spec(true), new Rng(31));
    const X2 = X.map((r, i) => (i % 4 === 0 ? r.map((v) => (v === null ? null : v * 50 + 100)) : [...r]));
    const b = await nestedCv(X2, { kind: "binary", y }, dev, spec(true), new Rng(31));
    expect(b.folds).toEqual(a.folds);
    expect(b.pred).toEqual(a.pred);
    const fa = finalModel(X, { kind: "binary", y }, dev, spec(true), new Rng(3));
    const fb = finalModel(X2, { kind: "binary", y }, dev, spec(true), new Rng(3));
    expect(fb.model!.medians).toEqual(fa.model!.medians);
  }, 60000);
  it("final-model predictions equal an independent calculation from lassoAt on the original scale", () => {
    const { X, y } = scaled(32, 90, 6);
    const dev = X.map((_, i) => i);
    const fm = finalModel(X, { kind: "binary", y }, dev, spec(false), new Rng(4));
    if (!fm.model) throw new Error(fm.error ?? "no model");
    const Xn = X as number[][];
    const ref = lassoAt({ kind: "binary", y }, Xn, dev, Xn[0].map((_, j) => j), fm.choice.lambda!);
    if ("error" in ref) throw new Error(ref.error);
    for (const i of [0, 7, 33, 89]) {
      const z = zRow(ref.std, Xn[i]);
      const e = z.reduce((s, v, j) => s + v * ref.coef.beta[j], ref.coef.b0);
      expect(Math.abs((fm.model.predict(i) as number) - 1 / (1 + Math.exp(-e)))).toBeLessThan(1e-6);
    }
  });
});

describe("nested cross-validation with random forest, SVM and gradient boosting", () => {
  const models: [string, ModelSpec][] = [
    ["rf", { kind: "rf", trees: 50, mtry: ["sqrt", 8] }],
    ["svm", { kind: "svm", C: [0.5, 2], gamma: ["scale", 0.01] }],
    ["gbm", { kind: "gbm", stages: [20, 50], depth: [1, 2], learningRate: 0.1, subsample: 0.8 }],
  ];
  const spec = (y: number[], model: ModelSpec, impute = false): NestedCvSpec => ({ outerK: 5, innerK: 4, repeats: 1, kCandidates: [], strata: y, impute, model });

  for (const [name, model] of models) {
    it(`${name}: random labels give a pooled AUC near 0.5`, async () => {
      const a: number[] = [];
      for (const seed of [1, 2, 3, 4]) {
        const { X, y } = noise(seed + 40, 80, 30);
        const r = await nestedCv(X, { kind: "binary", y }, X.map((_, i) => i), spec(y, model), new Rng(seed));
        a.push(r.pooled[0]!);
      }
      // 80 人（40 対 40）の AUC の SD はおよそ 0.065。4 回の平均で ±0.1 に収まることを求める
      expect(Math.abs(a.reduce((s, x) => s + x, 0) / a.length - 0.5)).toBeLessThan(0.1);
    }, 120000);

    it(`${name}: rows outside the development set do not change the results; the same seed reproduces`, async () => {
      const { X } = noise(51, 90, 8);
      const yy = X.map((r, i) => (r[0] - 0.7 * r[1] + 0.5 * (i % 3 === 0 ? 1 : -1) > 0 ? 1 : 0));
      X.forEach((r, i) => { if (i % 6 === 0) r[3] = null as unknown as number; });
      const dev = X.map((_, i) => i).filter((i) => i % 5 !== 0);
      const strata = dev.map((i) => yy[i]);
      const sp: NestedCvSpec = { ...spec(strata, model, true) };
      const a = await nestedCv(X, { kind: "binary", y: yy }, dev, sp, new Rng(8));
      const X2 = X.map((r, i) => (i % 5 === 0 ? r.map((v) => (v === null ? null : v * 30 - 7)) : [...r]));
      const b = await nestedCv(X2 as number[][], { kind: "binary", y: yy }, dev, sp, new Rng(8));
      expect(b.folds).toEqual(a.folds);
      expect(b.pred).toEqual(a.pred);
      expect(a.pooled[0]!).toBeGreaterThan(0.75);
      const fa = finalModel(X, { kind: "binary", y: yy }, dev, sp, new Rng(2)), fb = finalModel(X2 as number[][], { kind: "binary", y: yy }, dev, sp, new Rng(2));
      expect(fb.choice).toEqual(fa.choice);
      expect(dev.map((i) => fb.model!.predict(i))).toEqual(dev.map((i) => fa.model!.predict(i)));
    }, 120000);
  }

  it("SVM final model equals an independent fit on the training-row standardisation with γ = scale", () => {
    const { X } = noise(61, 70, 5);
    const Xs = X.map((r) => r.map((v, j) => v * (1 + 4 * j) + 2 * j));
    const y = X.map((r) => (r[0] + 0.5 * r[2] > 0 ? 1 : 0));
    const dev = X.map((_, i) => i).filter((i) => i % 7 !== 0);
    const fm = finalModel(Xs, { kind: "binary", y }, dev, spec(dev.map((i) => y[i]), { kind: "svm", C: [1], gamma: ["scale"] }), new Rng(3));
    const st = fitStandardizer(Xs, dev, [0, 1, 2, 3, 4]);
    if ("error" in st) throw new Error(st.error);
    const z = (r: number[]) => r.map((v, j) => (v - st.means[j]) / st.sds[j]);
    const m = svmFit(dev.map((i) => z(Xs[i])), dev.map((i) => y[i]), { C: 1, gamma: scaleGamma(dev.map((i) => z(Xs[i]))) });
    expect(fm.model!.resolved!.gamma).toBeCloseTo(scaleGamma(dev.map((i) => z(Xs[i]))), 14);
    for (const i of [0, 7, 14, 30]) expect(fm.model!.predict(i)).toBeCloseTo(m.decision(z(Xs[i])), 10);
  });

  it("chooses the candidate with the higher inner AUC, and validates the grid", async () => {
    const { X } = noise(71, 100, 4);
    const y = X.map((r) => (r[0] - r[1] > 0 ? 1 : 0));
    const all = X.map((_, i) => i);
    // γ が極端に大きいと、訓練の行以外では決定値がほぼ ρ だけになり AUC が下がる
    const fm = finalModel(X, { kind: "binary", y }, all, spec(y, { kind: "svm", C: [1], gamma: [1000, 0.1] }), new Rng(1));
    expect(fm.choice.params).toEqual({ C: 1, gamma: 0.1 });
    expect(fm.model!.resolved).toEqual({ C: 1, gamma: 0.1 });
    expect(fm.choice.scores[1]!).toBeGreaterThan(fm.choice.scores[0]!);
    expect(validateSpec(spec(y, { kind: "rf", trees: 0, mtry: [1] }), 100)).toMatch(/木の数/);
    expect(validateSpec(spec(y, { kind: "svm", C: [], gamma: ["scale"] }), 100)).toMatch(/C の候補/);
    expect(validateSpec(spec(y, { kind: "gbm", stages: [10], depth: [2], learningRate: 0, subsample: 1 }), 100)).toMatch(/学習率/);
    const sv = await nestedCv(X, { kind: "survival", surv: { time: all.map((i) => i + 1), event: all.map(() => 1) } }, all, { ...spec(y, { kind: "rf", trees: 5, mtry: [1] }), strata: null }, new Rng(1));
    expect(sv.folds.every((f) => f.score === null)).toBe(true);
  });
});
