import { describe, expect, it } from "vitest";
import fx from "./fixtures/rsf.json";
import { survivalForest } from "../src/core/model/survivalForest";
import { Rng } from "../src/core/rng";
import { cIndex } from "../src/core/survival/cindex";

const all = (n: number) => Array.from({ length: n }, (_, i) => i);

describe("random survival forest vs scikit-survival RandomSurvivalForest (no randomness: one tree, no bootstrap, all features)", () => {
  for (const c of fx.cases) {
    const surv = { time: c.time, event: c.event };
    it(`seed ${c.seed}: the risk of every training row and of new rows equals scikit-survival's, for every tie-free (min split, min leaf) setting (the features are drawn in a random order here, so a tie-free tree is what must match)`, () => {
      for (const f of c.fits) {
        const forest = survivalForest(surv, c.X, all(c.X.length), all(c.X[0].length), { trees: 1, mtry: c.X[0].length, minLeaf: f.minLeaf, minSplit: f.minSplit, bootstrap: false }, new Rng(1));
        c.X.forEach((x, i) => expect(Math.abs(forest.predict(x) - f.pred_train[i]), `train #${i} (${f.minSplit},${f.minLeaf})`).toBeLessThan(1e-9));
        f.Xnew.forEach((x, i) => expect(Math.abs(forest.predict(x) - f.pred_new[i]), `new #${i} (${f.minSplit},${f.minLeaf})`).toBeLessThan(1e-9));
        expect(new Set(f.pred_train).size).toBeGreaterThan(2); // 全部同じ値で通っていない
      }
    });
  }
  it("negative control: other settings give a different tree (the min leaf is not ignored)", () => {
    const c = fx.cases[0], f = c.fits[0];
    expect(f.minLeaf).toBeGreaterThan(1);
    const forest = survivalForest({ time: c.time, event: c.event }, c.X, all(c.X.length), all(c.X[0].length), { trees: 1, mtry: c.X[0].length, minLeaf: 1, minSplit: 2, bootstrap: false }, new Rng(1));
    const d = Math.max(...c.X.map((x, i) => Math.abs(forest.predict(x) - f.pred_train[i])));
    expect(d).toBeGreaterThan(1e-3);
  });
});

describe("random survival forest: bootstrap, mtry, reproducibility", () => {
  const c = fx.cases[1];
  const surv = { time: c.time, event: c.event };
  const run = (o: Partial<Parameters<typeof survivalForest>[4]>, seed = 1) => survivalForest(surv, c.X, all(c.X.length), all(c.X[0].length), { trees: 30, mtry: 3, minLeaf: 3, ...o }, new Rng(seed));
  it("the same seed gives the same predictions; another seed differs", () => {
    const a = run({}, 1), b = run({}, 1), d = run({}, 2);
    expect(c.X.map((x) => a.predict(x))).toEqual(c.X.map((x) => b.predict(x)));
    expect(c.X.some((x) => a.predict(x) !== d.predict(x))).toBe(true);
  });
  it("finds the planted signal: the out-of-sample ordering is better than chance, and the null (shuffled times) is near 0.5", () => {
    // 前半で作り、後半を予測する
    const n = c.X.length, half = n >> 1;
    const tr = all(half), te = all(n).slice(half);
    const f = survivalForest(surv, c.X, tr, all(c.X[0].length), { trees: 80, mtry: 3, minLeaf: 3 }, new Rng(5));
    const cs = cIndex({ time: te.map((i) => c.time[i]), event: te.map((i) => c.event[i]) }, te.map((i) => f.predict(c.X[i])));
    expect(cs!).toBeGreaterThan(0.6);
    const sh = new Rng(9);
    const perm = all(n).sort(() => sh.next() - 0.5);
    const surv0 = { time: perm.map((i) => c.time[i]), event: perm.map((i) => c.event[i]) };
    const f0 = survivalForest(surv0, c.X, tr, all(c.X[0].length), { trees: 80, mtry: 3, minLeaf: 3 }, new Rng(5));
    const c0 = cIndex({ time: te.map((i) => surv0.time[i]), event: te.map((i) => surv0.event[i]) }, te.map((i) => f0.predict(c.X[i])));
    expect(Math.abs((c0 as number) - 0.5)).toBeLessThan(0.2);
  });
  it("rejects bad settings and a training set without events", () => {
    expect(() => run({ trees: 0 })).toThrow(RangeError);
    expect(() => run({ mtry: 99 })).toThrow(RangeError);
    expect(() => run({ minLeaf: 0 })).toThrow(RangeError);
    expect(() => survivalForest({ time: [1, 2, 3, 4, 5, 6], event: [0, 0, 0, 0, 0, 0] }, [[1], [2], [3], [4], [5], [6]], all(6), [0], { trees: 1, mtry: 1, minLeaf: 1 }, new Rng(1))).toThrow(RangeError);
  });
});

describe("random survival forest with bootstrap and sqrt(features): the distribution of the held-out C-index matches scikit-survival", () => {
  it("mean over 20 seeds on a fixed split agrees within 3 standard errors; a forest with one feature per tree (a wrong variant) is worse", () => {
    const c = fx.cases[1], d = c.dist!;
    const surv = { time: c.time, event: c.event };
    const tr = all(d.split), te = all(c.X.length).slice(d.split);
    const mtry = Math.max(1, Math.floor(Math.sqrt(c.X[0].length)));
    const run = (seed: number, mt: number) => {
      const f = survivalForest(surv, c.X, tr, all(c.X[0].length), { trees: d.trees, mtry: mt, minLeaf: d.minLeaf, minSplit: d.minSplit }, new Rng(100 + seed));
      return cIndex({ time: te.map((i) => c.time[i]), event: te.map((i) => c.event[i]) }, te.map((i) => f.predict(c.X[i]))) as number;
    };
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    const sd = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + (b - mean(v)) ** 2, 0) / (v.length - 1));
    const ref = d.cindex, ts = Array.from({ length: 20 }, (_, s) => run(s, mtry));
    const se = Math.sqrt(sd(ref) ** 2 / 20 + sd(ts) ** 2 / 20);
    expect(Math.abs(mean(ts) - mean(ref)), `ts ${mean(ts).toFixed(4)} ref ${mean(ref).toFixed(4)} se ${se.toFixed(4)}`).toBeLessThan(3 * se + 1e-9);
    // 負例: mtry = 1 は分布が変わる（ref との差が、同じ標準誤差で見て大きい）
    const bad = Array.from({ length: 20 }, (_, s) => run(s, 1));
    expect(mean(ref) - mean(bad)).toBeGreaterThan(se);
  }, 120000);
});
