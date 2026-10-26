import { describe, expect, it } from "vitest";
import fx from "./fixtures/discrim.json";
import { betaCdf, betaQuantile } from "../src/core/survival/dist";
import { proportion } from "../src/core/discrim/binom";
import { auc, aucCi, bestCut, bestDirection, confusion, performance, predictAt, rocPoints } from "../src/core/discrim/roc";
import { logisticFit } from "../src/core/discrim/logistic";
import { calibrationBins, calibrationSlope } from "../src/core/discrim/calibration";
import { decisionCurve } from "../src/core/discrim/dca";
import { looTopK, selectTopK } from "../src/core/discrim/loo";
import { buildTree, monteCarloCi, predictTree, treeNodes, treePerformance, type TreeNode } from "../src/core/discrim/tree";
import { Rng } from "../src/core/rng";

const close = (a: number, b: number, rel: number, abs = 0) => expect(Math.abs(a - b)).toBeLessThanOrEqual(abs + rel * Math.abs(b));
const col = (X: number[][], j: number) => X.map((r) => r[j]);

describe("beta distribution and Clopper–Pearson vs scipy", () => {
  it("cdf", () => { for (const b of fx.beta) close(betaCdf(b.x, b.a, b.b), b.cdf, 1e-9, 1e-14); });
  it("exact CI", () => {
    for (const c of fx.cp) {
      const p = proportion(c.x, c.n);
      close(p.lower!, c.lower, 1e-8, 1e-12); close(p.upper!, c.upper, 1e-8, 1e-12);
    }
    expect(proportion(0, 0).value).toBeNull(); // 0 件で 0 や 1 にしない
  });
  it("quantile inverts the cdf", () => { close(betaCdf(betaQuantile(0.3, 2.5, 4), 2.5, 4), 0.3, 1e-9); });
});

describe("ROC, AUC, DeLong vs scikit-learn and a brute-force definition", () => {
  it("AUC and ROC points", () => {
    for (const c of fx.cases) {
      c.auc.forEach((a, j) => close(auc({ score: col(c.X, j), label: c.y })!, a, 1e-12));
      const pts = rocPoints({ score: col(c.X, 0), label: c.y });
      expect(pts.map((p) => p.fpr)).toEqual(c.roc0.fpr.map((v) => expect.closeTo(v, 12)));
      expect(pts.map((p) => p.tpr)).toEqual(c.roc0.tpr.map((v) => expect.closeTo(v, 12)));
    }
  });
  it("DeLong standard error matches the pairwise definition", () => {
    for (const c of fx.cases) c.delong.forEach((d, j) => {
      const r = aucCi({ score: col(c.X, j), label: c.y })!;
      close(r.auc, d.auc, 1e-12); close(r.se, d.se, 1e-9);
    });
  });
  it("direction: a lower-is-positive feature gets AUC 1 − AUC", () => {
    const l = { score: [1, 2, 3, 4], label: [1, 1, 0, 0] };
    expect(auc(l, "higher")).toBe(0);
    expect(auc(l, "lower")).toBe(1);
    expect(bestDirection(l)).toBe("lower");
  });
  it("ties count one half", () => { expect(auc({ score: [1, 1, 1, 1], label: [1, 1, 0, 0] })).toBe(0.5); });
  it("returns null, not a number, when one class is missing (negative)", () => {
    expect(auc({ score: [1, 2], label: [1, 1] })).toBeNull();
    expect(aucCi({ score: [1, 2], label: [0, 0] })).toBeNull();
    expect(bestCut({ score: [1, 2], label: [1, 1] }, "higher", "accuracy")).toBeNull();
  });
  it("best cut-off by accuracy and by Youden, both directions", () => {
    for (const c of fx.cases) for (const crit of ["accuracy", "youden"] as const) for (const d of ["higher", "lower"] as const) {
      const ref = (c.cut as Record<string, { cut: number; tp: number; fp: number; tn: number; fn: number }>)[`${crit}-${d}`];
      const r = bestCut({ score: col(c.X, 0), label: c.y }, d, crit)!;
      close(r.cut, ref.cut, 1e-12);
      expect(r.confusion).toEqual({ tp: ref.tp, fp: ref.fp, tn: ref.tn, fn: ref.fn });
      expect(confusion(c.y, predictAt(col(c.X, 0), r.cut, d))).toEqual(r.confusion);
    }
  });
  it("performance summary with exact CIs", () => {
    const p = performance({ tp: 25, fn: 2, tn: 9, fp: 12 });
    close(p.sensitivity.value!, 25 / 27, 1e-12);
    close(p.accuracy.lower!, proportion(34, 48).lower!, 1e-12);
  });
});

describe("logistic regression vs statsmodels", () => {
  it("coefficients, SE, p, log-likelihood, predictions", () => {
    for (const c of fx.cases) {
      const cols = [0, 1];
      const mu = cols.map((j) => col(c.X, j).reduce((a, b) => a + b, 0) / c.X.length);
      const sd = cols.map((j, k) => Math.sqrt(col(c.X, j).reduce((a, b) => a + (b - mu[k]) ** 2, 0) / (c.X.length - 1)));
      const Z = c.X.map((r) => cols.map((j, k) => (r[j] - mu[k]) / sd[k]));
      const f = logisticFit(Z, c.y);
      if (!f.ok) throw new Error(f.error);
      c.logit.beta.forEach((b, a) => { close(f.fit.beta[a], b, 1e-6, 1e-8); close(f.fit.se[a], c.logit.se[a], 1e-6, 1e-8); close(f.fit.p[a], c.logit.p[a], 1e-5, 1e-9); });
      close(f.fit.loglik, c.logit.ll, 1e-9);
      Z.forEach((z, i) => close(f.fit.predict(z), c.logit.pred[i], 1e-7));
    }
  });
  it("stops on perfect separation and on a single class (negative)", () => {
    expect(logisticFit([[1], [2], [3], [4]], [0, 0, 1, 1]).ok).toBe(false);
    expect(logisticFit([[1], [2]], [1, 1]).ok).toBe(false);
    expect(logisticFit([[1], [NaN], [3], [4]], [0, 1, 0, 1]).ok).toBe(false);
  });
});

describe("calibration and decision curve vs brute force", () => {
  it("bins, slope and intercept", () => {
    for (const c of fx.cases) {
      const bins = calibrationBins(c.logit.pred, c.y, 10);
      expect(bins.length).toBe(c.calib.length);
      bins.forEach((b, k) => { expect(b.n).toBe(c.calib[k].n); close(b.meanPredicted, c.calib[k].mean, 1e-12); close(b.observed.value!, c.calib[k].obs, 1e-12); });
      const s = calibrationSlope(c.logit.pred, c.y);
      if ("error" in s) throw new Error(s.error);
      close(s.slope, c.calib_slope.slope, 1e-6); close(s.intercept, c.calib_slope.intercept, 1e-6, 1e-8);
    }
    expect("error" in calibrationSlope([0, 0.5, 1], [0, 1, 1])).toBe(true);
  });
  it("net benefit, treat-all and treat-none lines", () => {
    for (const c of fx.cases) {
      const d = decisionCurve(c.logit.pred, c.y, c.dca.map((x) => x.pt));
      d.forEach((p, k) => { close(p.model, c.dca[k].model, 1e-12, 1e-14); close(p.all, c.dca[k].all, 1e-12, 1e-14); expect(p.none).toBe(0); });
    }
  });
  it("hand calculation: pt = 0.5 weighs a false positive as much as a true positive", () => {
    const p = decisionCurve([0.9, 0.6, 0.4, 0.2], [1, 0, 1, 0], [0.5])[0];
    close(p.model, 1 / 4 - (1 / 4) * 1, 1e-12, 1e-14); // TP=1 FP=1
  });
});

describe("LOO with selection inside the fold", () => {
  it("predictions match statsmodels for both the inside and the outside selection", () => {
    for (const c of fx.cases) {
      const a = looTopK(c.X, c.y, { k: 2, alpha: null, selectInside: true }), b = looTopK(c.X, c.y, { k: 2, alpha: null, selectInside: false });
      expect(a.failed + b.failed).toBe(0);
      a.pred.forEach((p, i) => close(p!, c.loo_inside[i], 1e-6, 1e-8));
      b.pred.forEach((p, i) => close(p!, c.loo_outside[i], 1e-6, 1e-8));
    }
  });
  it("on random labels the outside selection inflates the AUC and the inside one stays near 0.5 (leak detector)", () => {
    const rng = new Rng(99);
    let inside = 0, outside = 0;
    const R = 12;
    for (let r = 0; r < R; r++) {
      const n = 40, m = 120;
      const X = Array.from({ length: n }, () => Array.from({ length: m }, () => rng.next()));
      const y: number[] = Array.from({ length: n }, (_, i) => (i < n / 2 ? 1 : 0));
      for (let i = n - 1; i > 0; i--) { const j = rng.int(i + 1); [y[i], y[j]] = [y[j], y[i]]; }
      const ai = auc({ score: looTopK(X, y, { k: 2, alpha: null, selectInside: true }).pred as number[], label: y })!;
      const ao = auc({ score: looTopK(X, y, { k: 2, alpha: null, selectInside: false }).pred as number[], label: y })!;
      inside += ai / R; outside += ao / R;
    }
    expect(outside).toBeGreaterThan(0.65);
    expect(Math.abs(inside - 0.5)).toBeLessThan(0.1);
  });
  it("with an alpha nobody meets, falls back to the training prevalence and says which features were chosen", () => {
    const c = fx.cases[0];
    const r = looTopK(c.X, c.y, { k: 2, alpha: 1e-12, selectInside: true });
    const prev = (c.y.reduce((a, b) => a + b, 0) - c.y[0]) / (c.y.length - 1);
    close(r.pred[0]!, prev, 1e-12);
    expect(r.selectionCounts.every((n) => n === 0)).toBe(true);
    expect(selectTopK(c.X, c.y, c.y.map((_, i) => i), 2, null).length).toBe(2);
  });
});

describe("decision tree (the lung-cancer paper's definition)", () => {
  const flat = (t: TreeNode) => treeNodes(t).map((n) => ({ depth: n.depth, n: n.n, pos: n.pos, split: n.split ? { feature: n.split.feature, direction: n.split.direction, cut: n.split.cut } : null, label: n.label }));
  it("matches an independent brute-force tree on three datasets", () => {
    for (const c of fx.cases) {
      const t = buildTree(c.X, c.y, { stopRate: 0.15, minNode: 5, maxDepth: 6, candidates: [0, 1, 2, 3] });
      const got = flat(t), ref = c.tree;
      expect(got.length).toBe(ref.length);
      got.forEach((g, k) => {
        expect([g.depth, g.n, g.pos, g.label]).toEqual([ref[k].depth, ref[k].n, ref[k].pos, ref[k].label]);
        if (ref[k].split) { expect(g.split!.feature).toBe(ref[k].split.feature); expect(g.split!.direction).toBe(ref[k].split.direction); close(g.split!.cut, ref[k].split.cut, 1e-12); }
        else expect(g.split).toBeNull();
      });
    }
  });
  it("hand example: a perfectly separable feature gives one node and two pure leaves", () => {
    const X = [[1], [2], [3], [10], [11], [12], [13], [14]], y: number[] = [0, 0, 0, 1, 1, 1, 1, 1];
    const t = buildTree(X, y, { stopRate: 0.15, minNode: 2, maxDepth: 6, candidates: [0] });
    expect(treeNodes(t).length).toBe(3);
    expect(t.split).toMatchObject({ feature: 0, direction: "higher", cut: 6.5 });
    const p = treePerformance(t, X, y);
    expect(p.accuracy.value).toBe(1);
    expect(predictTree(t, [100])).toBe(1);
  });
  it("a branch with misclassification rate under 15% gets no further node; at 15% or more it does (boundary)", () => {
    // 右のグループ（x0 = 100 で全員同じ）は 20 例。偽陽性 fp 例は x1 だけで分けられる。
    // 陽性側の枝の誤分類率 = fp / 20: 3 例（15%）なら節点を足し、2 例（10%）なら止める
    const mk = (fp: number) => {
      const X: number[][] = [], y: number[] = [];
      for (let i = 0; i < 10; i++) { X.push([i, 0]); y.push(0); }
      for (let i = 0; i < 20 - fp; i++) { X.push([100, 0]); y.push(1); }
      for (let i = 0; i < fp; i++) { X.push([100, i % 2 ? 5 : 6]); y.push(0); }
      return buildTree(X, y, { stopRate: 0.15, minNode: 2, maxDepth: 6, candidates: [0, 1] });
    };
    expect(treeNodes(mk(2)).filter((n) => n.split).length).toBe(1);
    expect(treeNodes(mk(3)).filter((n) => n.split).length).toBe(2);
  });
  it("Monte Carlo CI: same seed same interval, another seed differs, and it brackets the point estimate; close to an independent simulation", () => {
    const X = fx.mc.X, y = fx.mc.y;
    const t = buildTree(X, y, { stopRate: 0.15, minNode: 5, maxDepth: 6, candidates: [0, 1, 2, 3] });
    const a = monteCarloCi(t, X, y, 3000, new Rng(1)), b = monteCarloCi(t, X, y, 3000, new Rng(1)), c = monteCarloCi(t, X, y, 3000, new Rng(2));
    expect(b).toEqual(a);
    expect(c.means).not.toEqual(a.means); // 区間の端は離散値で一致し得るので、平均で別シードを見分ける
    const pt = treePerformance(t, X, y);
    expect(a.accuracy.lower).toBeLessThan(pt.accuracy.value!);
    expect(a.accuracy.upper).toBeGreaterThan(pt.accuracy.value!);
    const ref = fx.mc.ref;
    for (const [k, r] of [["sensitivity", ref.sens], ["specificity", ref.spec], ["accuracy", ref.acc]] as const) {
      expect(Math.abs(a[k].lower - r[0])).toBeLessThan(0.04);
      expect(Math.abs(a[k].upper - r[1])).toBeLessThan(0.04);
    }
  });
});
