import { describe, expect, it } from "vitest";
import fx from "./fixtures/ml.json";
import { Rng } from "../src/core/rng";
import { auc } from "../src/core/discrim/roc";
import { buildTree, leafOf, normalize, rawImportance, type Tree } from "../src/core/model/tree";
import { randomForest, sqrtMtry, type ForestOptions } from "../src/core/model/randomForest";
import { gbm } from "../src/core/model/gbm";
import { scaleGamma, svmFit } from "../src/core/model/svm";

const X = fx.X as number[][], y = fx.y as number[];
const tr = fx.train as number[], te = fx.test as number[];
const cols = X[0].map((_, j) => j);
const ones = new Float64Array(X.length).fill(1);
const yte = te.map((i) => y[i]);
const close = (a: number[], b: number[], tol: number) => {
  expect(a.length).toBe(b.length);
  const d = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  expect(d).toBeLessThan(tol);
};
// 同じ定義・同じ float32 の値なら、違うのは足し算の順だけ。二重精度の丸め（1e-16）× 段数・行数を見込んで 1e-9。
const EXACT = 1e-9;

interface RefTree { feature: number[]; threshold: number[]; left: number[]; right: number[]; value: number[]; weight: number[] }

/** ノードの不純度の重み付き合計（Gini か二乗誤差）。sklearn とは別に、定義どおり計算する。 */
function impurityTimesW(S: number[], w: ArrayLike<number>, t: ArrayLike<number>, crit: "gini" | "mse") {
  let W = 0, s = 0, s2 = 0;
  for (const i of S) { W += w[i]; s += w[i] * t[i]; s2 += w[i] * t[i] * t[i]; }
  if (W === 0) return 0;
  const m = s / W;
  return W * (crit === "gini" ? 2 * m * (1 - m) : s2 / W - m * m);
}

/**
 * 木を sklearn の木と、根から同時にたどって照合する。各ノードで重みと値が一致し、分岐が同じなら子へ進む。
 * 分岐が違うときは、両方の分岐の不純度の減少が同じ（同点。sklearn は同点を内部の乱数の順で選ぶ）なら「同点」と数えて
 * そこで止め、違えば「不一致」と数える。
 */
function compareTree(ours: Tree, ref: RefTree, S0: number[], w: ArrayLike<number>, t: ArrayLike<number>, crit: "gini" | "mse") {
  let matched = 0, ties = 0, mismatch = 0;
  const walk = (a: number, b: number, S: number[]) => {
    const A = ours.nodes[a];
    if (Math.abs(A.weight - ref.weight[b]) > 1e-9 || Math.abs(A.value - ref.value[b]) > EXACT) { mismatch++; return; }
    matched++;
    if (A.feature < 0 && ref.feature[b] < 0) return;
    if (A.feature === ref.feature[b] && Math.abs(A.threshold - ref.threshold[b]) < 1e-12) {
      walk(A.left, ref.left[b], S.filter((i) => X[i][A.feature] <= A.threshold));
      walk(A.right, ref.right[b], S.filter((i) => X[i][A.feature] > A.threshold));
      return;
    }
    // 同じ特徴量で訓練の行の分け方も同じなのにしきい値が違う → 中点の取り方の誤り（同点ではない）
    const side = (f: number, th: number) => S.map((i) => X[i][f] <= th).join();
    if (A.feature === ref.feature[b] && side(A.feature, A.threshold) === side(A.feature, ref.threshold[b])) { mismatch++; return; }
    const gain = (f: number, th: number) => f < 0 ? 0 : impurityTimesW(S, w, t, crit) - impurityTimesW(S.filter((i) => X[i][f] <= th), w, t, crit) - impurityTimesW(S.filter((i) => X[i][f] > th), w, t, crit);
    const ga = gain(A.feature, A.threshold), gb = gain(ref.feature[b], ref.threshold[b]);
    if (A.feature >= 0 && ref.feature[b] >= 0 && Math.abs(ga - gb) <= 1e-9 * Math.max(1, Math.abs(gb))) ties++;
    else mismatch++;
  };
  walk(0, 0, S0);
  return { matched, ties, mismatch };
}

describe("CART vs scikit-learn DecisionTree", () => {
  for (const [name, c] of Object.entries(fx.cart)) {
    it(`${name}: every node agrees with sklearn, or is a tie between equally good splits`, () => {
      const w = new Float64Array(X.length);
      tr.forEach((i, q) => { w[i] = name === "weighted" ? fx.cartWeights[q] : 1; });
      const rows = tr.filter((i) => w[i] > 0);
      const t = buildTree(X, rows, w, y, cols, { criterion: "gini", maxDepth: c.depth, mtry: null }, null);
      const r = compareTree(t, c.tree, rows, w, y, "gini");
      expect(r.mismatch).toBe(0);
      expect(r.matched).toBeGreaterThan(c.tree.feature.length / 2);
    });
  }
  it("regression tree (squared error, depth 4) agrees with DecisionTreeRegressor", () => {
    const target = new Float64Array(X.length);
    tr.forEach((i, q) => { target[i] = fx.regTree.target[q]; });
    const t = buildTree(X, tr, ones, target, cols, { criterion: "mse", maxDepth: fx.regTree.depth, mtry: null }, null);
    const r = compareTree(t, fx.regTree.tree, tr, ones, target, "mse");
    expect(r.mismatch).toBe(0);
    expect(r.matched).toBeGreaterThan(fx.regTree.tree.feature.length / 2);
  });
  it("the comparison detects a tree that ignores the bootstrap weights", () => {
    // 負例: 重みを無視して木を作ると、重み付きの sklearn の木と不一致になる
    const rows = tr.filter((_, q) => fx.cartWeights[q] > 0);
    const t = buildTree(X, rows, ones, y, cols, { criterion: "gini", maxDepth: null, mtry: null }, null);
    const w = new Float64Array(X.length);
    tr.forEach((i, q) => { w[i] = fx.cartWeights[q]; });
    expect(compareTree(t, fx.cart.weighted.tree, rows, w, y, "gini").mismatch).toBeGreaterThan(0);
  });
});

describe("random forest", () => {
  it("without bootstrap and with all features, every tree is the plain CART", () => {
    const f = randomForest(X, y, tr, cols, { trees: 3, mtry: cols.length, bootstrap: false }, new Rng(1));
    const t = buildTree(X, tr, ones, y, cols, { criterion: "gini", maxDepth: null, mtry: null }, null);
    expect(te.map((i) => f.predict(X[i]))).toEqual(te.map((i) => t.nodes[leafOf(t, X[i])].value));
    close(f.importance, normalize(rawImportance(t, cols.length)), EXACT);
  });
  const Wd = fx.wide.X as number[][], yw = fx.wide.y as number[];
  const wcols = Wd[0].map((_, j) => j);
  const dist = (o: Partial<ForestOptions>) => Array.from({ length: fx.rfDist.auc.length }, (_, s) => {
    const f = randomForest(Wd, yw, fx.wide.train, wcols, { trees: fx.rfDist.trees, mtry: sqrtMtry(wcols.length), ...o }, new Rng(s + 1));
    return auc({ score: fx.wide.test.map((i) => f.predict(Wd[i])), label: fx.wide.test.map((i) => yw[i]) }) as number;
  });
  it("test-AUC distribution over 20 seeds agrees with sklearn; per-tree feature sampling does not", () => {
    const ok = sameDist(dist({}), fx.rfDist.auc), bad = sameDist(dist({ perTreeFeatures: true }), fx.rfDist.auc);
    expect(ok.z).toBeLessThan(3);
    expect(bad.z).toBeGreaterThan(3);
  }, 180000);
  it("importance is the mean of per-tree normalized impurity decreases, normalized again (sklearn's definition, checked in the fixture generator)", () => {
    const f = randomForest(X, y, tr, cols, { trees: 7, mtry: 3 }, new Rng(9));
    const per = f.trees.map((t) => normalize(rawImportance(t, cols.length)));
    close(f.importance, normalize(cols.map((j) => per.reduce((s, v) => s + v[j], 0) / per.length)), EXACT);
    const raw = normalize(cols.map((j) => f.trees.reduce((s, t) => s + rawImportance(t, cols.length)[j], 0)));
    expect(Math.max(...f.importance.map((v, j) => Math.abs(v - raw[j])))).toBeGreaterThan(1e-6);
  });
  it("is reproducible with the same seed and changes with another", () => {
    const a = randomForest(X, y, tr, cols, { trees: 20, mtry: 3 }, new Rng(5));
    const b = randomForest(X, y, tr, cols, { trees: 20, mtry: 3 }, new Rng(5));
    const c = randomForest(X, y, tr, cols, { trees: 20, mtry: 3 }, new Rng(6));
    expect(te.map((i) => b.predict(X[i]))).toEqual(te.map((i) => a.predict(X[i])));
    expect(te.map((i) => c.predict(X[i]))).not.toEqual(te.map((i) => a.predict(X[i])));
  });
});

/** 平均の差 ÷ 合わせた標準誤差。 */
function sameDist(a: number[], b: number[]) {
  const m = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
  const v = (x: number[]) => x.reduce((s, t) => s + (t - m(x)) ** 2, 0) / (x.length - 1);
  return { diff: m(a) - m(b), z: Math.abs(m(a) - m(b)) / Math.sqrt(v(a) / a.length + v(b) / b.length) };
}

describe("gradient boosting vs scikit-learn GradientBoostingClassifier", () => {
  for (const g of fx.gbmDet) {
    it(`subsample=1, ${g.stages} stages, depth ${g.depth}, rate ${g.lr}: ${g.invariant ? "probabilities and importances match" : "AUC within sklearn's range over random_state (ties matter)"}`, () => {
      const m = gbm(X, y, tr, cols, { stages: g.stages, maxDepth: g.depth, learningRate: g.lr, subsample: 1 }, new Rng(1));
      const bad = gbm(X, y, tr, cols, { stages: g.stages, maxDepth: g.depth, learningRate: g.lr, subsample: 1, meanLeaf: true }, new Rng(1));
      if (!g.invariant) {
        const a = auc({ score: te.map((i) => m.predict(X[i])), label: yte }) as number;
        expect(a).toBeGreaterThanOrEqual(g.aucRange[0] - 1e-12);
        expect(a).toBeLessThanOrEqual(g.aucRange[1] + 1e-12);
        return;
      }
      close(te.map((i) => m.raw(X[i])), g.raw, EXACT);
      close(te.map((i) => m.predict(X[i])), g.proba, EXACT);
      close(m.importance, g.importance, EXACT);
      expect(Math.max(...te.map((i, q) => Math.abs(bad.raw(X[i]) - g.raw[q])))).toBeGreaterThan(1e-3);
    });
  }
  it("subsample draws ⌊subsample·n⌋ distinct rows for every stage", () => {
    const m = gbm(X, y, tr, cols, { stages: 30, maxDepth: 2, learningRate: 0.1, subsample: 0.7 }, new Rng(4));
    for (const t of m.trees) expect(t.nodes[0].weight).toBe(Math.floor(0.7 * tr.length));
  });
  it("subsample<1: test-AUC distribution over 20 seeds agrees with sklearn", () => {
    const d = fx.gbmDist;
    const a = Array.from({ length: d.auc.length }, (_, s) => {
      const m = gbm(X, y, tr, cols, { stages: d.stages, maxDepth: d.depth, learningRate: d.lr, subsample: d.subsample }, new Rng(s + 1));
      return auc({ score: te.map((i) => m.predict(X[i])), label: yte }) as number;
    });
    expect(sameDist(a, d.auc).z).toBeLessThan(3);
  }, 60000);
});

describe("SVM vs scikit-learn SVC (libsvm)", () => {
  const Ztr = tr.map((i) => X[i]), ytr = tr.map((i) => y[i]);
  it("γ = 'scale' equals sklearn's value", () => {
    expect(scaleGamma(Ztr)).toBeCloseTo(fx.svm[1].gammaValue, 12);
  });
  for (const s of fx.svm) {
    it(`C=${s.C}, γ=${s.gamma}: decision values and support vectors match`, () => {
      const m = svmFit(Ztr, ytr, { C: s.C, gamma: s.gammaValue, tol: 1e-10 });
      expect(m.converged).toBe(true);
      // libsvm（sklearn）はカーネルの値を float32 で持つ（Qfloat）。こちらは倍精度なので、解は float32 の丸め
      // （相対 6e-8）の分だけ違う。カーネルを float32 に丸めて解くと差が 1e-10 未満になることを確かめた（実測の差は最大 1.04e-6）。
      close(te.map((i) => m.decision(X[i])), s.decision, 1e-5);
      expect(m.support).toEqual(s.support);
      expect(m.alpha.filter((a) => a >= s.C).length).toBe(s.nBound);
      if (s.nBound < s.support.length) {
        const bad = svmFit(Ztr, ytr, { C: s.C, gamma: s.gammaValue, tol: 1e-10, wrongRho: true });
        expect(Math.max(...te.map((i, q) => Math.abs(bad.decision(X[i]) - s.decision[q])))).toBeGreaterThan(1e-4);
      }
    });
  }
  it("the default tolerance gives the same AUC", () => {
    for (const s of fx.svm) {
      const m = svmFit(Ztr, ytr, { C: s.C, gamma: s.gammaValue });
      expect(auc({ score: te.map((i) => m.decision(X[i])), label: yte })).toBeCloseTo(s.auc, 10);
    }
  });
});
