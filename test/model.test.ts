import { describe, expect, it } from "vitest";
import fx from "./fixtures/model.json";
import { Rng } from "../src/core/rng";
import { coxFit } from "../src/core/survival/cox";
import { lassoAt, lassoCv, fitStd, zRow, lambdaMax, likOf } from "../src/core/model/lasso";
import { coxBoost } from "../src/core/model/coxBoost";
import { backwardSelect } from "../src/core/model/backward";
import { schoenfeld } from "../src/core/model/schoenfeld";
import { breslow, riskSets } from "../src/core/model/breslow";

const all = (n: number) => Array.from({ length: n }, (_, i) => i);
const cols = (X: number[][]) => all(X[0].length);

describe("LASSO-Cox（Breslow）vs statsmodels PHReg.fit_regularized", () => {
  const c = fx.lassoCox;
  const surv = { time: c.time, event: c.event };
  const task = { kind: "survival" as const, surv };
  it("λmax matches the reference and gives all-zero coefficients", () => {
    const st = fitStd(c.X, all(c.X.length), cols(c.X));
    if ("error" in st) throw new Error(st.error);
    const Z = c.X.map((r) => zRow(st, r));
    expect(lambdaMax(likOf(task, all(c.X.length)), Z).lmax).toBeCloseTo(c.lambdaMax, 10);
    const r = lassoAt(task, c.X, all(c.X.length), cols(c.X), c.lambdaMax * 1.0001);
    if ("error" in r) throw new Error(r.error);
    expect(r.coef.beta.every((b) => b === 0)).toBe(true);
  });
  for (const [k, f] of c.fits.entries()) {
    it(`coefficients and zero pattern match at λ #${k + 1}`, () => {
      const r = lassoAt(task, c.X, all(c.X.length), cols(c.X), f.lambda);
      if ("error" in r) throw new Error(r.error);
      // 参照の KKT 違反 ~1e-13、こちらの収束判定 1e-11 → 係数の差は 1e-6 以内を求める
      r.coef.beta.forEach((b, j) => expect(Math.abs(b - f.beta[j])).toBeLessThan(1e-6));
      expect(r.coef.beta.map((b) => b !== 0)).toEqual(f.beta.map((b) => b !== 0));
    });
  }
  it("a tiny λ approaches the unpenalized Breslow Cox", () => {
    const r = lassoAt(task, c.X, all(c.X.length), cols(c.X), 1e-9);
    if ("error" in r) throw new Error(r.error);
    r.coef.beta.forEach((b, j) => expect(Math.abs(b - c.breslowBeta[j])).toBeLessThan(1e-6));
  });
});

describe("ties are handled as Breslow (not Efron)", () => {
  const c = fx.ties;
  const surv = { time: c.time, event: c.event };
  it("the reference data really has many ties and the two methods differ", () => {
    expect(c.distinctTimes).toBeLessThan(c.time.length / 2);
    expect(Math.abs(c.breslowBeta[0] - c.efronBeta[0])).toBeGreaterThan(0.01);
  });
  it("LASSO-Cox with a tiny λ and boosting both reach the Breslow estimate", () => {
    const r = lassoAt({ kind: "survival", surv }, c.X, all(100), cols(c.X), 1e-9);
    if ("error" in r) throw new Error(r.error);
    r.coef.beta.forEach((b, j) => expect(Math.abs(b - c.breslowBeta[j])).toBeLessThan(1e-6));
    const g = coxBoost(surv, c.X, all(100), cols(c.X), { iterations: 20000, nu: 0.5 });
    if ("error" in g) throw new Error(g.error);
    g.beta.forEach((b, j) => expect(Math.abs(b - c.breslowBeta[j])).toBeLessThan(1e-4));
  });
});

describe("LASSO logistic vs scikit-learn (saga, unpenalized intercept)", () => {
  const c = fx.lassoLogistic;
  const task = { kind: "binary" as const, y: c.y };
  for (const [k, f] of c.fits.entries()) {
    it(`intercept, coefficients and zero pattern match at λ #${k + 1}`, () => {
      const r = lassoAt(task, c.X, all(c.X.length), cols(c.X), f.lambda);
      if ("error" in r) throw new Error(r.error);
      expect(Math.abs(r.coef.b0 - f.b0)).toBeLessThan(1e-6);
      r.coef.beta.forEach((b, j) => expect(Math.abs(b - f.beta[j])).toBeLessThan(1e-6));
      expect(r.coef.beta.map((b) => b !== 0)).toEqual(f.beta.map((b) => b !== 0));
      // 正確なヘッセ行列のニュートン法なので、0 から始めても数回で収束する（実測 4〜5 回。対角だけの近似では 25 回以上）
      expect(r.iterations).toBeLessThanOrEqual(10);
    });
  }
});

describe("LASSO cross-validation", () => {
  const c = fx.lassoCox;
  const task = { kind: "survival" as const, surv: { time: c.time, event: c.event } };
  const o = { nLambda: 30, ratio: 0.01, folds: 5, rule: "1se" as const };
  it("is reproducible with the same seed and the 1SE λ is not smaller than the min λ", () => {
    const a = lassoCv(task, c.X, all(90), cols(c.X), o, new Rng(3));
    const b = lassoCv(task, c.X, all(90), cols(c.X), o, new Rng(3));
    const d = lassoCv(task, c.X, all(90), cols(c.X), o, new Rng(4));
    if ("error" in a || "error" in b || "error" in d) throw new Error("cv failed");
    expect(b.cvm).toEqual(a.cvm);
    expect(d.foldOf).not.toEqual(a.foldOf);
    expect(a.lambdas[a.idx1se]).toBeGreaterThanOrEqual(a.lambdas[a.idxMin]);
    expect(a.lambda).toBe(a.lambdas[a.idx1se]);
    // 本当に効く 2 列（0, 1）は選ばれる
    expect(a.nonzero).toEqual(expect.arrayContaining([0, 1]));
  });
  it("uses only the given rows: changing a row outside them changes nothing", () => {
    const rows = all(80);
    const a = lassoCv(task, c.X, rows, cols(c.X), o, new Rng(5));
    const X2 = c.X.map((r) => [...r]);
    X2[85][0] = 1e6;
    const t2 = { kind: "survival" as const, surv: { time: [...c.time], event: [...c.event] } };
    t2.surv.time[86] = 1e-4;
    const b = lassoCv(t2, X2, rows, cols(c.X), o, new Rng(5));
    if ("error" in a || "error" in b) throw new Error("cv failed");
    expect(b.lambdas).toEqual(a.lambdas);
    expect(b.cvm).toEqual(a.cvm);
    expect(b.std.means).toEqual(a.std.means);
  });
  it("CV deviance equals an independent per-fold computation (binary and survival)", () => {
    const l = fx.lassoLogistic;
    const cases = [
      { tk: { kind: "binary" as const, y: l.y } as const, X: l.X },
      { tk: task, X: c.X },
    ];
    for (const { tk, X } of cases) {
      const n = X.length;
      const r = lassoCv(tk, X, all(n), cols(X), { nLambda: 6, ratio: 0.05, folds: 4, rule: "min" }, new Rng(9));
      if ("error" in r) throw new Error(r.error);
      r.lambdas.forEach((lam, a) => {
        const raw: { w: number; v: number }[] = [];
        for (let f = 0; f < 4; f++) {
          const tr = all(n).filter((i) => r.foldOf[i] !== f), te = all(n).filter((i) => r.foldOf[i] === f);
          const m = lassoAt(tk, X, tr, cols(X), lam);
          if ("error" in m) throw new Error(m.error);
          const eta = (i: number) => zRow(m.std, X[i]).reduce((s, v, j) => s + v * m.coef.beta[j], m.coef.b0);
          if (tk.kind === "binary") {
            const dev = te.reduce((s, i) => { const e = eta(i); return s - 2 * (tk.y[i] * e - Math.log1p(Math.exp(e))); }, 0);
            raw.push({ w: te.length, v: dev / te.length });
          } else {
            const sAll = tk.surv, sTr = { time: tr.map((i) => sAll.time[i]), event: tr.map((i) => sAll.event[i]) };
            const lAll = breslow(sAll, riskSets(sAll), all(n).map(eta)).ll;
            const lTr = breslow(sTr, riskSets(sTr), tr.map(eta)).ll;
            const e = te.reduce((s, i) => s + sAll.event[i], 0);
            raw.push({ w: e, v: (-2 * (lAll - lTr)) / e });
          }
        }
        const W = raw.reduce((s, x) => s + x.w, 0);
        const m = raw.reduce((s, x) => s + x.w * x.v, 0) / W;
        const sd = Math.sqrt(raw.reduce((s, x) => s + x.w * (x.v - m) ** 2, 0) / W / 3);
        expect(Math.abs((r.cvm[a] as number) - m)).toBeLessThan(1e-7);
        expect(Math.abs((r.cvsd[a] as number) - sd)).toBeLessThan(1e-7);
      });
    }
  });
  it("binary: the min rule picks the λ with the smallest CV deviance", () => {
    const l = fx.lassoLogistic;
    const r = lassoCv({ kind: "binary", y: l.y }, l.X, all(120), cols(l.X), { ...o, rule: "min" }, new Rng(1));
    if ("error" in r) throw new Error(r.error);
    const m = Math.min(...r.cvm.filter((x): x is number => x !== null));
    expect(r.cvm[r.chosen]).toBe(m);
    expect(r.predict(l.X[0])).toBeGreaterThan(0);
    expect(r.predict(l.X[0])).toBeLessThan(1);
  });
});

describe("Cox boosting", () => {
  const c = fx.lassoCox;
  const surv = { time: c.time, event: c.event };
  it("converges to the unpenalized Breslow Cox with many iterations (low dimension)", () => {
    const r = coxBoost(surv, c.X, all(90), cols(c.X), { iterations: 20000, nu: 0.5 });
    if ("error" in r) throw new Error(r.error);
    r.beta.forEach((b, j) => expect(Math.abs(b - c.breslowBeta[j])).toBeLessThan(1e-4));
    // −対数部分尤度は単調に下がる
    for (let m = 1; m < r.risk.length; m++) expect(r.risk[m]).toBeLessThanOrEqual(r.risk[m - 1] + 1e-12);
  });
  it("the first step picks the largest |score| at β=0, and importance sums to the total risk reduction", () => {
    const r = coxBoost(surv, c.X, all(90), cols(c.X), { iterations: 100, nu: 0.1 });
    if ("error" in r) throw new Error(r.error);
    const st = fitStd(c.X, all(90), cols(c.X));
    if ("error" in st) throw new Error(st.error);
    const Z = c.X.map((x) => zRow(st, x));
    const ev = breslow(surv, riskSets(surv), new Array(90).fill(0));
    const U = Z[0].map((_, j) => Math.abs(Z.reduce((a, z, k) => a + z[j] * ev.g[k], 0)));
    expect(r.selected[0]).toBe(U.indexOf(Math.max(...U)));
    const tot = r.importance.reduce((a, b) => a + b, 0);
    expect(tot).toBeCloseTo(r.risk[0] - r.risk[100], 10);
    expect(r.ranking.length).toBe(r.beta.filter((b) => b !== 0).length);
  });
  it("one step moves the chosen coefficient by ν·Σxu/Σx²", () => {
    const st = fitStd(c.X, all(90), cols(c.X));
    if ("error" in st) throw new Error(st.error);
    const Z = c.X.map((x) => zRow(st, x));
    const ev = breslow(surv, riskSets(surv), new Array(90).fill(0));
    for (const nu of [0.1, 0.37]) {
      const r = coxBoost(surv, c.X, all(90), cols(c.X), { iterations: 1, nu });
      if ("error" in r) throw new Error(r.error);
      const j = r.selected[0];
      const xu = Z.reduce((a, z, k) => a + z[j] * ev.g[k], 0), xx = Z.reduce((a, z) => a + z[j] * z[j], 0);
      expect(r.beta[j]).toBeCloseTo((nu * xu) / xx, 12);
    }
  });
});

describe("backward selection vs statsmodels (Efron)", () => {
  const c = fx.backward;
  const surv = { time: c.time, event: c.event };
  it("Efron Cox on all 8 standardized variables matches statsmodels", () => {
    const st = fitStd(c.X, all(150), cols(c.X));
    if ("error" in st) throw new Error(st.error);
    const r = coxFit(surv, c.X.map((x) => zRow(st, x)));
    if (!r.ok) throw new Error(r.error);
    r.fit.beta.forEach((b, j) => expect(Math.abs(b - c.efronAllBeta[j])).toBeLessThan(1e-6));
    r.fit.p.forEach((p, j) => expect(Math.abs(p - c.efronAllP[j])).toBeLessThan(1e-6));
  });
  for (const run of c.runs) {
    it(`follows the same steps and keeps the same variables (in ${run.pIn}, out ${run.pOut}, max ${run.maxVars})`, () => {
      const r = backwardSelect(surv, c.X, all(150), cols(c.X), { pIn: run.pIn, pOut: run.pOut, maxVars: run.maxVars });
      if ("error" in r) throw new Error(r.error);
      expect(r.steps.map((s) => [s.action, s.col])).toEqual(run.steps.map((s) => [s.action, s.col]));
      r.steps.forEach((s, k) => expect(Math.abs((s.p as number) - run.steps[k].p)).toBeLessThan(1e-6));
      expect(r.cols).toEqual(run.cols);
    });
  }
  it("caps the number of variables, removing the largest p first", () => {
    const run = c.runs[0];
    const r = backwardSelect(surv, c.X, all(150), cols(c.X), { pIn: 0.1, pOut: 0.15, maxVars: 2 });
    if ("error" in r) throw new Error(r.error);
    expect(r.cols.length).toBe(2);
    const caps = r.steps.filter((s) => s.action === "cap");
    expect(caps.length).toBe(run.preCap.length - 2);
    expect(r.cols).toEqual(run.preCap.filter((x) => !caps.some((s) => s.col === x)));
  });
  it("rejects thresholds where the entry p is not below the exit p", () => {
    expect(backwardSelect(surv, c.X, all(150), cols(c.X), { pIn: 0.15, pOut: 0.1, maxVars: 4 })).toHaveProperty("error");
  });
});

describe("Schoenfeld residuals vs statsmodels", () => {
  const c = fx.schoenfeld;
  const surv = { time: c.time, event: c.event };
  const st = fitStd(c.X, all(c.X.length), cols(c.X));
  if ("error" in st) throw new Error(st.error);
  const Z = c.X.map((x) => zRow(st, x));
  it("residuals, scaled residuals and the tests match", () => {
    const r = schoenfeld(surv, Z, c.beta, c.cov);
    if ("error" in r) throw new Error(r.error);
    expect(r.rows).toEqual(c.rows);
    r.resid.forEach((row, i) => row.forEach((v, j) => expect(Math.abs(v - c.resid[i][j])).toBeLessThan(1e-9)));
    r.scaled.forEach((row, i) => row.forEach((v, j) => expect(Math.abs(v - c.scaled[i][j])).toBeLessThan(1e-8)));
    r.tests.forEach((t, j) => { expect(t.chi2).toBeCloseTo(c.chi2[j], 8); expect(t.p).toBeCloseTo(c.p[j], 8); });
    expect(r.global.chi2).toBeCloseTo(c.globalChi2, 8);
    expect(r.global.p).toBeCloseTo(c.globalP, 8);
  });
});

/** 区分的に指数の生存時間（t<cut ではハザード exp(b1·x)、以降 exp(b2·x)）。打ち切りなし。 */
function piecewise(rng: Rng, n: number, b1: number, b2: number, cut: number) {
  const x: number[] = [], time: number[] = [], event: number[] = [];
  for (let i = 0; i < n; i++) {
    const u1 = rng.next(), u2 = rng.next() || 1e-12;
    const xi = Math.sqrt(-2 * Math.log(u2)) * Math.cos(2 * Math.PI * u1);
    const E = -Math.log(rng.next() || 1e-12);
    const l1 = Math.exp(b1 * xi), l2 = Math.exp(b2 * xi);
    time.push(E < l1 * cut ? E / l1 : cut + (E - l1 * cut) / l2);
    x.push(xi);
    event.push(1);
  }
  return { surv: { time, event }, Z: x.map((v) => [v]) };
}

describe("Schoenfeld test: positive and negative controls", () => {
  it("detects an effect that reverses over time (p < 0.001)", () => {
    const d = piecewise(new Rng(1), 300, 1, -1, 0.5);
    const f = coxFit(d.surv, d.Z);
    if (!f.ok) throw new Error(f.error);
    const r = schoenfeld(d.surv, d.Z, f.fit.beta, f.fit.cov);
    if ("error" in r) throw new Error(r.error);
    expect(r.tests[0].p).toBeLessThan(0.001);
  });
  it("rejects at about the nominal rate when hazards are proportional", () => {
    const rng = new Rng(2);
    let rej = 0;
    const N = 200;
    for (let k = 0; k < N; k++) {
      const d = piecewise(rng, 100, 0.7, 0.7, 0.5);
      const f = coxFit(d.surv, d.Z);
      if (!f.ok) throw new Error(f.error);
      const r = schoenfeld(d.surv, d.Z, f.fit.beta, f.fit.cov);
      if ("error" in r) throw new Error(r.error);
      if (r.tests[0].p < 0.05) rej++;
    }
    // 二項 (200, 0.05) の 99.9% の範囲はおよそ 2〜22
    expect(rej).toBeGreaterThanOrEqual(2);
    expect(rej).toBeLessThanOrEqual(22);
  });
});

