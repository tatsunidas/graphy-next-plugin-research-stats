import { describe, expect, it } from "vitest";
import fx from "./fixtures/survival.json";
import { chi2Sf, normalCdf, normalQuantile, normalTwoSided } from "../src/core/survival/dist";
import { atRiskAt, kaplanMeier, survAt } from "../src/core/survival/km";
import { logrank } from "../src/core/survival/logrank";
import { coxFit, lrDifference } from "../src/core/survival/cox";
import { bootstrapCorrectedC, cIndex } from "../src/core/survival/cindex";
import { rmst, rmstDifference } from "../src/core/survival/rmst";
import { mannWhitney } from "../src/core/survival/mannwhitney";
import { benjaminiHochberg } from "../src/core/survival/multiplicity";
import { groupBy } from "../src/core/survival/grouping";
import { CutScanner, permutationCorrectedP } from "../src/core/survival/permutation";
import { Rng } from "../src/core/rng";
import { checkSurv } from "../src/core/survival/types";

const close = (a: number, b: number, rel: number, abs = 0) => expect(Math.abs(a - b)).toBeLessThanOrEqual(abs + rel * Math.abs(b));

describe("distributions vs scipy", () => {
  it("chi2 / normal / quantile", () => {
    for (const d of fx.chi2) close(chi2Sf(d.x, d.df), d.sf, 1e-9);
    for (const d of fx.normal) { close(normalTwoSided(d.z), d.two, 1e-9); close(normalCdf(d.z), d.cdf, 1e-9, 1e-15); }
    for (const d of fx.quantile) close(normalQuantile(d.p), d.z, 1e-9, 1e-12);
  });
});

describe("Kaplan–Meier", () => {
  it("matches a hand calculation (times 1..5, events 1,0,1,1,0)", () => {
    const km = kaplanMeier({ time: [1, 2, 3, 4, 5], event: [1, 0, 1, 1, 0] });
    const s = km.points.map((p) => p.surv);
    close(s[0], 0.8, 1e-12); close(s[2], 0.8 * (2 / 3), 1e-12); close(s[3], 0.8 * (2 / 3) * 0.5, 1e-12);
    close(km.points[0].se, 0.8 * Math.sqrt(1 / 20), 1e-12);
    expect(km.median).toBe(4); // S(4) = 0.2667 が最初に 0.5 以下
    expect(atRiskAt({ time: [1, 2, 3, 4, 5], event: [1, 0, 1, 1, 0] }, 3)).toBe(3);
    expect(survAt(km, 0.5)).toBe(1);
  });
  it("matches statsmodels on tied, censored data", () => {
    for (const c of fx.cases) {
      const km = kaplanMeier({ time: c.time, event: c.event });
      const ev = km.points.filter((p) => p.events > 0);
      expect(ev.length).toBe(c.km.times.length);
      ev.forEach((p, i) => {
        close(p.time, c.km.times[i], 1e-12);
        close(p.surv, c.km.surv[i], 1e-10);
        if (c.km.se[i] !== null) close(p.se, c.km.se[i] as number, 1e-8, 1e-12);
        expect(p.atRisk).toBe(c.km.nrisk[i]);
      });
    }
  });
  it("changes when every censoring is turned into an event (negative control)", () => {
    const c = fx.cases[0];
    const a = kaplanMeier({ time: c.time, event: c.event });
    const b = kaplanMeier({ time: c.time, event: c.event.map(() => 1) });
    expect(Math.abs(a.points[a.points.length - 1].surv - b.points[b.points.length - 1].surv)).toBeGreaterThan(0.01);
  });
  it("CI stays inside [0, 1] and brackets the estimate", () => {
    const km = kaplanMeier(fx.cases[1] as never);
    for (const p of km.points) { expect(p.lower).toBeGreaterThanOrEqual(0); expect(p.upper).toBeLessThanOrEqual(1); expect(p.lower).toBeLessThanOrEqual(p.surv + 1e-12); expect(p.upper).toBeGreaterThanOrEqual(p.surv - 1e-12); }
  });
});

describe("log-rank vs statsmodels survdiff", () => {
  it("2 and 3 groups", () => {
    for (const c of fx.cases) {
      for (const [name, k] of [["lr2", 2], ["lr3", 3]] as const) {
        const ref = c[name];
        const r = logrank({ time: c.time, event: c.event }, ref.group, k);
        if ("error" in r) throw new Error(r.error);
        close(r.chi2, ref.chi2, 1e-8); close(r.p, ref.p, 1e-7, 1e-14);
      }
    }
  });
  it("reports an error instead of a number when there is one group", () => {
    expect("error" in logrank({ time: [1, 2, 3], event: [1, 1, 1] }, [0, 0, 0], 2)).toBe(true);
  });
});

describe("Cox (Efron) vs statsmodels PHReg", () => {
  it("univariate and multivariate coefficients, SE, log-likelihoods, C-index", () => {
    for (const c of fx.cases) {
      for (const name of ["cox1", "cox3"] as const) {
        const ref = c[name];
        const X = c.x.map((row) => ref.cols.map((j) => row[j]));
        const r = coxFit({ time: c.time, event: c.event }, X);
        if (!r.ok) throw new Error(r.error);
        ref.beta.forEach((b, a) => { close(r.fit.beta[a], b, 1e-6, 1e-8); close(r.fit.se[a], ref.se[a], 1e-6, 1e-8); });
        close(r.fit.loglik, ref.llf, 1e-9); close(r.fit.loglik0, ref.ll0, 1e-9);
        const ci = cIndex({ time: c.time, event: c.event }, r.fit.linearPredictor);
        close(ci!, ref.cindex, 1e-12);
      }
    }
  });
  it("stops with an error on perfect separation (negative control), without NaN", () => {
    const r = coxFit({ time: [1, 2, 3, 4, 5, 6], event: [1, 1, 1, 1, 1, 1] }, [[1], [1], [1], [0], [0], [0]]);
    expect(r.ok).toBe(false);
  });
  it("rejects constant covariates and missing values", () => {
    expect(coxFit({ time: [1, 2, 3], event: [1, 1, 0] }, [[1], [1], [1]]).ok).toBe(false);
    expect(coxFit({ time: [1, 2, 3], event: [1, 1, 0] }, [[1], [NaN], [3]]).ok).toBe(false);
  });
  it("LR difference of nested models", () => {
    const c = fx.cases[0];
    const s = { time: c.time, event: c.event };
    const a = coxFit(s, c.x.map((r) => [r[1]])), b = coxFit(s, c.x.map((r) => [r[1], r[0]]));
    if (!a.ok || !b.ok) throw new Error("fit");
    const d = lrDifference(a.fit, b.fit, 1);
    close(d.chi2, 2 * (b.fit.loglik - a.fit.loglik), 1e-12);
    expect(d.p).toBeGreaterThan(0); expect(d.p).toBeLessThan(1);
  });
});

describe("RMST", () => {
  it("mean matches the integral of the statsmodels curve; SE is close to the bootstrap SE", () => {
    for (const c of fx.cases) {
      const km = kaplanMeier({ time: c.time, event: c.event });
      const r = rmst(km, c.rmst.tau);
      if ("error" in r) throw new Error(r.error);
      close(r.value, c.rmst.value, 1e-10);
      close(r.se, c.rmst.bootSe, 0.15); // 解析式とブートストラップは別の経路。有限標本なので 15% 以内
    }
  });
  it("refuses a tau beyond the follow-up and differences the two groups", () => {
    const km = kaplanMeier({ time: [1, 2, 3], event: [1, 0, 1] });
    expect("error" in rmst(km, 10)).toBe(true);
    const d = rmstDifference({ time: [1, 2, 3, 4], event: [1, 1, 1, 0] }, { time: [2, 4, 6, 8], event: [1, 1, 1, 0] }, 4);
    if ("error" in d) throw new Error(d.error);
    expect(d.diff).toBeGreaterThan(0);
  });
});

describe("Mann–Whitney, BH", () => {
  it("matches scipy (asymptotic, ties)", () => {
    for (const m of fx.mannwhitney) {
      const r = mannWhitney(m.x, m.y);
      if ("error" in r) throw new Error(r.error);
      close(r.p, m.p, 1e-8, 1e-12);
    }
  });
  it("BH matches scipy and skips missing p", () => {
    const q = benjaminiHochberg(fx.bh.p);
    q.forEach((v, i) => close(v!, fx.bh.q[i], 1e-12));
    const withNull = benjaminiHochberg([0.01, null, 0.04]);
    expect(withNull[1]).toBeNull();
    close(withNull[0]!, 0.02, 1e-12); // m = 2（null は数えない）
  });
});

describe("grouping", () => {
  it("median, value, tertile; ties that empty a group are errors", () => {
    const v = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const m = groupBy(v, { kind: "median" });
    if ("error" in m) throw new Error(m.error);
    expect(m.group).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1]); // 中央値 5 は低群
    const t = groupBy(v, { kind: "tertile" });
    if ("error" in t) throw new Error(t.error);
    expect(t.k).toBe(3);
    expect("error" in groupBy([1, 1, 1, 1], { kind: "median" })).toBe(true);
    expect("error" in groupBy([1, NaN, 3], { kind: "median" })).toBe(true);
  });
});

describe("optimal cut-off and the permutation-corrected p", () => {
  it("scanner agrees with the general log-rank at its best cut", () => {
    const c = fx.cases[0];
    const s = { time: c.time, event: c.event };
    const vals = c.x.map((r) => r[0]);
    const sc = new CutScanner(s, 0.1);
    const best = sc.scan(sc.inTimeOrder(vals));
    const g = groupBy(vals, { kind: "optimal", cut: best.cut });
    if ("error" in g) throw new Error(g.error);
    const lr = logrank(s, g.group, 2);
    if ("error" in lr) throw new Error(lr.error);
    close(best.chi2, lr.chi2, 1e-9);
  });
  it("raw minimum-p is anti-conservative on noise, corrected p is calibrated (detector)", () => {
    const rng = new Rng(2024);
    let rawSig = 0, corrSig = 0;
    const R = 100;
    for (let r = 0; r < R; r++) {
      const n = 60;
      const time = Array.from({ length: n }, () => -Math.log(1 - rng.next()));
      const event = Array.from({ length: n }, () => (rng.next() < 0.7 ? 1 : 0));
      const noise = Array.from({ length: n }, () => rng.next());
      const sc = new CutScanner({ time, event }, 0.1);
      const res = permutationCorrectedP(sc, noise, 199, rng);
      if ("error" in res) throw new Error(res.error);
      if (res.rawP < 0.05) rawSig++;
      if (res.correctedP < 0.05) corrSig++;
    }
    expect(rawSig / R).toBeGreaterThan(0.2); // 探索した素の p は、ノイズでも 5% よりずっと多く有意になる
    expect(corrSig / R).toBeLessThan(0.12);   // 補正 p は約 5%
  });
  it("same seed gives the same corrected p; another seed differs", () => {
    const c = fx.cases[1];
    const sc = new CutScanner({ time: c.time, event: c.event }, 0.1);
    const v = c.x.map((r) => r[1]);
    const a = permutationCorrectedP(sc, v, 99, new Rng(5)), b = permutationCorrectedP(sc, v, 99, new Rng(5)), d = permutationCorrectedP(sc, v, 99, new Rng(6));
    if ("error" in a || "error" in b || "error" in d) throw new Error("err");
    expect(a.correctedP).toBe(b.correctedP);
    expect(a.permutations).toBe(99);
    expect(d.correctedP).not.toBe(a.correctedP);
  });
  it("a strong effect keeps a small corrected p", () => {
    const rng = new Rng(3);
    const n = 80;
    const x = Array.from({ length: n }, () => rng.next());
    const time = x.map((v) => -Math.log(1 - rng.next()) / Math.exp(3 * (v - 0.5)));
    const sc = new CutScanner({ time, event: time.map(() => 1) }, 0.1);
    const r = permutationCorrectedP(sc, x, 199, rng);
    if ("error" in r) throw new Error(r.error);
    expect(r.correctedP).toBeLessThan(0.05);
  });
  it("no admissible cut gives an error, not a p", () => {
    const sc = new CutScanner({ time: [1, 2, 3, 4], event: [1, 1, 1, 1] }, 0.1);
    expect("error" in permutationCorrectedP(sc, [1, 1, 1, 1], 10, new Rng(1))).toBe(true);
  });
});

describe("bootstrap-corrected C-index", () => {
  const c = fx.cases[1];
  const s = { time: c.time, event: c.event };
  const fitAll = (cols: number[]) => (idx: number[]) => {
    const r = coxFit({ time: idx.map((i) => s.time[i]), event: idx.map((i) => s.event[i]) }, idx.map((i) => cols.map((j) => c.x[i][j])));
    if (!r.ok) return null;
    return (i: number) => cols.reduce((acc, j, a) => acc + c.x[i][j] * r.fit.beta[a], 0);
  };
  it("is reproducible per seed, and optimism is positive", () => {
    const a = bootstrapCorrectedC(s, fitAll([0, 1, 2]), 100, new Rng(1)), b = bootstrapCorrectedC(s, fitAll([0, 1, 2]), 100, new Rng(1)), d = bootstrapCorrectedC(s, fitAll([0, 1, 2]), 100, new Rng(2));
    if ("error" in a || "error" in b || "error" in d) throw new Error("err");
    expect(a).toEqual(b);
    expect(d.corrected).not.toBe(a.corrected);
    expect(a.optimism).toBeGreaterThan(0);
    expect(a.corrected).toBeLessThan(a.apparent);
  });
  it("optimism is larger for a model that includes pure-noise covariates", () => {
    const r = new Rng(77);
    const extra = Array.from({ length: s.time.length }, () => Array.from({ length: 12 }, () => r.next()));
    const fit = (idx: number[]) => {
      const X = idx.map((i) => [c.x[i][0], ...extra[i]]);
      const m = coxFit({ time: idx.map((i) => s.time[i]), event: idx.map((i) => s.event[i]) }, X);
      if (!m.ok) return null;
      return (i: number) => [c.x[i][0], ...extra[i]].reduce((acc, v, a) => acc + v * m.fit.beta[a], 0);
    };
    const small = bootstrapCorrectedC(s, fitAll([0]), 100, new Rng(9)), big = bootstrapCorrectedC(s, fit, 100, new Rng(9));
    if ("error" in small || "error" in big) throw new Error("err");
    expect(big.optimism).toBeGreaterThan(small.optimism);
  });
  it("C-index of a risk equal for everybody is 0.5, and null when nothing is comparable", () => {
    expect(cIndex({ time: [1, 2, 3], event: [1, 1, 0] }, [1, 1, 1])).toBe(0.5);
    expect(cIndex({ time: [1, 2], event: [0, 0] }, [1, 2])).toBeNull();
  });
});

describe("input checks", () => {
  it("rejects non-positive time and event other than 0/1", () => {
    expect(checkSurv({ time: [0, 1], event: [1, 1] })).toMatch(/正/);
    expect(checkSurv({ time: [1, 1], event: [2, 1] })).toMatch(/0 か 1/);
    expect(checkSurv({ time: [1, 2], event: [1, 0] })).toBeNull();
  });
});
