import { describe, expect, it } from "vitest";
import fx from "./fixtures/reliability.json";
import { fQuantile, tQuantile } from "../src/core/survival/dist";
import { icc, iccCategory } from "../src/core/reliability/icc";
import { blandAltman } from "../src/core/reliability/blandAltman";
import { Rng } from "../src/core/rng";

const close = (a: number, b: number, rel: number, abs = 0) => expect(Math.abs(a - b)).toBeLessThanOrEqual(abs + rel * Math.abs(b));

describe("F and t quantiles vs scipy", () => {
  it("matches", () => {
    for (const d of fx.dist.f) close(fQuantile(d.p, d.d1, d.d2), d.q, 1e-8);
    for (const d of fx.dist.t) close(tQuantile(d.p, d.df), d.q, 1e-8, 1e-12);
  });
});

describe("ICC(A,1) and ICC(A,k) vs pingouin (two-way, absolute agreement)", () => {
  it("point estimates and 95% CIs on five datasets (2–4 conditions, with and without a systematic offset)", () => {
    for (const c of fx.icc) {
      const r = icc(c.M);
      if ("error" in r) throw new Error(r.error);
      for (const [mine, ref] of [[r.single, c.ref.single], [r.average, c.ref.average]] as const) {
        close(mine.icc, ref.icc, 1e-8, 1e-10); close(mine.lower, ref.lower, 1e-6, 1e-8); close(mine.upper, ref.upper, 1e-6, 1e-8);
      }
    }
  });
  it("hand calculation: 3 subjects, 2 conditions", () => {
    // 行 (1,2), (2,3), (3,5): MSR = 3.5, MSC = 1.5·… を式どおりに手で出した値
    const r = icc([[1, 2], [2, 3], [3, 5]]);
    if ("error" in r) throw new Error(r.error);
    // 平均 = 16/6, 行平均 = 1.5, 2.5, 4 → SSR = 2·((1.5−8/3)²+(2.5−8/3)²+(4−8/3)²), SSC = 3·((2−8/3)²+(10/3−8/3)²)
    const gm = 16 / 6, rm = [1.5, 2.5, 4], cm = [2, 10 / 3];
    const ssr = 2 * rm.reduce((a, m) => a + (m - gm) ** 2, 0), ssc = 3 * cm.reduce((a, m) => a + (m - gm) ** 2, 0);
    const sst = [1, 2, 2, 3, 3, 5].reduce((a, v) => a + (v - gm) ** 2, 0);
    const msr = ssr / 2, msc = ssc / 1, mse = (sst - ssr - ssc) / 2;
    close(r.single.icc, (msr - mse) / (msr + mse + (2 * (msc - mse)) / 3), 1e-12);
    close(r.msr, msr, 1e-12); close(r.mse, mse, 1e-12);
  });
  it("two identical conditions give ICC 1 and 'excellent' (negative control for the detector below)", () => {
    const col = [3.2, 1.1, 7.5, 4.4, 9.9, 2.0];
    const r = icc(col.map((v) => [v, v]));
    if ("error" in r) throw new Error(r.error);
    close(r.single.icc, 1, 1e-12);
    close(r.single.lower, 1, 1e-9); close(r.single.upper, 1, 1e-9);
    expect(r.category).toBe("excellent");
  });
  it("independent random conditions give ICC near 0, never below 0 after truncation, and 'poor'", () => {
    const rng = new Rng(4);
    let sum = 0, neg = 0;
    const R = 60;
    for (let t = 0; t < R; t++) {
      const r = icc(Array.from({ length: 40 }, () => [rng.next(), rng.next()]));
      if ("error" in r) throw new Error(r.error);
      sum += r.single.icc;
      if (r.single.icc < 0) { neg++; expect(r.singleTruncated.icc).toBe(0); expect(r.singleTruncated.lower).toBe(0); }
      expect(r.category).toBe("poor");
    }
    expect(Math.abs(sum / R)).toBeLessThan(0.05);
    expect(neg).toBeGreaterThan(0); // 負になる場合が実際にあり、切る処理が通る
  });
  it("a constant offset keeps a perfect correlation but lowers the ICC (absolute agreement, not consistency)", () => {
    const base = Array.from({ length: 30 }, (_, i) => Math.sin(i) * 3 + i * 0.2);
    const same = icc(base.map((v) => [v, v])), shifted = icc(base.map((v) => [v, v + 5]));
    if ("error" in same || "error" in shifted) throw new Error("err");
    close(same.single.icc, 1, 1e-12);
    expect(shifted.single.icc).toBeLessThan(0.8); // 相関は 1 のまま
    const r2 = (() => { const m = base.reduce((a, b) => a + b, 0) / base.length; return base.reduce((a, v) => a + (v - m) * (v + 5 - (m + 5)), 0); })();
    expect(r2).toBeGreaterThan(0);
  });
  it("is unchanged by reordering the subjects or the conditions", () => {
    const c = fx.icc[1].M, a = icc(c), b = icc([...c].reverse().map((r) => [...r].reverse()));
    if ("error" in a || "error" in b) throw new Error("err");
    close(a.single.icc, b.single.icc, 1e-12); close(a.single.lower, b.single.lower, 1e-9); close(a.average.upper, b.average.upper, 1e-9);
  });
  it("systematic offset only (no error variance) gives a finite interval from the limit formula, with a note", () => {
    const r = icc([[1, 3], [2, 4], [5, 7], [9, 11]]);
    if ("error" in r) throw new Error(r.error);
    expect(r.mse).toBeCloseTo(0, 12);
    expect(r.note).toMatch(/極限/);
    expect(r.single.lower).toBeLessThanOrEqual(r.single.icc);
    expect(r.single.upper).toBeGreaterThanOrEqual(r.single.icc);
  });
  it("stops with a reason: all-equal values, one condition, too few subjects, missing (negative)", () => {
    expect("error" in icc([[2, 2], [2, 2], [2, 2]])).toBe(true);
    expect("error" in icc([[1], [2], [3]])).toBe(true);
    expect("error" in icc([[1, 2], [2, 3]])).toBe(true);
    expect("error" in icc([[1, 2], [NaN, 3], [4, 5]])).toBe(true);
  });
  it("category boundaries use the truncated lower bound: <0.5 poor, <0.75 moderate, <=0.9 good, >0.9 excellent", () => {
    expect([0.49, 0.5, 0.74, 0.75, 0.9, 0.91].map(iccCategory)).toEqual(["poor", "moderate", "moderate", "good", "good", "excellent"]);
  });
});

describe("Bland–Altman vs numpy/scipy", () => {
  it("bias, SD, limits of agreement and their confidence intervals", () => {
    for (const c of fx.ba) {
      const r = blandAltman(c.a, c.b);
      if ("error" in r) throw new Error(r.error);
      close(r.bias, c.bias, 1e-12); close(r.sd, c.sd, 1e-12);
      close(r.biasCi[0], c.bias_ci[0], 1e-9); close(r.biasCi[1], c.bias_ci[1], 1e-9);
      close(r.lowerLoa, c.lo, 1e-12); close(r.upperLoa, c.hi, 1e-12);
      close(r.lowerLoaCi[0], c.lo_ci[0], 1e-9); close(r.upperLoaCi[1], c.hi_ci[1], 1e-9);
      expect(r.points.length).toBe(c.a.length);
    }
  });
  it("hand calculation and stopping", () => {
    const r = blandAltman([1, 2, 3, 4], [2, 3, 4, 7]);
    if ("error" in r) throw new Error(r.error);
    close(r.bias, 1.5, 1e-12); // 差 1,1,1,3
    expect(r.points[3]).toEqual([5.5, 3]);
    expect("error" in blandAltman([1, 2], [1, 2])).toBe(true);
    expect("error" in blandAltman([1, 2, 3], [1, 2])).toBe(true);
  });
});
