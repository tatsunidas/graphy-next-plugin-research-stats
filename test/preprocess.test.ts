import { describe, expect, it } from "vitest";
import fx from "./fixtures/preprocess.json";
import { shapiroWilk } from "../src/core/stats/shapiro";
import { levene } from "../src/core/stats/levene";
import { pairedT, studentT, welchT } from "../src/core/stats/ttest";
import { wilcoxonSignedRank } from "../src/core/stats/wilcoxon";
import { mannWhitney } from "../src/core/survival/mannwhitney";
import { selectPaired, selectUnpaired } from "../src/core/stats/assumptions";
import { dropByPValue } from "../src/core/stats/dropByPValue";
import { fitStandardizer, needsStandardization } from "../src/core/stats/standardize";
import { normalQuantile } from "../src/core/survival/dist";

/** |a − b| <= abs + rel·|b|。null（計算できない）は null どうしでだけ一致。 */
function same(a: number | null, b: number | null, abs: number, rel: number, label: string) {
  if (b === null || a === null) { expect(a, label).toBe(b); return; }
  expect(Math.abs(a - b), `${label}: ${a} vs ${b}`).toBeLessThanOrEqual(abs + rel * Math.abs(b));
}
const ok = <T extends object>(r: T | { error: string }): T => {
  if ("error" in r) throw new Error(r.error);
  return r;
};

// 許容の根拠:
//  - Shapiro–Wilk: scipy の係数計算は正規分布の分位点・上側確率を約 1e-7 の精度の近似で求める実装と思われ（未確認）、
//    こちらは分位点を二分法で厳密に求めるため、W は 1e-8、p は絶対 1e-7 + 相対 1e-6 の差を許す。実測の最大差は W 1.7e-9、p 2.2e-8。
//  - それ以外は不完全ベータ・ガンマ関数の連分数（収束判定 1e-15）と和の丸めだけが差の源で、実測は相対 1e-11 以下。
//    統計量は 1e-10、p は絶対 1e-10 + 相対 1e-9 を許す。
describe("Shapiro–Wilk vs scipy.stats.shapiro", () => {
  it("matches on every case (n = 3 … 5000, several distributions)", () => {
    expect(fx.shapiro.length).toBeGreaterThanOrEqual(30);
    for (const c of fx.shapiro) {
      const r = ok(shapiroWilk(c.x));
      const label = `${c.kind} n=${c.x.length}`;
      same(r.w, c.w, 1e-8, 0, `${label} W`);
      same(r.p, c.p, 1e-7, 1e-6, `${label} p`);
    }
  });
  it("covers every branch (n = 3, 4–11, ≥ 12) and the constant case", () => {
    const ns = new Set(fx.shapiro.map((c) => c.x.length));
    for (const n of [3, 4, 11, 12, 5000]) expect(ns.has(n)).toBe(true);
    const r = ok(shapiroWilk([2, 2, 2, 2]));
    expect(r.p).toBeNull();
    expect(r.w).toBeNull();
  });
  it("returns errors outside 3 ≤ n ≤ 5000", () => {
    expect("error" in shapiroWilk([1, 2])).toBe(true);
    expect("error" in shapiroWilk(Array.from({ length: 5001 }, (_, i) => i))).toBe(true);
    expect("error" in shapiroWilk([1, 2, NaN])).toBe(true);
  });
});

describe("Levene / Brown–Forsythe vs scipy.stats.levene", () => {
  it("matches for median and mean centres, 2 and 3 groups", () => {
    expect(fx.levene.length).toBeGreaterThanOrEqual(30);
    for (const c of fx.levene) {
      const r = ok(levene(c.groups, c.center as "median" | "mean"));
      same(r.f, c.f, 1e-10, 1e-9, `${c.tag} ${c.center} F`);
      same(r.p, c.p, 1e-10, 1e-9, `${c.tag} ${c.center} p`);
    }
    expect(new Set(fx.levene.map((c) => c.center))).toEqual(new Set(["median", "mean"]));
  });
  it("errors for fewer than 2 groups", () => {
    expect("error" in levene([[1, 2, 3]])).toBe(true);
  });
});

describe("t tests vs scipy", () => {
  it("Student / Welch (two-sample)", () => {
    expect(fx.ttestInd.length).toBeGreaterThanOrEqual(30);
    for (const c of fx.ttestInd) {
      const r = ok((c.equalVar ? studentT : welchT)(c.x, c.y));
      const label = `${c.tag} ${c.equalVar ? "student" : "welch"}`;
      same(r.t, c.t, 1e-10, 1e-9, `${label} t`);
      same(r.df, c.df, 1e-9, 1e-10, `${label} df`);
      same(r.p, c.p, 1e-10, 1e-9, `${label} p`);
    }
  });
  it("Welch df differs from Student df when variances differ (guards against using the wrong formula)", () => {
    const c = fx.ttestInd.find((d) => !d.equalVar && d.tag === "normal-200-200-3")!;
    const w = ok(welchT(c.x, c.y)), s = ok(studentT(c.x, c.y));
    expect(s.df).toBe(398);
    expect(w.df!).toBeLessThan(380);
  });
  it("paired", () => {
    expect(fx.ttestRel.length).toBeGreaterThanOrEqual(30);
    for (const c of fx.ttestRel) {
      const r = ok(pairedT(c.x, c.y));
      same(r.t, c.t, 1e-10, 1e-9, `${c.tag} t`);
      same(r.df, c.df, 0, 0, `${c.tag} df`);
      same(r.p, c.p, 1e-10, 1e-9, `${c.tag} p`);
    }
  });
});

describe("Wilcoxon signed-rank vs scipy.stats.wilcoxon (auto, wilcox, no correction)", () => {
  it("matches the statistic and p, and uses every method", () => {
    expect(fx.wilcoxon.length).toBeGreaterThanOrEqual(30);
    const methods = new Set<string>();
    for (const c of fx.wilcoxon) {
      const r = ok(wilcoxonSignedRank(c.x, c.y));
      same(r.statistic, c.stat, 1e-9, 0, `${c.tag} stat`);
      same(r.p, c.p, 1e-10, 1e-9, `${c.tag} p`);
      if (r.method) methods.add(r.method);
    }
    expect(methods).toEqual(new Set(["exact", "permutation", "approx"]));
  });
  it("reports the zero differences it dropped, and cannot compute when all differences are zero", () => {
    const r = ok(wilcoxonSignedRank([1, 2, 3, 4, 5, 6], [1, 2, 3.5, 4.2, 4, 6]));
    expect(r.zerosDropped).toBe(3);
    expect(r.n).toBe(3);
    const z = ok(wilcoxonSignedRank([1, 2, 3], [1, 2, 3]));
    expect(z.p).toBeNull();
  });
});

describe("Mann–Whitney (existing implementation) vs scipy.stats.mannwhitneyu(method=asymptotic)", () => {
  it("matches including ties", () => {
    expect(fx.mannwhitney.length).toBeGreaterThanOrEqual(30);
    for (const c of fx.mannwhitney) {
      const r = ok(mannWhitney(c.x, c.y));
      same(r.u1, c.u1, 1e-9, 0, `${c.tag} U1`);
      same(r.p, c.p, 1e-10, 1e-9, `${c.tag} p`);
    }
  });
});

describe("test selection", () => {
  it("matches the rule written with scipy functions (test and p)", () => {
    expect(fx.select.unpaired.length + fx.select.paired.length).toBeGreaterThanOrEqual(30);
    const seen = new Set<string>();
    for (const c of fx.select.unpaired) {
      const r = ok(selectUnpaired(c.x, c.y, c.alpha));
      expect(r.test, `${c.tag} α=${c.alpha}`).toBe(c.test);
      same(r.p, c.p, 1e-10, 1e-8, `${c.tag} p`);
      seen.add(r.test);
    }
    for (const c of fx.select.paired) {
      const r = ok(selectPaired(c.x, c.y, c.alpha));
      expect(r.test, `${c.tag} α=${c.alpha}`).toBe(c.test);
      same(r.p, c.p, 1e-10, 1e-8, `${c.tag} p`);
      seen.add(r.test);
    }
    expect(seen).toEqual(new Set(["student", "welch", "mannwhitney", "paired-t", "wilcoxon"]));
  });

  // 正規の分位点を等間隔に並べた「見かけが正規」の標本と、指数分布の分位点の「正規でない」標本を意図して作る
  const normalLike = (n: number, mean: number, sd: number) => Array.from({ length: n }, (_, i) => mean + sd * normalQuantile((i + 0.5) / n));
  const skewed = (n: number, scale: number) => Array.from({ length: n }, (_, i) => -scale * Math.log(1 - (i + 0.5) / n));

  it("normal and equal variance -> Student", () => {
    const r = ok(selectUnpaired(normalLike(40, 0, 1), normalLike(40, 0.8, 1), 0.05));
    expect(r.test).toBe("student");
    expect(r.levene!.p!).toBeGreaterThanOrEqual(0.05);
    expect(r.why.ja).toContain("Student");
    expect(r.why.en).toContain("Student");
  });
  it("normal and unequal variance -> Welch", () => {
    const r = ok(selectUnpaired(normalLike(40, 0, 1), normalLike(40, 0.8, 5), 0.05));
    expect(r.test).toBe("welch");
    expect(r.levene!.p!).toBeLessThan(0.05);
  });
  it("not normal -> Mann–Whitney (Levene is not consulted)", () => {
    const r = ok(selectUnpaired(skewed(40, 1), normalLike(40, 1, 1), 0.05));
    expect(r.test).toBe("mannwhitney");
    expect(r.levene).toBeNull();
  });
  it("a group that cannot be tested for normality (n < 3, constant) -> Mann–Whitney", () => {
    expect(ok(selectUnpaired([1, 2], normalLike(30, 0, 1), 0.05)).test).toBe("mannwhitney");
    expect(ok(selectUnpaired([3, 3, 3, 3], normalLike(30, 0, 1), 0.05)).test).toBe("mannwhitney");
  });
  it("paired: normal differences -> paired t, skewed differences -> Wilcoxon", () => {
    const base = normalLike(30, 10, 2);
    const a = ok(selectPaired(base.map((v, i) => v + normalLike(30, 0.5, 1)[(i * 7) % 30]), base, 0.05));
    expect(a.test).toBe("paired-t");
    const b = ok(selectPaired(base.map((v, i) => v + skewed(30, 1)[(i * 7) % 30]), base, 0.05));
    expect(b.test).toBe("wilcoxon");
    expect(b.wilcoxon).toBeDefined();
  });
  it("the same data flips the choice when α changes, and α has no default", () => {
    const x = normalLike(25, 0, 1).map((v, i) => (i === 3 ? v + 6 : v));
    const y = normalLike(25, 0.5, 1);
    const px = ok(shapiroWilk(x)).p!;
    expect(px).toBeGreaterThan(1e-8);
    expect(px).toBeLessThan(0.1);
    const lo = ok(selectUnpaired(x, y, px / 2)), hi = ok(selectUnpaired(x, y, Math.min(0.99, px * 2)));
    expect(lo.test).not.toBe("mannwhitney");
    expect(hi.test).toBe("mannwhitney");
    expect("error" in selectUnpaired(x, y, 0)).toBe(true);
    expect("error" in selectUnpaired(x, y, 1)).toBe(true);
  });
});

describe("dropByPValue", () => {
  const n = 30;
  const q = (i: number) => normalQuantile((i + 0.5) / n);
  // 行 0..29 = 群 0、行 30..59 = 群 1
  const rows0 = Array.from({ length: n }, (_, i) => i), rows1 = Array.from({ length: n }, (_, i) => n + i);
  const X: (number | null)[][] = [];
  for (let g = 0; g < 2; g++) for (let i = 0; i < n; i++) X.push([q(i) + 3 * g, q((i * 7) % n), 4, i < 5 && g === 0 ? null : q(i) + 3 * g]);
  const names = ["shifted", "same", "constant", "shifted-with-missing"];

  it("unpaired: keeps the group-different column, drops the same-distribution and the constant column", () => {
    const r = ok(dropByPValue(X, names, { kind: "unpaired", rows0, rows1 }, 0.05, 0.05));
    expect(r.keep).toEqual([0, 3]);
    expect(r.records[1].p!).toBeGreaterThanOrEqual(0.05);
    expect(r.records[1].dropReason!.ja).toContain("しきい値");
    expect(r.records[2].p).toBeNull();
    expect(r.records[2].dropReason!.ja).toContain("計算できない");
    expect(r.records[3].n).toEqual([25, 30]);
    expect(r.records[0].selection!.test).toBe("student");
  });
  it("unpaired: threshold equal to p drops (p ≥ threshold), and a threshold above p keeps", () => {
    const p = ok(dropByPValue(X, names, { kind: "unpaired", rows0, rows1 }, 0.05, 1)).records[0].p!;
    expect(ok(dropByPValue(X, names, { kind: "unpaired", rows0, rows1 }, 0.05, p)).keep).not.toContain(0);
    expect(ok(dropByPValue(X, names, { kind: "unpaired", rows0, rows1 }, 0.05, p * 1.01)).keep).toContain(0);
  });
  it("paired: keeps the column that changed, drops the one that did not, drops a column with no difference at all", () => {
    // 行 i と行 n+i が同じ対象の前後
    const pairs = rows0.map((i) => [i, n + i] as [number, number]);
    const P: (number | null)[][] = [];
    for (let t = 0; t < 2; t++) for (let i = 0; i < n; i++) P.push([q(i) + 1.5 * t + 0.05 * q((i * 11) % n), q(i) + (t === 0 ? 1 : -1) * 0.3 * q((i * 7) % n), 5]);
    const r = ok(dropByPValue(P, ["changed", "unchanged", "same"], { kind: "paired", pairs }, 0.05, 0.05));
    expect(r.keep).toEqual([0]);
    expect(r.records[2].p).toBeNull();
    expect(r.records[1].selection!.test).toBe("paired-t");
  });
  it("reads only the rows it is given", () => {
    const Y = X.map((r) => [...r]);
    const r0 = ok(dropByPValue(Y, names, { kind: "unpaired", rows0: rows0.slice(0, 20), rows1: rows1.slice(0, 20) }, 0.05, 0.05));
    for (let i of [...rows0.slice(20), ...rows1.slice(20)]) Y[i] = Y[i].map(() => 1e6);
    const r1 = ok(dropByPValue(Y, names, { kind: "unpaired", rows0: rows0.slice(0, 20), rows1: rows1.slice(0, 20) }, 0.05, 0.05));
    expect(r1.records.map((r) => r.p)).toEqual(r0.records.map((r) => r.p));
  });
  it("rejects a threshold outside (0, 1]", () => {
    expect("error" in dropByPValue(X, names, { kind: "unpaired", rows0, rows1 }, 0.05, 0)).toBe(true);
  });
});

describe("standardizer", () => {
  const X: (number | null)[][] = [[1, 5, 2], [2, 5, null], [3, 5, 4], [4, 5, 8], [100, 5, -50]];
  it("uses sample SD (n − 1) of the given rows only, leaves missing as null, marks SD 0 columns unusable", () => {
    const s = fitStandardizer(X, [0, 1, 2, 3]);
    expect(s.means[0]).toBeCloseTo(2.5, 12);
    expect(s.sds[0]).toBeCloseTo(Math.sqrt(5 / 3), 12);
    expect(s.usable).toEqual([true, false, true]);
    const z = s.apply(X, [0, 1, 4]);
    expect(z[0][0]).toBeCloseTo((1 - 2.5) / Math.sqrt(5 / 3), 12);
    expect(z[0][1]).toBeNull();
    expect(z[1][2]).toBeNull();
    expect(Number.isFinite(z[2][0])).toBe(true); // 推定に使っていない行にも適用できる
  });
  it("does not read rows outside the fitting rows: rewriting them changes nothing, rewriting fitting rows does", () => {
    const fitRows = [0, 1, 2, 3];
    const base = fitStandardizer(X, fitRows);
    const Y = X.map((r) => [...r]);
    Y[4] = [-1e9, 123, 1e9];
    const other = fitStandardizer(Y, fitRows);
    expect(other.means).toEqual(base.means);
    expect(other.sds).toEqual(base.sds);
    const Z = X.map((r) => [...r]);
    Z[3][0] = 40;
    expect(fitStandardizer(Z, fitRows).means[0]).not.toBe(base.means[0]);
    expect(fitStandardizer(X, [0, 1, 2, 3, 4]).means[0]).not.toBe(base.means[0]);
  });
  it("model table: scale-sensitive models need it, tree models do not", () => {
    for (const m of ["lasso", "svm", "logistic", "knn"] as const) expect(needsStandardization(m)).toBe(true);
    for (const m of ["tree", "rf", "gbm"] as const) expect(needsStandardization(m)).toBe(false);
  });
});
