import { describe, expect, it } from "vitest";
import fx from "./fixtures/enet.json";
import { fitStd, lassoAt, lassoCv, likOf, zRow } from "../src/core/model/lasso";
import { Rng } from "../src/core/rng";

const all = (n: number) => Array.from({ length: n }, (_, i) => i);

/** −(1/n)·l(β) + λ[α‖β‖₁ + (1−α)/2‖β‖²]（標準化した尺度の係数）。 */
function objective(c: { X: number[][]; time: number[]; event: number[] }, beta: number[], lambda: number, alpha: number): number {
  const rows = all(c.X.length);
  const st = fitStd(c.X, rows, all(c.X[0].length));
  if ("error" in st) throw new Error(st.error);
  const Z = c.X.map((r) => zRow(st, r));
  const lik = likOf({ kind: "survival", surv: { time: c.time, event: c.event } }, rows);
  const ll = lik.eval(Z.map((z) => z.reduce((a, v, j) => a + v * beta[j], 0))).ll;
  return -ll / rows.length + lambda * beta.reduce((a, b) => a + alpha * Math.abs(b) + ((1 - alpha) / 2) * b * b, 0);
}

describe("Elastic Net・Ridge の Cox（Breslow）vs scikit-survival Coxnet・自前の Ridge 参照（tools/gen_enet_fixture.py）", () => {
  for (const c of fx.cases) {
    const task = { kind: "survival" as const, surv: { time: c.time, event: c.event } };
    it(`seed ${c.seed}: coefficients and the zero pattern match for α = 1 (LASSO), 0.5, 0.2 (Coxnet) and 0 (Ridge)`, () => {
      let nz = 0;
      for (const f of c.fits) {
        const r = lassoAt(task, c.X, all(c.X.length), all(c.X[0].length), f.lambda, f.alpha);
        if ("error" in r) throw new Error(`α=${f.alpha} λ=${f.lambda}: ${r.error}`);
        // 目的関数は底が平らで、参照（Coxnet の収束許容 1e-13）との係数の差は 2e-6 まで出る。こちらの目的関数が参照より悪くないこと（≦ 参照 + 1e-12）を先に確かめ、係数は 5e-6 以内を求める
        expect(objective(c, r.coef.beta, f.lambda, f.alpha), `objective α=${f.alpha} λ=${f.lambda}`).toBeLessThanOrEqual(objective(c, f.beta, f.lambda, f.alpha) + 1e-12);
        r.coef.beta.forEach((b, j) => expect(Math.abs(b - f.beta[j]), `α=${f.alpha} λ=${f.lambda} #${j}`).toBeLessThan(5e-6));
        expect(r.coef.beta.map((b) => b !== 0), `α=${f.alpha} λ=${f.lambda}`).toEqual(f.beta.map((b) => b !== 0));
        nz += f.beta.filter((b) => b !== 0).length;
      }
      expect(nz).toBeGreaterThan(0); // 全部 0 の組だけで通っていない
    });
  }
  it("negative control: the penalty mix matters (α = 0.5 does not equal the LASSO fit at the same λ; Ridge keeps every coefficient non-zero)", () => {
    const c = fx.cases[0];
    const task = { kind: "survival" as const, surv: { time: c.time, event: c.event } };
    const at = (alpha: number) => (lassoAt(task, c.X, all(c.X.length), all(c.X[0].length), 0.05, alpha) as { coef: { beta: number[] } }).coef.beta;
    const lasso = at(1), enet = at(0.5), ridge = at(0);
    expect(Math.max(...lasso.map((b, j) => Math.abs(b - enet[j])))).toBeGreaterThan(1e-3);
    expect(ridge.every((b) => b !== 0)).toBe(true);
    expect(lasso.some((b) => b === 0)).toBe(true);
  });
  it("cross-validation runs for α = 0.5 and α = 0 (the λ path is built from α = 0.001 for Ridge) and is reproducible", () => {
    const c = fx.cases[1];
    const task = { kind: "survival" as const, surv: { time: c.time, event: c.event } };
    for (const alpha of [0.5, 0]) {
      const o = { nLambda: 30, ratio: 1e-2, folds: 5, rule: "min" as const, alpha };
      const a = lassoCv(task, c.X, all(c.X.length), all(c.X[0].length), o, new Rng(3));
      const b = lassoCv(task, c.X, all(c.X.length), all(c.X[0].length), o, new Rng(3));
      if ("error" in a || "error" in b) throw new Error(JSON.stringify(a));
      expect(a.lambda).toBe(b.lambda);
      if (alpha === 0) expect(a.nonzero.length).toBe(c.X[0].length);
    }
  });
});
