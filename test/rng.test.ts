import { describe, expect, it } from "vitest";
import fixture from "./fixtures/rng.json";
import { Rng } from "../src/core/rng";

describe("Rng (xoshiro128**)", () => {
  it("matches the reference sequence written separately in Python", () => {
    for (const [seed, want] of Object.entries(fixture as Record<string, number[]>)) {
      const r = new Rng(Number(seed));
      expect(Array.from({ length: want.length }, () => r.nextU32())).toEqual(want);
    }
  });
  it("repeats for the same seed and differs for another seed", () => {
    const a = new Rng(7), b = new Rng(7), c = new Rng(8);
    const xs = Array.from({ length: 5 }, () => a.next());
    expect(Array.from({ length: 5 }, () => b.next())).toEqual(xs);
    expect(Array.from({ length: 5 }, () => c.next())).not.toEqual(xs);
  });
  it("gives numbers in [0, 1) and integers in range", () => {
    const r = new Rng(1);
    for (let i = 0; i < 1000; i++) {
      const x = r.next();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      const k = r.int(7);
      expect(Number.isInteger(k) && k >= 0 && k < 7).toBe(true);
    }
  });
  it("covers every integer in a small range about evenly", () => {
    const r = new Rng(99);
    const counts = new Array(4).fill(0);
    for (let i = 0; i < 40000; i++) counts[r.int(4)]++;
    for (const c of counts) expect(Math.abs(c - 10000)).toBeLessThan(500); // 5 SD 以内（SD ≈ 87）
  });
  it("refuses a bad range (negative)", () => {
    const r = new Rng(1);
    expect(() => r.int(0)).toThrow();
    expect(() => r.int(1.5)).toThrow();
  });
});
