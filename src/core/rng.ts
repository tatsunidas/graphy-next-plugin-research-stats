/**
 * シード付き乱数（xoshiro128**）。シードは splitmix32 で 4 語の状態に広げる。
 * ブートストラップ・分割が同じシードで同じ結果になるように、Math.random は使わない。
 * 参照の系列は tools/gen_rng_fixture.py（別に書いた実装）が test/fixtures/rng.json に出す。
 */

const rotl = (x: number, k: number) => ((x << k) | (x >>> (32 - k))) >>> 0;

export class Rng {
  private s: [number, number, number, number];

  constructor(seed: number) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x9e3779b9) >>> 0;
      let t = a ^ (a >>> 16);
      t = Math.imul(t, 0x21f0aaad) >>> 0;
      t = (t ^ (t >>> 15)) >>> 0;
      t = Math.imul(t, 0x735a2d97) >>> 0;
      return (t ^ (t >>> 15)) >>> 0;
    };
    this.s = [next(), next(), next(), next()];
  }

  /** 次の 32 bit 符号なし整数。 */
  nextU32(): number {
    const s = this.s;
    const result = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] = (s[2] ^ s[0]) >>> 0;
    s[3] = (s[3] ^ s[1]) >>> 0;
    s[1] = (s[1] ^ s[2]) >>> 0;
    s[0] = (s[0] ^ s[3]) >>> 0;
    s[2] = (s[2] ^ t) >>> 0;
    s[3] = rotl(s[3], 11);
    return result;
  }

  /** [0, 1) の実数。 */
  next(): number {
    return this.nextU32() / 4294967296;
  }

  /** 0 以上 n 未満の整数（偏りが出ないよう棄却法）。 */
  int(n: number): number {
    if (!Number.isInteger(n) || n <= 0 || n > 4294967296) throw new RangeError(`n must be an integer in 1..2^32: ${n}`);
    const limit = 4294967296 - (4294967296 % n);
    let x = this.nextU32();
    while (x >= limit) x = this.nextU32();
    return x % n;
  }
}
