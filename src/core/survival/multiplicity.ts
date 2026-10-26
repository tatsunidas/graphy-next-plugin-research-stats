/** Benjamini–Hochberg 補正。p が null（計算できなかった）の項目は比較数に数えず、q も null。 */
export function benjaminiHochberg(p: (number | null)[]): (number | null)[] {
  const items = p.map((v, i) => ({ v, i })).filter((x): x is { v: number; i: number } => x.v !== null && Number.isFinite(x.v));
  const m = items.length;
  items.sort((a, b) => a.v - b.v);
  const q = new Array<number | null>(p.length).fill(null);
  let running = 1;
  for (let k = m - 1; k >= 0; k--) {
    running = Math.min(running, (items[k].v * m) / (k + 1));
    q[items[k].i] = Math.min(1, running);
  }
  return q;
}
