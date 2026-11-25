import { sourceOf } from "../../src/core/delta";
import { Rng } from "../../src/core/rng";

/**
 * 同じ患者に複数行がある合成データ。患者 i は (i % 3) + 1 行。f_a・f_b は患者の潜在値 + 行ごとの揺らぎ（結果に効く）、
 * f_n0・f_n1 は無関係。結果は 2 値（outcome）と生存（time・event）。行の ID は RowID、患者は Patient。
 */
export function multiRow(nPatients: number, seed: number, o: { shift?: number } = {}) {
  const r = new Rng(seed);
  const norm = () => { let s = 0; for (let k = 0; k < 6; k++) s += r.next(); return s - 3; };
  const shift = o.shift ?? 1.4;
  const feat = ["RowID,Patient,f_a,f_b,f_n0,f_n1"], clin = ["Patient,outcome,time,event,age"];
  for (let i = 0; i < nPatients; i++) {
    const g = r.next() < 0.45 ? 1 : 0, a = norm() + shift * g, b = norm() + 0.7 * g;
    const rows = (i % 3) + 1;
    for (let k = 0; k < rows; k++) feat.push([`P${i}_L${k}`, `P${i}`, (a + 0.4 * norm()).toFixed(4), (b + 0.4 * norm()).toFixed(4), norm().toFixed(4), norm().toFixed(4)].join(","));
    const time = (-Math.log(1 - r.next()) / Math.exp(0.9 * a)).toFixed(4);
    clin.push(`P${i},${g ? "resp" : "no"},${Number(time) + 0.001},${r.next() < 0.8 ? 1 : 0},${40 + (i % 30)}`);
  }
  return { features: [sourceOf("f.csv", feat.join("\n") + "\n", "a".repeat(64))], clinical: sourceOf("c.csv", clin.join("\n") + "\n", null) };
}

export const roles = { featureIdColumn: "RowID", patientColumn: "Patient", clinicalIdColumn: "Patient", outcomeColumn: "outcome", timeColumn: "time", eventColumn: "event" };
export const opts = { seed: 1, pluginVersion: "t", now: () => "2026-01-01T00:00:00Z" };
