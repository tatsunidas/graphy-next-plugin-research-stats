/**
 * Kaplan–Meier 曲線（SVG）。群は色（色覚に配慮した Okabe–Ito）と線種の両方で区別し、
 * 信頼区間は薄い帯、打ち切りは「+」、図の下にリスク表を置く。数値の元は kmPointsCsv で書き出せる。
 */
import { atRiskAt } from "../core/survival/km";
import type { KmGroup } from "../core/r1";

const NS = "http://www.w3.org/2000/svg";
const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)"];
const DASH = ["", "7 3", "2 2"];

function el(tag: string, attrs: Record<string, string | number> = {}, text?: string): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e as SVGElement;
}

function niceStep(max: number, target = 6): number {
  const raw = max / target, pow = 10 ** Math.floor(Math.log10(raw)), f = raw / pow;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * pow;
}

export interface KmLabels {
  time: string;
  survival: string;
  atRisk: string;
  ci: string;
}

export function kmChart(groups: KmGroup[], labels: KmLabels): SVGElement {
  const W = 680, rowH = 16, H = 330 + groups.length * rowH + 26;
  const L = 52, R = 18, T = 12, plotW = W - L - R, plotH = 270;
  const maxT = Math.max(...groups.map((g) => g.km.points[g.km.points.length - 1]?.time ?? 0));
  const step = niceStep(maxT), xmax = Math.ceil(maxT / step) * step;
  const X = (t: number) => L + (t / xmax) * plotW, Y = (s: number) => T + (1 - s) * plotH;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "data-testid": "rs-km-svg", style: "max-width:720px;height:auto" });
  svg.append(el("title", {}, labels.survival));
  // 目盛りと格子
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    svg.append(el("line", { x1: L, x2: L + plotW, y1: Y(v), y2: Y(v), class: "rs-grid" }), el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "rs-axis" }, v.toFixed(1)));
  }
  const ticks: number[] = [];
  for (let t = 0; t <= xmax + 1e-9; t += step) ticks.push(Number(t.toPrecision(10)));
  for (const t of ticks) {
    svg.append(el("line", { x1: X(t), x2: X(t), y1: T + plotH, y2: T + plotH + 4, class: "rs-axisline" }), el("text", { x: X(t), y: T + plotH + 17, "text-anchor": "middle", class: "rs-axis" }, String(t)));
  }
  svg.append(el("line", { x1: L, x2: L, y1: T, y2: T + plotH, class: "rs-axisline" }), el("line", { x1: L, x2: L + plotW, y1: T + plotH, y2: T + plotH, class: "rs-axisline" }));
  svg.append(el("text", { x: L + plotW / 2, y: T + plotH + 34, "text-anchor": "middle", class: "rs-axis" }, labels.time));
  svg.append(el("text", { transform: `translate(14 ${T + plotH / 2}) rotate(-90)`, "text-anchor": "middle", class: "rs-axis" }, labels.survival));

  groups.forEach((g, k) => {
    const color = COLORS[k % COLORS.length];
    const pts = g.km.points;
    // 階段の座標列（時間 0 から最後の観察まで）
    const stepPath = (pick: (p: (typeof pts)[number]) => number) => {
      let d = `M${X(0)} ${Y(1)}`;
      let prev = 1;
      for (const p of pts) {
        d += ` H${X(p.time)} V${Y(pick(p))}`;
        prev = pick(p);
      }
      void prev;
      return d;
    };
    // 信頼区間の帯: 上側を進み、下側を戻る
    let band = `M${X(0)} ${Y(1)}`;
    for (const p of pts) band += ` H${X(p.time)} V${Y(p.upper)}`;
    for (let i = pts.length - 1; i >= 0; i--) {
      band += ` V${Y(pts[i].lower)} H${X(i > 0 ? pts[i - 1].time : 0)}`;
    }
    band += " Z";
    svg.append(el("path", { d: band, fill: color, opacity: 0.12, stroke: "none" }));
    svg.append(el("path", { d: stepPath((p) => p.surv), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": DASH[k % DASH.length] }));
    for (const p of pts) if (p.censored > 0) {
      const x = X(p.time), y = Y(p.surv);
      svg.append(el("path", { d: `M${x - 3} ${y} H${x + 3} M${x} ${y - 3} V${y + 3}`, stroke: color, "stroke-width": 1.2, fill: "none" }));
    }
  });

  // 凡例（群のしきい値・人数・イベント数）
  groups.forEach((g, k) => {
    const y = T + 14 + k * 16, x = L + plotW - 6;
    svg.append(
      el("line", { x1: x - 150, x2: x - 126, y1: y - 4, y2: y - 4, stroke: COLORS[k % COLORS.length], "stroke-width": 2, "stroke-dasharray": DASH[k % DASH.length] }),
      el("text", { x: x - 120, y, class: "rs-axis", "data-testid": "rs-km-legend" }, `${g.label}  (n=${g.n}, ${g.events})`),
    );
  });
  svg.append(el("text", { x: L + plotW - 6, y: T + 14 + groups.length * 16 + 4, "text-anchor": "end", class: "rs-axis rs-faint" }, labels.ci));

  // リスク表
  const y0 = T + plotH + 54;
  svg.append(el("text", { x: L, y: y0 - 6, class: "rs-axis rs-bold" }, labels.atRisk));
  groups.forEach((g, k) => {
    const y = y0 + 8 + k * rowH;
    svg.append(el("line", { x1: 6, x2: 20, y1: y - 4, y2: y - 4, stroke: COLORS[k % COLORS.length], "stroke-width": 2, "stroke-dasharray": DASH[k % DASH.length] }));
    for (const t of ticks) svg.append(el("text", { x: X(t), y, "text-anchor": "middle", class: "rs-axis", "data-testid": "rs-km-risk" }, String(atRiskAt(g.surv, t))));
  });
  return svg;
}
