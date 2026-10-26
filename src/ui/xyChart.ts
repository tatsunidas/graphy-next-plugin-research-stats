/**
 * 汎用の折れ線・点の図（SVG）。ROC・較正・決定曲線に使う。系列は色（Okabe–Ito）と線種の両方で区別し、
 * 凡例は図の右下に置く。元の数値は各 CSV で書き出せる。
 */
const NS = "http://www.w3.org/2000/svg";
const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"];
const DASH = ["", "7 3", "2 2", "10 3 2 3", "1 3"];

function el(tag: string, attrs: Record<string, string | number> = {}, text?: string): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e as SVGElement;
}

export interface XySeries {
  label: string;
  /** [x, y] または [x, y, 下限, 上限]（誤差線）。 */
  points: number[][];
  kind: "line" | "step" | "points" | "scatter";
}

export interface XyOptions {
  xLabel: string;
  yLabel: string;
  xRange: [number, number];
  yRange: [number, number];
  /** 対角線（ROC・較正の理想）。 */
  diagonal?: boolean;
  testId: string;
  /** 目盛りの小数桁。 */
  digits?: number;
}

export function xyChart(series: XySeries[], o: XyOptions): SVGElement {
  const W = 420, H = 380, L = 54, R = 14, T = 12, B = 44, pw = W - L - R, ph = H - T - B;
  const [x0, x1] = o.xRange, [y0, y1] = o.yRange;
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * pw, Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * ph;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "data-testid": o.testId, style: "max-width:460px;height:auto" });
  svg.append(el("title", {}, o.yLabel));
  const d = o.digits ?? 1;
  for (let i = 0; i <= 5; i++) {
    const vy = y0 + ((y1 - y0) * i) / 5, vx = x0 + ((x1 - x0) * i) / 5;
    svg.append(el("line", { x1: L, x2: L + pw, y1: Y(vy), y2: Y(vy), class: "rs-grid" }), el("text", { x: L - 6, y: Y(vy) + 4, "text-anchor": "end", class: "rs-axis" }, vy.toFixed(d)));
    svg.append(el("line", { x1: X(vx), x2: X(vx), y1: T + ph, y2: T + ph + 4, class: "rs-axisline" }), el("text", { x: X(vx), y: T + ph + 17, "text-anchor": "middle", class: "rs-axis" }, vx.toFixed(1)));
  }
  svg.append(el("line", { x1: L, x2: L, y1: T, y2: T + ph, class: "rs-axisline" }), el("line", { x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "rs-axisline" }));
  svg.append(el("text", { x: L + pw / 2, y: H - 6, "text-anchor": "middle", class: "rs-axis" }, o.xLabel));
  svg.append(el("text", { transform: `translate(13 ${T + ph / 2}) rotate(-90)`, "text-anchor": "middle", class: "rs-axis" }, o.yLabel));
  if (o.diagonal) svg.append(el("line", { x1: X(x0), y1: Y(y0), x2: X(x1), y2: Y(y1), class: "rs-axisline", "stroke-dasharray": "3 4", opacity: 0.6 }));
  series.forEach((s, k) => {
    const color = COLORS[k % COLORS.length], dash = DASH[k % DASH.length];
    const clip = (v: number) => Math.min(y1, Math.max(y0, v));
    if (s.kind === "scatter") {
      for (const p of s.points) svg.append(el("circle", { cx: X(p[0]), cy: Y(clip(p[1])), r: 3, fill: color, opacity: 0.65 }));
    } else if (s.kind === "points") {
      for (const p of s.points) {
        if (p.length >= 4) svg.append(el("line", { x1: X(p[0]), x2: X(p[0]), y1: Y(clip(p[2])), y2: Y(clip(p[3])), stroke: color, "stroke-width": 1.2, opacity: 0.6 }));
        svg.append(el("circle", { cx: X(p[0]), cy: Y(clip(p[1])), r: 3.5, fill: color }));
      }
      if (s.points.length > 1) svg.append(el("path", { d: s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0])} ${Y(clip(p[1]))}`).join(" "), fill: "none", stroke: color, "stroke-width": 1, opacity: 0.5 }));
    } else {
      let dstr = "";
      s.points.forEach((p, i) => {
        const x = X(p[0]), y = Y(clip(p[1]));
        if (i === 0) dstr = `M${x} ${y}`;
        else dstr += s.kind === "step" ? ` H${x} V${y}` : ` L${x} ${y}`;
      });
      svg.append(el("path", { d: dstr, fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash }));
    }
  });
  series.forEach((s, k) => {
    const y = T + ph - 8 - (series.length - 1 - k) * 15, x = L + pw - 6;
    svg.append(el("line", { x1: x - 168, x2: x - 144, y1: y - 4, y2: y - 4, stroke: COLORS[k % COLORS.length], "stroke-width": 2, "stroke-dasharray": DASH[k % DASH.length] }), el("text", { x: x - 138, y, class: "rs-axis", "data-testid": `${o.testId}-legend` }, s.label));
  });
  return svg;
}
