/** ウィンドウの中だけに効く CSS（接頭辞 rs-）。ライト・ダーク両方で読めるよう、色は変数で持つ。 */
export const CSS = `
.rs-root{--bg:#fff;--fg:#1f2d3d;--muted:#5a6672;--line:#d7dee6;--accent:#0b5cad;--warn:#8a5300;--warn-bg:#fff6e5;--err:#b00020;--err-bg:#fdecee;--c1:#0072B2;--c2:#D55E00;--c3:#009E73;--c4:#CC79A7;--c5:#6a6a6a;--cat-poor:#D55E00;--cat-mod:#E69F00;--cat-good:#56B4E9;--cat-exc:#009E73;--grid:#e6ebf0;
  font:13px/1.55 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif;color:var(--fg);background:var(--bg);padding:12px 16px;box-sizing:border-box;height:100%;overflow:auto}
@media (prefers-color-scheme:dark){.rs-root{--bg:#16191d;--fg:#e3e8ec;--muted:#9aa6b0;--line:#2c353d;--accent:#7fb2d6;--warn:#e6c065;--warn-bg:#3a3018;--err:#ff8a98;--err-bg:#3d1c22;--c1:#56B4E9;--c2:#E69F00;--c3:#66D9A8;--c4:#E8A0C8;--c5:#b0b0b0;--cat-poor:#F07A3A;--cat-mod:#E6B84A;--cat-good:#7CC6F0;--cat-exc:#33C9A0;--grid:#2c353d}}
.rs-root h2{display:flex;align-items:center;gap:8px;font-size:15px;margin:0 0 8px;color:var(--accent)}
.rs-root h3{font-size:12px;margin:12px 0 2px;color:var(--muted);font-weight:600;letter-spacing:.02em}
.rs-card{border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}
.rs-card.rs-off{background:color-mix(in srgb,var(--line) 25%,transparent)}
.rs-num{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:var(--accent);color:var(--bg);font-size:12px}
.rs-reason{color:var(--muted);font-size:12px;margin:2px 0 4px}
.rs-pass{color:var(--accent);font-weight:600;margin:8px 0 4px}
.rs-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
.rs-field{display:flex;flex-direction:column;gap:3px;min-width:0}
.rs-files{list-style:none;margin:0;padding:0}
.rs-files li{display:flex;align-items:center;gap:6px;padding:1px 0}
.rs-fname{font-weight:600;overflow-wrap:anywhere}
.rs-root button.rs-x{padding:0 7px;line-height:1.4;border-radius:50%}
.rs-qc{font-size:12px;border-radius:4px;padding:1px 6px}
.rs-chip-bad{color:var(--warn)}
.rs-banner{background:var(--warn-bg);color:var(--warn);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:12px}
.rs-row{display:grid;grid-template-columns:minmax(180px,260px) minmax(0,1fr);gap:4px 12px;align-items:center;margin:5px 0}
.rs-row>label{color:var(--muted)}
@media (max-width:620px){.rs-row{grid-template-columns:minmax(0,1fr)}}
.rs-root select,.rs-root input[type=number],.rs-root input[type=text],.rs-root input[type=file]{max-width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:2px 6px}
.rs-root button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:6px;padding:4px 12px;cursor:pointer}
.rs-root button.rs-primary{background:var(--accent);border-color:var(--accent);color:#fff}
.rs-root button:disabled{opacity:.5;cursor:not-allowed}
.rs-note{color:var(--muted);font-size:12px}
.rs-info{display:inline-flex;align-items:center;justify-content:center;margin-left:4px;width:16px;height:16px;border:1px solid var(--line);border-radius:50%;font-size:11px;color:var(--accent);cursor:help;user-select:none}
.rs-err{background:var(--err-bg);color:var(--err);border-radius:6px;padding:6px 10px;margin:6px 0}
.rs-warn{background:var(--warn-bg);color:var(--warn);border-radius:6px;padding:6px 10px;margin:6px 0}
.rs-root ul{margin:4px 0;padding-left:20px}
.rs-root details{margin:6px 0}
.rs-chips{display:flex;flex-wrap:wrap;gap:4px 12px}
.rs-big{font-size:15px;font-weight:600}
.rs-table{border-collapse:collapse;font-size:12px;width:100%}
.rs-table th,.rs-table td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;white-space:nowrap}
.rs-table tr.rs-sel{background:color-mix(in srgb,var(--accent) 14%,transparent)}
.rs-table tbody tr{cursor:pointer}
.rs-tablewrap{overflow-x:auto;max-height:340px;overflow-y:auto;border:1px solid var(--line);border-radius:6px}
.rs-progress{height:8px;border-radius:4px;background:var(--line);overflow:hidden;min-width:240px}
.rs-progress>div{height:100%;background:var(--accent)}
.rs-bar{height:8px;background:var(--c1);border-radius:2px;min-width:1px}
.rs-grid{stroke:var(--grid);stroke-width:1}
.rs-axisline{stroke:var(--muted);stroke-width:1}
.rs-axis{fill:var(--fg);font-size:11px}
.rs-faint{fill:var(--muted)}
.rs-bold{font-weight:600}
.rs-charts{display:flex;flex-wrap:wrap;gap:12px}
.rs-charts>div{flex:1 1 380px;min-width:0}
.rs-cm{border-collapse:collapse;text-align:center}
.rs-cm td,.rs-cm th{border:1px solid var(--line);padding:4px 12px}
.rs-tree ul{list-style:none;padding-left:18px;border-left:1px solid var(--line)}
.rs-tree>ul{border-left:none;padding-left:0}
.rs-tree li{margin:3px 0}
.rs-tag{display:inline-block;padding:0 6px;border-radius:4px;border:1px solid var(--line);font-size:11px}
`;
