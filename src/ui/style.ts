/** ウィンドウの中だけに効く CSS（接頭辞 rs-）。ライト・ダーク両方で読めるよう、色は変数で持つ。 */
export const CSS = `
.rs-root{--bg:#fff;--fg:#1f2d3d;--muted:#5a6672;--line:#d7dee6;--accent:#0b5cad;--warn:#8a5300;--warn-bg:#fff6e5;--err:#b00020;--err-bg:#fdecee;--c1:#0072B2;--c2:#D55E00;--c3:#009E73;--c4:#CC79A7;--c5:#6a6a6a;--cat-poor:#D55E00;--cat-mod:#E69F00;--cat-good:#56B4E9;--cat-exc:#009E73;--grid:#e6ebf0;
  font:13px/1.55 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif;color:var(--fg);background:var(--bg);padding:12px 16px;box-sizing:border-box;height:100%;overflow:auto}
@media (prefers-color-scheme:dark){.rs-root{--bg:#16191d;--fg:#e3e8ec;--muted:#9aa6b0;--line:#2c353d;--accent:#7fb2d6;--warn:#e6c065;--warn-bg:#3a3018;--err:#ff8a98;--err-bg:#3d1c22;--c1:#56B4E9;--c2:#E69F00;--c3:#66D9A8;--c4:#E8A0C8;--c5:#b0b0b0;--cat-poor:#F07A3A;--cat-mod:#E6B84A;--cat-good:#7CC6F0;--cat-exc:#33C9A0;--grid:#2c353d}}
.rs-root h2{font-size:14px;margin:18px 0 6px;color:var(--accent)}
.rs-banner{background:var(--warn-bg);color:var(--warn);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:12px}
.rs-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0}
.rs-row>label{min-width:260px;color:var(--muted)}
.rs-root select,.rs-root input[type=number],.rs-root input[type=text],.rs-root input[type=file]{max-width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:2px 6px}
.rs-root button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:6px;padding:4px 12px;cursor:pointer}
.rs-root button.rs-primary{background:var(--accent);border-color:var(--accent);color:#fff}
.rs-root button:disabled{opacity:.5;cursor:not-allowed}
.rs-note{color:var(--muted);font-size:12px}
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
