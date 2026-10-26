/** ウィンドウの中だけに効く CSS（接頭辞 rs-）。ライト・ダーク両方で読めるよう、色は変数で持つ。 */
export const CSS = `
.rs-root{--bg:#fff;--fg:#1f2d3d;--muted:#5a6672;--line:#d7dee6;--accent:#0b5cad;--warn:#8a5300;--warn-bg:#fff6e5;--err:#b00020;--err-bg:#fdecee;
  font:13px/1.55 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif;color:var(--fg);background:var(--bg);padding:12px 16px;box-sizing:border-box;height:100%;overflow:auto}
@media (prefers-color-scheme:dark){.rs-root{--bg:#16191d;--fg:#e3e8ec;--muted:#9aa6b0;--line:#2c353d;--accent:#7fb2d6;--warn:#e6c065;--warn-bg:#3a3018;--err:#ff8a98;--err-bg:#3d1c22}}
.rs-root h2{font-size:14px;margin:18px 0 6px;color:var(--accent)}
.rs-banner{background:var(--warn-bg);color:var(--warn);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:12px}
.rs-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0}
.rs-row>label{min-width:260px;color:var(--muted)}
.rs-root select,.rs-root input[type=number],.rs-root input[type=file]{max-width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:2px 6px}
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
`;
