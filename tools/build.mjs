/**
 * ui.js を 1 ファイルにバンドルする。
 *
 * GRAPHY-Next のプラグインは「plugin.json ＋ ui.js」をフォルダ直下に置く形で配信される。
 * 分割チャンク・追加アセットは配信されないので、単一ファイル・外部依存なしにする。
 */
import { build } from "esbuild";
import { readFileSync, writeFileSync } from "node:fs";

const pkg = JSON.parse(readFileSync("plugin.json", "utf8"));

const result = await build({
  entryPoints: ["src/ui.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  platform: "browser",
  splitting: false,
  // 版をビルド時に埋め込む（実行時に plugin.json を読む手段が無い）。結果の記録に残す。
  define: { __PLUGIN_VERSION__: JSON.stringify(pkg.version) },
  minify: false, // 配布物を読んで検証できる状態を保つ
  legalComments: "inline",
  banner: {
    js:
      `/* ${pkg.name} v${pkg.version} — Research statistics for GRAPHY-Next\n` +
      ` * ${pkg.homepage ?? ""}\n` +
      ` * 研究用。診断・治療の判断には使わない。\n` +
      ` * このファイルは tools/build.mjs が src/ から生成します。直接編集しないこと。\n */`,
  },
  write: false,
});

const out = result.outputFiles[0];
writeFileSync("ui.js", out.text);
console.log(`ui.js を生成しました (${(out.contents.byteLength / 1024).toFixed(1)} KB)`);
