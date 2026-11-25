#!/usr/bin/env bash
# src/core を esbuild で 1 本の ESM にまとめ、Node で実行する（tsx が無い環境でも動く）。
set -euo pipefail
cd "$(dirname "$0")/../.."
BUNDLE="${RS_VERIFY_BUNDLE:-$(mktemp -d)/run_core.mjs}"
npx --no-install esbuild tools/verify_real/run_core.mts --bundle --platform=node --format=esm --target=node20 --log-level=warning --outfile="$BUNDLE"
exec node "$BUNDLE" "$@"
