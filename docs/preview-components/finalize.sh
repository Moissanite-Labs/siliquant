#!/usr/bin/env bash
# Rebuild the Siliquant preview images from source. Offline once dependencies are installed.
#   ./finalize.sh              build + render + check + compress + previews into renders/
#   ./finalize.sh --install-og also copy renders/og-a16-v2.png to ../assets/og-a16-v2.png (the site's asset folder)
set -euo pipefail
cd "$(dirname "$0")"
node src/build.mjs
node render.mjs og-a16-v2 1200 630
node render.mjs x-post-1600x900 1600 900
node render.mjs x-post-1080x1350 1080 1350
for n in og-a16-v2 x-post-1600x900 x-post-1080x1350; do
  if command -v pngquant >/dev/null; then pngquant --force --quality=80-98 --speed 1 --output "renders/$n.png" "renders/$n.raw.png"
  else echo "pngquant not found: renders/$n.png is the uncompressed render" >&2; cp "renders/$n.raw.png" "renders/$n.png"; fi
done
node previews.mjs
og=$(wc -c < renders/og-a16-v2.png)
[ "$og" -lt 300000 ] || { echo "og-a16-v2.png is $og bytes; the link card limit is 300 KB" >&2; exit 1; }
if [ "${1:-}" = "--install-og" ]; then cp renders/og-a16-v2.png ../assets/og-a16-v2.png; echo "installed ../assets/og-a16-v2.png"; fi
ls -l renders/*.png
