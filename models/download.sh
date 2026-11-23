#!/usr/bin/env bash
# Download model weights (and optionally the raw CCTV images) from the GitHub Release. Run from the repo root:
#   bash models/download.sh          # weights only -> models/
#   bash models/download.sh --data   # also raw images (1.6 GB) -> data/cctv_images/
set -euo pipefail
TAG=models-v1
BASE=https://github.com/fifaak/dsde-proj-fifa/releases/download/$TAG
cd "$(dirname "$0")"
FILES="truck_clipvitb.pt yolo11s.pt yolo11s_nonms.mlpackage.zip clip_ViT-B-16.mlpackage.zip"
[ "${1:-}" = "--data" ] && FILES="$FILES cctv_images.zip"
curl -fsSL -o SHA256SUMS "$BASE/SHA256SUMS"
for f in $FILES; do
  [ -e "$f" ] || curl -fL -o "$f" "$BASE/$f"
  grep " $f\$" SHA256SUMS | shasum -a 256 -c -
done
for z in *.mlpackage.zip; do [ -e "$z" ] && unzip -qo "$z" && rm "$z"; done
if [ -e cctv_images.zip ]; then
  mkdir -p ../data && unzip -qo cctv_images.zip -d ../data && rm cctv_images.zip
  echo "images ready in $(cd ../data && pwd)/cctv_images"
fi
echo "models ready in $(pwd)"
