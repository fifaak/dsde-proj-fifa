#!/usr/bin/env bash
# Download model weights from the GitHub Release into models/. Run from the repo root:
#   bash models/download.sh
set -euo pipefail
TAG=models-v1
BASE=https://github.com/fifaak/dsde-proj-fifa/releases/download/$TAG
cd "$(dirname "$0")"
for f in truck_clipvitb.pt yolo11s.pt yolo11s_nonms.mlpackage.zip clip_ViT-B-16.mlpackage.zip SHA256SUMS; do
  [ -e "$f" ] || curl -fL -o "$f" "$BASE/$f"
done
grep -v candidates_emb SHA256SUMS | shasum -a 256 -c -
for z in *.mlpackage.zip; do unzip -qo "$z" && rm "$z"; done
echo "models ready in $(pwd)"
