#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"

clone_or_update() {
  local url="$1"
  local dst="$2"
  if [ ! -d "$dst/.git" ]; then
    git clone "$url" "$dst"
  else
    git -C "$dst" pull --ff-only
  fi
}

clone_or_update https://github.com/NVlabs/mimicgen.git "$VENDOR/mimicgen"
clone_or_update https://github.com/ARISE-Initiative/robomimic.git "$VENDOR/robomimic"

printf 'MimicGen dependency is available at %s\n' "$VENDOR/mimicgen"
printf 'robomimic dependency is available at %s\n' "$VENDOR/robomimic"
