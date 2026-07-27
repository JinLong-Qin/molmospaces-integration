#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"

if [ ! -d "$VENDOR/mimicgen/.git" ]; then
  git clone https://github.com/NVlabs/mimicgen.git "$VENDOR/mimicgen"
else
  git -C "$VENDOR/mimicgen" pull --ff-only
fi

printf 'MimicGen dependency is available at %s\n' "$VENDOR/mimicgen"
