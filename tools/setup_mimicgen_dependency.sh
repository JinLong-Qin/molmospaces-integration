#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"

# Pinned to the commits used for the public MolmoSpaces integration workline.
# Override these environment variables only if you intentionally want to test a
# newer upstream dependency revision.
MIMICGEN_URL="${MIMICGEN_URL:-https://github.com/NVlabs/mimicgen.git}"
MIMICGEN_COMMIT="${MIMICGEN_COMMIT:-72bd767c255545f462e7ccfb2731f2e5d4c1d9bb}"
ROBOMIMIC_URL="${ROBOMIMIC_URL:-https://github.com/ARISE-Initiative/robomimic.git}"
ROBOMIMIC_COMMIT="${ROBOMIMIC_COMMIT:-e10526b9a40c78b41f1e37e60041dc0ec0a5f60f}"

clone_or_checkout() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local dst="$4"

  if [ ! -d "$dst/.git" ]; then
    git clone "$url" "$dst"
  else
    git -C "$dst" remote set-url origin "$url"
    git -C "$dst" fetch --tags origin
  fi

  git -C "$dst" fetch origin "$commit"
  git -C "$dst" checkout --detach "$commit"
  printf '%s pinned at %s (%s)\n' "$name" "$(git -C "$dst" rev-parse --short HEAD)" "$dst"
}

clone_or_checkout "MimicGen" "$MIMICGEN_URL" "$MIMICGEN_COMMIT" "$VENDOR/mimicgen"
clone_or_checkout "robomimic" "$ROBOMIMIC_URL" "$ROBOMIMIC_COMMIT" "$VENDOR/robomimic"

cat <<EOF

Dependency checkouts are ready.
Add them to your active environment with:
  pip install -e "$VENDOR/robomimic"
  pip install -e "$VENDOR/mimicgen"

Or set PYTHONPATH if you do not install editable packages:
  export MIMICGEN_ROOT="$VENDOR/mimicgen"
  export ROBOMIMIC_ROOT="$VENDOR/robomimic"
  export PYTHONPATH="$ROOT:\$MIMICGEN_ROOT:\$ROBOMIMIC_ROOT:\${PYTHONPATH:-}"
EOF
