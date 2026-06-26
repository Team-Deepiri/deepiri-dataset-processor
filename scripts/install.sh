#!/usr/bin/env bash
# Install deepiri-dataset-processor via curl:
#   curl -fsSL https://raw.githubusercontent.com/Team-Deepiri/deepiri-dataset-processor/main/scripts/install.sh | bash
set -euo pipefail

REPO="Team-Deepiri/deepiri-dataset-processor"
REPO_URL="https://github.com/${REPO}.git"
BRANCH="${DEEPIRI_DATASET_PROCESSOR_BRANCH:-main}"
KEEP_DIR="${DEEPIRI_DATASET_PROCESSOR_KEEP_DIR:-0}"
EXTRAS="${DEEPIRI_DATASET_PROCESSOR_EXTRAS:-}"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Clone (when needed) and pip-install deepiri-dataset-processor.

Options:
  -h, --help     Show this help
  --dry-run      Print actions without installing
  --all          Install [all] optional extras

Environment:
  DEEPIRI_DATASET_PROCESSOR_SRC       Existing checkout
  DEEPIRI_DATASET_PROCESSOR_BRANCH    Git branch (default: main)
  DEEPIRI_DATASET_PROCESSOR_KEEP_DIR  Keep clone when set to 1
  DEEPIRI_DATASET_PROCESSOR_EXTRAS    Pip extras, e.g. [semantic] or [all]

Requires: git, python3 (>=3.11)
Verify:   python3 -c "import deepiri_dataset_processor; print('ok')"
EOF
}

log() { printf '==> %s\n' "$*"; }

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --all) EXTRAS="[all]"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for cmd in git python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd is required." >&2; exit 1; }
done

ROOT=""
CLEANUP=""

if [[ -n "${DEEPIRI_DATASET_PROCESSOR_SRC:-}" && -f "${DEEPIRI_DATASET_PROCESSOR_SRC}/pyproject.toml" ]]; then
  ROOT="${DEEPIRI_DATASET_PROCESSOR_SRC}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != bash ]] && [[ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  ROOT="$(mktemp -d)"
  [[ "$KEEP_DIR" != "1" ]] && CLEANUP="$ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would clone ${REPO_URL} to ${ROOT}"
    log "Would pip install -e .${EXTRAS}"
    exit 0
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
fi

[[ "$DRY_RUN" -eq 1 ]] && { log "Would install from ${ROOT}"; exit 0; }

trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT
cd "$ROOT"

VENV="${ROOT}/.venv"
log "Creating venv at ${VENV}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel -q
log "Installing deepiri-dataset-processor${EXTRAS}"
"$VENV/bin/pip" install -e ".${EXTRAS}" -q

"$VENV/bin/python" -c "import deepiri_dataset_processor; print('deepiri-dataset-processor import ok')"
echo ""
echo "Activate: source ${VENV}/bin/activate"
echo "Verify:   python3 -c \"import deepiri_dataset_processor; print('ok')\""
