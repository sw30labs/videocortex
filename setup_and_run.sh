#!/usr/bin/env bash
# Set up videocortex from a source checkout, run the test suite, and render
# the synthetic sample. Does not download the 20 GB model stack unless you
# ask for it.
#
# Usage:
#   ./setup_and_run.sh                 # venv + tests + synthetic sample
#   ./setup_and_run.sh --predict       # + tribev2 + torch
#   ./setup_and_run.sh --fetch         # also pull the five HuggingFace repos
#   ./setup_and_run.sh --render FILE   # run the model on a clip (implies --predict)
#   ./setup_and_run.sh --setup-only    # venv + deps + tests, no sample / fetch / render
#   ./setup_and_run.sh --deck          # then start the loopback command deck
#   ./setup_and_run.sh --no-tests      # skip pytest
#   ./setup_and_run.sh --no-sample     # skip the synthetic demo
#   ./setup_and_run.sh --no-open       # do not open the contact sheet / deck tab
#   ./setup_and_run.sh --help
#
# Extra flags after `--` go to `videocortex render` when --render is set:
#   ./setup_and_run.sh --render clip.mp4 -- --max-frames 12 --views full
#
# Env overrides:
#   VIDEOCORTEX_PYTHON     interpreter for the venv (>= 3.11)
#   VIDEOCORTEX_NO_OPEN=1  same as --no-open
#
# The default path never installs torch. That is deliberate: the renderer
# should prove itself on this machine before you commit to the model.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ORIG_PWD="$PWD"
cd "$ROOT"

SETUP_ONLY=0
RUN_TESTS=1
RUN_SAMPLE=1
OPEN=1
INSTALL_PREDICT=0
FETCH=0
DECK=0
RENDER_PATH=""
RENDER_EXTRA=()
VENV=.venv

usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; }

require_value() {
  if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [ "${2#--}" != "$2" ]; then
    echo "ERROR: $1 needs a value" >&2
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --predict)     INSTALL_PREDICT=1 ;;
    --fetch)
      FETCH=1
      INSTALL_PREDICT=1
      ;;
    --render)
      require_value "$@"
      RENDER_PATH="$2"
      INSTALL_PREDICT=1
      shift
      ;;
    --setup-only)  SETUP_ONLY=1 ;;
    --deck)        DECK=1 ;;
    --no-tests)    RUN_TESTS=0 ;;
    --no-sample)   RUN_SAMPLE=0 ;;
    --no-open)     OPEN=0 ;;
    --)
      shift
      RENDER_EXTRA=("$@")
      break
      ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "ERROR: unknown option '$1' (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

[ -n "${VIDEOCORTEX_NO_OPEN:-}" ] && OPEN=0

if [ "${#RENDER_EXTRA[@]}" -gt 0 ] && [ -z "$RENDER_PATH" ]; then
  echo "ERROR: extra arguments after -- need --render FILE" >&2
  exit 1
fi

if [ "$DECK" -eq 1 ] && [ "$SETUP_ONLY" -eq 1 ]; then
  echo "ERROR: --deck and --setup-only cannot be combined" >&2
  exit 1
fi

# ── paths from the caller's cwd ─────────────────────────────────────────────
# We cd to the repo root first, so a relative --render path has to be resolved
# against where the user actually typed it.
abspath_file() {
  local p="$1"
  if [ -f "$p" ]; then
    (cd "$(dirname "$p")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$p")")
    return 0
  fi
  if [ -f "$ORIG_PWD/$p" ]; then
    (cd "$(dirname "$ORIG_PWD/$p")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$p")")
    return 0
  fi
  return 1
}

if [ -n "$RENDER_PATH" ]; then
  _render_arg="$RENDER_PATH"
  if ! RENDER_PATH="$(abspath_file "$_render_arg")"; then
    echo "ERROR: --render file not found: $_render_arg" >&2
    exit 1
  fi
  unset _render_arg
fi

stimulus_flag() {
  local ext
  ext="$(printf '%s' "${1##*.}" | tr '[:upper:]' '[:lower:]')"
  case "$ext" in
    wav|mp3|m4a|flac|ogg|aac) printf '%s\n' --audio ;;
    txt|md)                   printf '%s\n' --text ;;
    *)                        printf '%s\n' --video ;;
  esac
}

# ── interpreter ─────────────────────────────────────────────────────────────
# pyproject requires >= 3.11. Prefer uv (it can fetch CPython 3.11 if this
# machine only has the macOS 3.9 stub); otherwise pick the newest on PATH.
# An existing venv that is too old is thrown away, not reused.
has_uv() { command -v uv >/dev/null 2>&1; }

python_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

pick_python() {
  local candidate
  if [ -n "${VIDEOCORTEX_PYTHON:-}" ]; then
    if [ ! -x "${VIDEOCORTEX_PYTHON}" ] && ! command -v "${VIDEOCORTEX_PYTHON}" >/dev/null 2>&1; then
      echo "ERROR: VIDEOCORTEX_PYTHON is not executable: $VIDEOCORTEX_PYTHON" >&2
      exit 1
    fi
    if ! python_ok "$VIDEOCORTEX_PYTHON"; then
      echo "ERROR: VIDEOCORTEX_PYTHON must be Python 3.11 or newer" >&2
      exit 1
    fi
    printf '%s\n' "$VIDEOCORTEX_PYTHON"
    return
  fi
  if [ -x "$VENV/bin/python" ] && python_ok "$VENV/bin/python"; then
    printf '%s\n' "$VENV/bin/python"
    return
  fi
  for candidate in python3.13 python3.12 python3.11 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo ""
}

if [ "$INSTALL_PREDICT" -eq 1 ] && ! has_uv; then
  echo "ERROR: uv is required to run the model (upstream shells out to \`uvx whisperx\`)." >&2
  echo "       Install it: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ -x "$VENV/bin/python" ] && ! python_ok "$VENV/bin/python"; then
  echo "==> Existing $VENV is Python < 3.11 — recreating"
  rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment in $VENV"
  if has_uv; then
    uv venv --python "${VIDEOCORTEX_PYTHON:-3.11}" "$VENV"
  else
    PY="$(pick_python)"
    if [ -z "$PY" ]; then
      echo "ERROR: no Python >= 3.11 on PATH. Install uv (https://docs.astral.sh/uv/)" >&2
      echo "       or set VIDEOCORTEX_PYTHON." >&2
      exit 1
    fi
    "$PY" -m venv "$VENV"
  fi
fi

PY="$VENV/bin/python"
if ! python_ok "$PY"; then
  echo "ERROR: $PY is not Python >= 3.11" >&2
  exit 1
fi
echo "==> Using $PY ($("$PY" --version 2>&1))"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

EXTRAS="dev"
if [ "$INSTALL_PREDICT" -eq 1 ]; then
  EXTRAS="predict,dev"
fi

echo "==> Installing videocortex (extras: $EXTRAS)"
if has_uv; then
  if [ "$INSTALL_PREDICT" -eq 1 ]; then
    echo "    tribev2 + torch — this is large and not quiet on purpose"
    uv pip install --python "$PY" -e ".[$EXTRAS]"
    echo "==> TRIBE v2 is CC-BY-NC-4.0 (non-commercial). Llama-3.2-3B is gated: hf auth login"
  else
    uv pip install --python "$PY" --quiet -e ".[$EXTRAS]"
  fi
else
  echo "    uv not on PATH — using pip. The model path needs uv later (\`uvx whisperx\`)."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e ".[$EXTRAS]"
fi

# Examples add the repo root to sys.path; PYTHONPATH covers a venv whose
# editable .pth was skipped (seen on some macOS layouts).
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

if [ "$RUN_TESTS" -eq 1 ]; then
  echo "==> Running test suite"
  "$PY" -m pytest -q
fi

echo "==> Preflight"
DOCTOR_RC=0
if [ "$INSTALL_PREDICT" -eq 1 ]; then
  "$PY" -m videocortex doctor || DOCTOR_RC=$?
else
  "$PY" -m videocortex doctor --renderer
  echo "    renderer-only. --predict when you want the model."
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
  echo "==> setup-only: sample not rendered"
  exit "$DOCTOR_RC"
fi

# Don't start a 20 GB download or an inference run with a known blocker.
# The synthetic sample does not need the model, so a gated Llama is not a
# reason to skip it.
if [ "$DOCTOR_RC" -ne 0 ] && { [ "$FETCH" -eq 1 ] || [ -n "$RENDER_PATH" ]; }; then
  echo "ERROR: doctor found blocking problems. Fix them before --fetch / --render." >&2
  exit "$DOCTOR_RC"
fi

SAMPLE_SHEET="examples/sample_run/contact_sheet.png"

if [ "$RUN_SAMPLE" -eq 1 ]; then
  echo "==> Synthetic sample (not model output)"
  "$PY" examples/make_sample.py
  "$PY" -m videocortex draw examples/sample_run/predictions.npy --max-frames 6
  echo "==> Sample at $SAMPLE_SHEET"
  if [ "$OPEN" -eq 1 ]; then
    if command -v open >/dev/null 2>&1; then
      open "$SAMPLE_SHEET"
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$SAMPLE_SHEET" >/dev/null 2>&1 || true
    fi
  fi
fi

if [ "$FETCH" -eq 1 ]; then
  echo "==> Fetching checkpoint + frozen encoders (~15–20 GB)"
  "$PY" -m videocortex fetch
fi

if [ -n "$RENDER_PATH" ]; then
  FLAG="$(stimulus_flag "$RENDER_PATH")"
  echo "==> Rendering $RENDER_PATH ($FLAG)"
  "$PY" -m videocortex render "$FLAG" "$RENDER_PATH" "${RENDER_EXTRA[@]}"
fi

if [ "$DECK" -eq 1 ]; then
  echo "==> Command deck → http://127.0.0.1:8730  (Ctrl+C to stop)"
  if [ "$OPEN" -eq 0 ]; then
    exec "$PY" -m videocortex serve --no-browser
  fi
  exec "$PY" -m videocortex serve
fi

if [ "$DOCTOR_RC" -ne 0 ]; then
  echo "==> Model path is not ready (see preflight). Sample does not need it."
  exit "$DOCTOR_RC"
fi
