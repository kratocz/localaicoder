#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
uv run "$SCRIPT_DIR/main.py"
