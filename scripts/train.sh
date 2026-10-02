#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/nano.yaml}"
python -m deon6.training.pretrain --config "$CONFIG"
