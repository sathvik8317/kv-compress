#!/usr/bin/env bash
# Fetch LongBench (THUDM/LongBench, ~114 MB) into data/longbench_raw/data/*.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/longbench_raw
curl -L -o data/longbench_raw/data.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -q -o data/longbench_raw/data.zip -d data/longbench_raw
rm data/longbench_raw/data.zip
echo "done: $(ls data/longbench_raw/data | wc -l) files"
