#!/usr/bin/env bash
# Download the raw LOBSTER AAPL sample (110 MB) into data/raw/ and rebuild
# the processed file. Official source: LOBSTER's free sample files,
# https://lobsterdata.com/info/DataSamples.php (AAPL, 2012-06-21, 10 levels).
# This script uses a GitHub mirror whose SHA-256 matches (see SOURCES.md).
set -euo pipefail
cd "$(dirname "$0")"
REPO=https://github.com/amaiti2/queue-aware-lob-alpha
DIR=data/LOBSTER_SampleFile_AAPL_2012-06-21_10
rm -rf _mirror raw
git clone --depth 1 --filter=blob:none --sparse -q "$REPO" _mirror
git -C _mirror sparse-checkout set "$DIR"
mkdir -p raw && mv "_mirror/$DIR"/*.csv raw/ && rm -rf _mirror
sha256sum raw/*.csv
python ../src/lobster.py raw
