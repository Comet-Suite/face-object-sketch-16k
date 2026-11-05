#!/usr/bin/env bash
# Unpack every images/*.tar shard into ./images/ (safe to re-run).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! compgen -G "images/*.tar" > /dev/null; then
  echo "No tar shards found in images/. Nothing to do."
  exit 0
fi

for t in images/*.tar; do
  echo "unpacking $t"
  tar -xf "$t"
done

n=$(find images -name '*.jpg' | wc -l | tr -d ' ')
echo "done — $n JPEGs in ./images/"
