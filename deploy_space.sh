#!/bin/bash
set -e
SPACE_DIR=../doc-qa-rag-space
mkdir -p "$SPACE_DIR/src" "$SPACE_DIR/data"
cp app.py "$SPACE_DIR/"
cp src/__init__.py src/config.py src/retrieve.py src/generate.py "$SPACE_DIR/src/"
cp data/chunks.jsonl "$SPACE_DIR/data/"
rm -rf "$SPACE_DIR/data/index" && cp -r data/index "$SPACE_DIR/data/index"
cp requirements-space.txt "$SPACE_DIR/requirements.txt"
cp SPACE_README.md "$SPACE_DIR/README.md"
echo "Copied."
