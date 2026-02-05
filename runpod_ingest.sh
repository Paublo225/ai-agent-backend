#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script assumes you are in the root of your project directory (/workspace on RunPod).

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Starting PDF ingestion with vision analysis..."
# Replace /path/to/your/manuals with the actual path to your PDF manuals on RunPod
# Example: If your manuals are in /workspace/manuals, use /workspace/manuals
python -m backend.ingestion manuals --with-vision --state-file .ingestion/runpod_state.json

echo "Ingestion complete!"

