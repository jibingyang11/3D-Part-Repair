#!/bin/bash
# Package experiment results for submission
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_NAME="3dpart_results_${TIMESTAMP}"

echo "Packaging results..."

# Create archive
tar -czf "${OUTPUT_NAME}.tar.gz" \
    outputs/tables/ \
    outputs/figures/ \
    outputs/metrics/ \
    configs/ \
    README.md \
    requirements.txt

echo "Results packaged: ${OUTPUT_NAME}.tar.gz"
