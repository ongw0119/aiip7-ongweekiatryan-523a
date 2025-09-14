#!/bin/bash

# Gas Monitoring ML Pipeline Execution Script
# This script runs the machine learning pipeline for activity level prediction

echo "Starting Gas Monitoring ML Pipeline..."
echo "====================================="

# Change to the directory containing the script
cd "$(dirname "$0")"

# Run the main pipeline
python src/pipeline.py --db data/gas_monitoring.db --model all

echo "Pipeline execution completed!"
