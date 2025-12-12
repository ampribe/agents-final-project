#!/bin/bash

set -e

echo "Running all agent variants on 10 tasks..."
echo "Results will be saved to data/runs/"
echo ""

echo "=== [1/4] Running Single Agent ==="
uv run python3 scripts/run_batch.py --batch-config scripts/batch_single_agent.yaml
echo ""

echo "=== [2/4] Running Multi-Agent Full (Researcher + Coder + Tester) ==="
uv run python3 scripts/run_batch.py --batch-config scripts/batch_multi_full.yaml
echo ""

echo "=== [3/4] Running Multi-Agent No Researcher (Coder + Tester) ==="
uv run python3 scripts/run_batch.py --batch-config scripts/batch_multi_no_researcher.yaml
echo ""

echo "=== [4/4] Running Multi-Agent Coder Only ==="
uv run python3 scripts/run_batch.py --batch-config scripts/batch_multi_no_researcher_no_tester.yaml
echo ""

echo "✓ All runs complete!"
echo "Total runs: 40 (4 configurations × 10 tasks)"
echo ""
echo "Results structure:"
echo "  data/runs/single/                (10 task runs)"
echo "  data/runs/multi_full/            (10 task runs)"
echo "  data/runs/multi_no_researcher/   (10 task runs)"
echo "  data/runs/multi_coder_only/      (10 task runs)"
