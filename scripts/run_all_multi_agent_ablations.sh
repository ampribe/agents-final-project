#!/bin/bash

set -e

echo "Running MultiAgent ablation study on all 10 tasks..."
echo "Results will be saved to data/multi_reproduced/{full,no_researcher,no_tester,coder_only}/"
echo ""

echo "=== [1/4] Running FULL (Researcher + Coder + Tester) ==="
python3 scripts/run_batch.py --batch-config scripts/multi_agent_full.yaml
echo ""

echo "=== [2/4] Running NO_RESEARCHER (Coder + Tester) ==="
python3 scripts/run_batch.py --batch-config scripts/multi_agent_no_researcher.yaml
echo ""

echo "=== [3/4] Running NO_TESTER (Researcher + Coder) ==="
python3 scripts/run_batch.py --batch-config scripts/multi_agent_no_tester.yaml
echo ""

echo "=== [4/4] Running CODER_ONLY ==="
python3 scripts/run_batch.py --batch-config scripts/multi_agent_coder_only.yaml
echo ""

echo "✓ All ablations complete!"
echo "Total runs: 40 (4 configurations × 10 tasks)"
echo ""
echo "Results structure:"
echo "  data/multi_reproduced/full/          - 10 runs"
echo "  data/multi_reproduced/no_researcher/ - 10 runs"
echo "  data/multi_reproduced/no_tester/     - 10 runs"
echo "  data/multi_reproduced/coder_only/    - 10 runs"
