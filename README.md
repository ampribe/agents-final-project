# AlgorithmOptimizer: A Multi-Agent System for Optimizing Numerical Computing in Python
This repository includes the code for my paper.
`agents` includes the code for the agents, including the single agent, the multi-agent system, and variants of the multi-agent system.
`evaluation` includes code for running the agents on the AlgoTune benchmark.
`data` includes the logs of all of the agent runs used for the paper.

To run the agents, you will need to do the following.
1. Clone this repository. `git clone git@github.com:ampribe/agents-final-project.git`
2. Clone the AlgoTune benchmark repository in the project root. `git clone git@github.com:oripress/AlgoTune.git`
3. Install the `uv` package manager (for managing the virtual environments used by the evaluation harness). `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. Install dependencies in a virtual environment. `uv sync` or `python3 -m venv .venv`, `pip3 install .`, and `source .venv/bin/activate`
5. Authenticate with the Globus inference service. `uv run inference_auth_token.py authenticate`
6. Create the desired output directory.
7. You can now run the code. You can run with the following agents: `agents.single_agent:SingleAgent`, `agents.multi_agent_variants:MultiAgentFull`, `agents.multi_agent_variants:MultiAgentNoResearcher`, and `agents.multi_agent_variants:MultiAgentCoderOnly`. A directory will be created with the agent's solution, logs, and results of the tests.

```
uv run run_evaluation.py \
    --agent agents.single_agent:SingleAgent \
    --model openai/gpt-oss-120b \
    --task minimum_spanning_tree \
    --runs-dir data/runs
    --max-steps 30
```

```
uv run run_evaluation.py \
    --agent agents.multi_agent_variants:MultiAgentFull \
    --model openai/gpt-oss-120b \
    --tasks-file evaluation/tasks.yaml
    --runs-dir data/runs
```


