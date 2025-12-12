"""Helpers to load configuration and import agents."""
import yaml
from pathlib import Path
import importlib


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def import_agent(agent_path: str):
    if ":" not in agent_path:
        raise ValueError("agent_path must be MODULE:CLASS, e.g. algotune_agent:AlgoTuneAgent")
    module_name, class_name = agent_path.split(":", 1)
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    return agent_cls
