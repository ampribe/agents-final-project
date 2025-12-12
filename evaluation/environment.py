"""Agent virtual environment setup."""
import subprocess
from pathlib import Path


def setup_venv(venv_path: Path, packages: list[str]) -> Path:
    """
    Create virtual environment using uv and install packages.
    Returns path to python executable in the venv.
    """
    print(f"\n{'='*60}")
    print(f"Setting up virtual environment")
    print(f"{'='*60}")
    print(f"- Location: {venv_path}")
    print(f"- Packages: {len(packages)} packages to install")

    print(f"- Creating venv with uv...")
    subprocess.run(["uv", "venv", str(venv_path)], check=True)
    python_path = venv_path / "bin" / "python"

    if packages:
        print(f"- Installing packages...")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python_path)] + packages,
            check=True
        )
        print(f"- All packages installed\n")

    return python_path
