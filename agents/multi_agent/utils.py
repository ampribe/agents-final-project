from pathlib import Path


def rel_path(workdir: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(workdir.resolve())
        return rel.as_posix()
    except Exception:
        return target.resolve().as_posix()
