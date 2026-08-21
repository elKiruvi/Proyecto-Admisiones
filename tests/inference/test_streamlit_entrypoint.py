import os
import subprocess
import sys
from pathlib import Path


def test_streamlit_entrypoint_imports_without_repository_pythonpath() -> None:
    repository_root = Path(__file__).parents[2]
    entrypoint = repository_root / "src" / "inference" / "streamlit_app.py"
    script = """
import runpy
import sys
from pathlib import Path

entrypoint = Path(sys.argv[1]).resolve()
repository_root = entrypoint.parents[2]
source_root = repository_root / "src"
sys.path[:] = [
    path
    for path in sys.path
    if path not in {"", str(repository_root), str(source_root)}
]
runpy.run_path(str(entrypoint), run_name="__main__")
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(entrypoint)],
        check=False,
        cwd=repository_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
