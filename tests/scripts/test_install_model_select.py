"""Tests for the model-selection logic in the installers (TASK-062).

Both installers expose a QUIPU_TEST_MODEL_SELECT_ONLY=1 test hook: the
script runs its normal model-selection + config-persist steps, prints
``CHOSEN_MODEL=<value>``, and exits 0 — skipping venv/pip/network steps
entirely. This lets us exercise the selection logic via subprocess without
running a full install.

Coverage (non-interactive / piped-stdin path — this is what subprocess.run
drives; stdin is never a TTY under pytest):
  (a) No saved model, no input -> resolves to "none" (keyword-only),
      prints an informational note, exit code 0.
  (b) A previously saved real model -> honored verbatim.
  (c) A previously saved "none" -> round-trips to "none".
  (d) An unrecognized/corrupt saved value -> resolves to "none", exit 0
      (no silent substitution of a specific model).
  (e) Persisted config file contains "MODEL=none" verbatim after a
      keyword-only resolution.

NOT covered here (documented limitation, see class docstring below):
  the interactive (real-TTY) prompt loop's "invalid input re-prompts,
  then a valid answer resolves it" behavior. Both installers gate the
  interactive loop behind an actual TTY check (`[ -t 0 ]` /
  `[System.Console]::IsInputRedirected`) so that piped/CI installs never
  hang on a prompt. Subprocess-piped stdin is therefore never a TTY and
  cannot reach that branch; simulating a real TTY would need a pty, which
  is not available on Windows (this sandbox). The loop's correctness
  (no auto-accepted default, 'none' always valid, invalid input re-prompts,
  EOF aborts instead of spinning) is verified by direct code inspection of
  the `while` / `while ($null -eq $choice)` loops in the two scripts.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SH_INSTALLER = _REPO_ROOT / "scripts" / "install-quipu-global.sh"
_PS1_INSTALLER = _REPO_ROOT / "install.ps1"


def _bash_safe_path(path: Path) -> str:
    path_string = path.as_posix()
    if sys.platform != "win32" or path_string[1:3] != ":/":
        return path_string

    bash_root = "/mnt" if subprocess.run(
        ["bash", "-c", "test -d /mnt/c"], capture_output=True
    ).returncode == 0 else ""
    return f"{bash_root}/{path_string[0].lower()}{path_string[2:]}"


def _run_sh(tmp_path: Path, config_lines: list[str] | None = None) -> subprocess.CompletedProcess:
    quipu_home = tmp_path / "quipu-home"
    quipu_home.mkdir()
    if config_lines is not None:
        (quipu_home / "config").write_text("\n".join(config_lines) + "\n", newline="\n")

    env = os.environ.copy()

    command = (
        f"QUIPU_HOME={shlex.quote(_bash_safe_path(quipu_home))} "
        "QUIPU_TEST_MODEL_SELECT_ONLY=1 "
        f"exec bash {shlex.quote(_bash_safe_path(_SH_INSTALLER))}"
    )
    return subprocess.run(
        ["bash", "-c", command],
        input="",
        capture_output=True,
        text=True,
        env=env,
    )


def _chosen_model(result: subprocess.CompletedProcess) -> str:
    for line in result.stdout.splitlines():
        if line.startswith("CHOSEN_MODEL="):
            return line.split("=", 1)[1]
    raise AssertionError(f"CHOSEN_MODEL= not found in stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


class TestShInstallerNonInteractive:
    def test_no_saved_model_resolves_to_none(self, tmp_path):
        result = _run_sh(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert _chosen_model(result) == "none"
        assert "keyword-only" in result.stdout.lower()

    def test_saved_real_model_is_honored(self, tmp_path):
        result = _run_sh(tmp_path, config_lines=["MODEL=bge-m3"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert _chosen_model(result) == "bge-m3"

    def test_saved_none_roundtrips(self, tmp_path):
        result = _run_sh(tmp_path, config_lines=["MODEL=none"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert _chosen_model(result) == "none"

    def test_unrecognized_saved_value_resolves_to_none_not_a_random_model(self, tmp_path):
        result = _run_sh(tmp_path, config_lines=["MODEL=nomic-embed-v1-typo"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert _chosen_model(result) == "none"

    def test_config_persists_none_verbatim(self, tmp_path):
        result = _run_sh(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        config_text = (tmp_path / "quipu-home" / "config").read_text()
        assert "MODEL=none" in config_text.splitlines()


@pytest.mark.skipif(sys.platform != "win32", reason="install.ps1 requires PowerShell")
class TestPs1InstallerNonInteractive:
    def _run(self, tmp_path: Path, config_lines: list[str] | None = None) -> subprocess.CompletedProcess:
        quipu_home = tmp_path / "quipu-home"
        quipu_home.mkdir()
        if config_lines is not None:
            (quipu_home / "config").write_text("\n".join(config_lines) + "\n")

        env = os.environ.copy()
        env["QUIPU_HOME"] = str(quipu_home)
        env["QUIPU_TEST_MODEL_SELECT_ONLY"] = "1"

        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(_PS1_INSTALLER)],
            input="",
            capture_output=True,
            text=True,
            env=env,
        )

    def _chosen_model(self, result: subprocess.CompletedProcess) -> str:
        for line in result.stdout.splitlines():
            if line.startswith("CHOSEN_MODEL="):
                return line.split("=", 1)[1].strip()
        raise AssertionError(f"CHOSEN_MODEL= not found in stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def test_no_saved_model_resolves_to_none(self, tmp_path):
        result = self._run(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self._chosen_model(result) == "none"
        assert "keyword-only" in result.stdout.lower()

    def test_saved_real_model_is_honored(self, tmp_path):
        result = self._run(tmp_path, config_lines=["MODEL=bge-m3"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self._chosen_model(result) == "bge-m3"

    def test_saved_none_roundtrips(self, tmp_path):
        result = self._run(tmp_path, config_lines=["MODEL=none"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self._chosen_model(result) == "none"

    def test_config_persists_none_verbatim(self, tmp_path):
        result = self._run(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        config_text = (tmp_path / "quipu-home" / "config").read_text()
        assert "MODEL=none" in config_text.splitlines()
