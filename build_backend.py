#!/usr/bin/env python3
"""
PEP 517 build backend wrapper for maturin.

This ensures Groggy stubs are regenerated on each wheel/editable build by:
1) Building the extension in-place (maturin develop --release --skip-install)
2) Running the dynamic stub generator via scripts/generate_stubs.py
3) Delegating to maturin's PEP 517 hooks
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import maturin as _maturin


REPO_ROOT = Path(__file__).resolve().parent


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    python_src = str(REPO_ROOT / "python-groggy" / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_src if not existing else f"{python_src}{os.pathsep}{existing}"
    return env


def _ensure_extension_built() -> None:
    cmd = [sys.executable, "-m", "maturin", "develop", "--release", "--skip-install"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_subprocess_env())


def _generate_stubs() -> None:
    cmd = [sys.executable, "scripts/generate_stubs.py"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_subprocess_env())


def _prep_for_build() -> None:
    _ensure_extension_built()
    _generate_stubs()


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    _prep_for_build()
    return _maturin.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    _prep_for_build()
    return _maturin.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(
    sdist_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return _maturin.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    return _maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    return _maturin.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    return _maturin.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return _maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return _maturin.prepare_metadata_for_build_editable(metadata_directory, config_settings)
