"""Claude Code source — wraps the existing ``..extractor`` parser so it
exposes the same ``load_all`` / ``PROJECTS_DIR`` surface as ``cursor``."""

from __future__ import annotations

from .. import extractor as _ex

PROJECTS_DIR = _ex.PROJECTS_DIR


def load_all(projects_dir=None):
    return _ex.load_all(projects_dir or _ex.PROJECTS_DIR)
