"""Render a WrappedReport (a SimpleNamespace tree) to self-contained HTML.

The Jinja template only does attribute access on ``r``, so any object
that exposes the right attributes works — ``api_client.analyze`` returns
a ``SimpleNamespace`` recursively built from the backend's JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape


def _format_number(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _format_money(n, decimals: int = 0) -> str:
    try:
        return f"{float(n):,.{decimals}f}"
    except Exception:
        return str(n)


def render_html(report: Any, out_path: Path) -> Path:
    env = Environment(
        loader=PackageLoader("tracercc", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["number"] = _format_number
    env.filters["money"] = _format_money
    tpl = env.get_template("dashboard.html.j2")
    html = tpl.render(r=report)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
