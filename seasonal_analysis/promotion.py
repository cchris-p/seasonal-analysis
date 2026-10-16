"""Promotion evidence report for seasonal windows (``SEAS-001F``).

Assembles every confidence criterion computed by ``SEAS-001C``/``SEAS-001D`` into a
machine JSON payload plus a self-contained Markdown report. The Markdown is meant
to be pasted into an OpenCode session for review.

Promotion itself is **human-decided** (``SEA-EXE-001``): this module only reports
evidence; it never auto-promotes and applies no automatic hard gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _get(d: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _criterion(
    *, key: str, value: Any, status: str, detail: str = ""
) -> Dict[str, Any]:
    return {"key": key, "value": value, "status": status, "detail": detail}


def build_promotion_report(
    *,
    window: Dict[str, Any],
    evidence: Dict[str, Any],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the promotion evidence report (JSON shape) for one window.

    Args:
        window: ``symbol``, ``entry_mmdd``, ``exit_mmdd``, ``direction``,
            ``direction_mode``, ``curve_mode``.
        evidence: ``discovery`` (gross/net metrics), ``validation`` (train/test),
            ``stats`` (``data_snooping``/``robustness``/``min_sample``), and
            ``assumptions`` (``data_source``/``data_provenance``/``cost_model``).
        generated_at: Optional ISO timestamp.

    Returns:
        Report dict with ``criteria``, ``summary`` and the raw ``evidence``.
    """
    evidence = evidence or {}
    spa = _get(evidence, "stats", "data_snooping", "spa") or {}
    rc = _get(evidence, "stats", "data_snooping", "reality_check") or {}
    robustness = _get(evidence, "stats", "robustness") or {}
    min_sample = _get(evidence, "stats", "min_sample") or {}
    validation = evidence.get("validation") or {}
    discovery = evidence.get("discovery") or {}
    assumptions = evidence.get("assumptions") or {}

    spa_p = spa.get("p_value")
    rc_p = rc.get("p_value")
    n_strategies = spa.get("n_strategies") if spa else rc.get("n_strategies")
    full_grid = _get(evidence, "stats", "data_snooping", "full_grid_size")

    jitter = robustness.get("jitter") or {}
    subperiod = robustness.get("subperiod_consistency") or {}

    criteria: List[Dict[str, Any]] = [
        _criterion(
            key="data_snooping",
            value={"hansen_spa_p": spa_p, "reality_check_p": rc_p},
            status=("pass" if isinstance(spa_p, (int, float)) and spa_p <= 0.05 else "review"),
            detail=(
                f"Hansen SPA headline (candidates={n_strategies}, search-space={full_grid}); "
                "p<=0.05 is evidence against data snooping."
            ),
        ),
        _criterion(
            key="out_of_sample",
            value={
                "split_mode": validation.get("split_mode"),
                "test_avg_return": _get(validation, "test_metrics", "avg_return"),
                "test_win_rate": _get(validation, "test_metrics", "win_rate"),
                "test_trades": _get(validation, "test_metrics", "num_trades"),
            },
            status=(
                "pass"
                if (_get(validation, "test_metrics", "avg_return") or 0) > 0
                else "review"
            ),
            detail="Out-of-sample/holdout or walk-forward result (test metrics only).",
        ),
        _criterion(
            key="sample_adequacy",
            value={
                "independent_years": min_sample.get("independent_years"),
                "test_trades": min_sample.get("test_trades"),
            },
            status=(
                "pass"
                if int(min_sample.get("independent_years") or 0) >= 8
                else "review"
            ),
            detail="Independent years (soft floor ~8-10); promotion is human-decided.",
        ),
        _criterion(
            key="robustness_jitter",
            value=jitter,
            status=(
                "pass"
                if (jitter.get("fraction_positive") or 0) >= 0.8
                else "review"
            ),
            detail="Share of +/-2 trading-day window shifts with a positive mean return.",
        ),
        _criterion(
            key="robustness_subperiods",
            value=subperiod,
            status=("pass" if subperiod.get("all_positive") else "review"),
            detail="Early/late subperiod consistency of the return series.",
        ),
        _criterion(
            key="cost",
            value={
                "costs_included": _get(discovery, "costs_included")
                or bool(assumptions.get("costs_included")),
                "net_avg_profit_points": _get(discovery, "net_avg_profit_points"),
                "net_win_rate": _get(discovery, "net_win_rate"),
                "cost_model": assumptions.get("cost_model"),
            },
            status=(
                "pass"
                if (_get(discovery, "net_avg_profit_points") or 0) > 0
                and bool(_get(discovery, "costs_included") or assumptions.get("costs_included"))
                else "review"
            ),
            detail="Net-of-cost metrics (tick-derived variable spread + commission).",
        ),
        _criterion(
            key="data_provenance",
            value={
                "data_source": assumptions.get("data_source"),
                "time_base": _get(assumptions, "data_provenance", "time_base"),
                "quality": _get(assumptions, "data_provenance", "quality"),
            },
            status=(
                "pass"
                if str(assumptions.get("data_source") or "") == "dukascopy"
                else "review"
            ),
            detail="Canonical Dukascopy GMT+0_US-DST source with asserted provenance.",
        ),
    ]

    statuses = [c["status"] for c in criteria]
    if all(s == "pass" for s in statuses):
        overall = "strong"
    elif any(s == "pass" for s in statuses):
        overall = "mixed"
    else:
        overall = "insufficient"

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "report_kind": "seasonal_window_promotion_evidence",
        "promotion_kind": "human_decided",
        "window": {
            "symbol": window.get("symbol"),
            "entry_mmdd": window.get("entry_mmdd"),
            "exit_mmdd": window.get("exit_mmdd"),
            "direction": window.get("direction"),
            "direction_mode": window.get("direction_mode"),
            "curve_mode": window.get("curve_mode"),
        },
        "criteria": criteria,
        "summary": {
            "overall_evidence": overall,
            "n_pass": sum(1 for s in statuses if s == "pass"),
            "n_review": sum(1 for s in statuses if s == "review"),
        },
        "evidence": evidence,
    }
    return report


def render_promotion_markdown(report: Dict[str, Any]) -> str:
    """Render a self-contained Markdown promotion report (paste-into-OpenCode)."""
    w = report.get("window") or {}
    crit = {c.get("key"): c for c in (report.get("criteria") or [])}
    summary = report.get("summary") or {}

    def crit_row(key: str) -> str:
        c = crit.get(key) or {}
        status = str(c.get("status") or "review").upper()
        return f"| `{key}` | {status} | {c.get('detail') or ''} |"

    lines: List[str] = []
    lines.append(
        f"# Seasonal window promotion evidence — {w.get('symbol')} "
        f"{w.get('entry_mmdd')}→{w.get('exit_mmdd')}"
    )
    lines.append("")
    lines.append(f"- Generated: {report.get('generated_at')}")
    lines.append(f"- Direction: {w.get('direction')} ({w.get('direction_mode')})")
    lines.append(f"- Curve mode: `{w.get('curve_mode')}`")
    lines.append(f"- Overall evidence: **{summary.get('overall_evidence')}** "
                 f"({summary.get('n_pass')} pass / {summary.get('n_review')} review)")
    lines.append("")
    lines.append("Promotion is a **human decision** informed by this report; the report "
                 "applies no automatic gate.")
    lines.append("")
    lines.append("## Confidence criteria")
    lines.append("")
    lines.append("| Criterion | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    for key in (
        "data_snooping",
        "out_of_sample",
        "sample_adequacy",
        "robustness_jitter",
        "robustness_subperiods",
        "cost",
        "data_provenance",
    ):
        lines.append(crit_row(key))
    lines.append("")

    ds = crit.get("data_snooping", {}).get("value") or {}
    lines.append("## Significance (data snooping)")
    lines.append("")
    lines.append(f"- Hansen SPA p-value: **{_fmt(ds.get('hansen_spa_p'))}** (headline)")
    lines.append(f"- White's Reality Check p-value: {_fmt(ds.get('reality_check_p'))}")
    ds_meta = (report.get("evidence") or {}).get("stats", {}).get("data_snooping") or {}
    lines.append(f"- Candidates used: {ds_meta.get('n_candidates_used')} "
                 f"(full grid {ds_meta.get('full_grid_size')}, periods {ds_meta.get('n_periods')})")
    lines.append("")

    oos = crit.get("out_of_sample", {}).get("value") or {}
    lines.append("## Out-of-sample")
    lines.append("")
    lines.append(f"- Split mode: `{oos.get('split_mode')}`")
    lines.append(f"- Test avg return: {_fmt(oos.get('test_avg_return'))}, "
                 f"win rate {_pct(oos.get('test_win_rate'))}, trades {oos.get('test_trades')}")
    lines.append("")

    jit = crit.get("robustness_jitter", {}).get("value") or {}
    lines.append("## Robustness")
    lines.append("")
    lines.append(f"- Jitter (±{jit.get('jitter_days', 2)} trading days): "
                 f"{jit.get('n_variants')} variants, median {_fmt(jit.get('median'))}, "
                 f"fraction positive {_pct(jit.get('fraction_positive'))}")
    sub = crit.get("robustness_subperiods", {}).get("value") or {}
    lines.append(f"- Subperiods: all positive = {sub.get('all_positive')}, "
                 f"sign consistency {_pct(sub.get('sign_consistency'))}")
    lines.append("")

    cost = crit.get("cost", {}).get("value") or {}
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- Costs included: {cost.get('costs_included')}")
    lines.append(f"- Net avg profit (points): {_fmt(cost.get('net_avg_profit_points'))}, "
                 f"net win rate {_pct(cost.get('net_win_rate'))}")
    lines.append("")

    prov = crit.get("data_provenance", {}).get("value") or {}
    lines.append("## Data provenance")
    lines.append("")
    lines.append(f"- Source: `{prov.get('data_source')}` (time base {prov.get('time_base')}, "
                 f"quality {prov.get('quality')})")
    lines.append("")
    lines.append("---")
    lines.append("Paste this report into an OpenCode session to record the promotion decision.")
    return "\n".join(lines) + "\n"
