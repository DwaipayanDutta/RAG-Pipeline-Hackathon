"""
Small evaluation harness over a queries file.

Accepted query formats (also used by `--queries`):
    ["question", ...]
    [{"query": "question", "expected": ["substring", ...] | "REFUSAL",
      "expect_conflict": true, "expect_tier": "cmdb"}, ...]

A query passes when every expected substring is in the answer (case-insensitive), or the
pipeline refused when "REFUSAL" was expected, and the optional conflict / tier checks hold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .models import REFUSAL, QueryResult

QueryItem = Dict[str, Any]


def load_queries(path: Union[str, Path]) -> List[QueryItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", [])
    items = []
    for d in data:
        if isinstance(d, str):
            items.append({"query": d})
        elif isinstance(d, dict) and d.get("query"):
            items.append(d)
        else:
            raise ValueError(f"Unsupported query entry in {path}: {d!r}")
    return items


def check(item: QueryItem, res: QueryResult) -> Tuple[bool, List[str]]:
    problems = []
    expected = item.get("expected")
    if expected == "REFUSAL":
        if res.answer != REFUSAL:
            problems.append("expected a refusal")
    elif expected:
        for s in ([expected] if isinstance(expected, str) else expected):
            if s.lower() not in res.answer.lower():
                problems.append(f"missing '{s}'")
    if item.get("expect_conflict") is True and not res.conflicts:
        problems.append("expected a conflict to be reported")
    if item.get("expect_tier") and res.priority_tier != item["expect_tier"]:
        problems.append(f"tier {res.priority_tier} != {item['expect_tier']}")
    return not problems, problems


def evaluate(pipeline, items: List[QueryItem]) -> Dict[str, Any]:
    rows = []
    for item in items:
        res = pipeline.ask(item["query"])
        ok, problems = check(item, res) if ("expected" in item or "expect_tier" in item
                                             or "expect_conflict" in item) else (None, [])
        rows.append({"query": item["query"], "passed": ok, "problems": problems, "answer": res.answer,
                     "priority_tier": res.priority_tier, "conflicts": len(res.conflicts),
                     "result": res.to_dict()})
    scored = [r for r in rows if r["passed"] is not None]
    passed = sum(1 for r in scored if r["passed"])
    return {"total": len(rows), "scored": len(scored), "passed": passed,
            "accuracy": round(passed / len(scored), 3) if scored else None, "rows": rows}


def format_report(report: Dict[str, Any]) -> str:
    lines = []
    for r in report["rows"]:
        mark = "PASS" if r["passed"] else ("----" if r["passed"] is None else "FAIL")
        lines.append(f"[{mark}] {r['query']}")
        lines.append(f"       -> {r['answer'][:160]}")
        if r["problems"]:
            lines.append(f"       !! {'; '.join(r['problems'])}")
    lines.append(f"\n{report['passed']}/{report['scored']} scored queries passed "
                 f"(accuracy {report['accuracy']}).")
    return "\n".join(lines)
