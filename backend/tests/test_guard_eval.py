"""The labeled guard eval (scripts/eval_guard.py): only known gaps may be missed."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_guard.py"
spec = importlib.util.spec_from_file_location("eval_guard", SCRIPT)
eval_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_guard)


def test_every_mismatch_is_a_marked_known_gap():
    data = eval_guard.load()
    result = eval_guard.run_offline(data)
    unexpected = [m["id"] for m in result["mismatches"] if not m.get("known_gap")]
    assert unexpected == []


def test_known_gaps_are_still_gaps():
    # If the guard starts catching one, remove its known_gap mark so the report stays honest.
    data = eval_guard.load()
    missed = {m["id"] for m in eval_guard.run_offline(data)["mismatches"]}
    marked = {c["id"] for c in data["cases"] if c.get("known_gap")}
    assert marked == missed


def test_templated_fallbacks_pass_the_guard():
    assert eval_guard.fallback_failures(eval_guard.load()) == []


def test_case_set_covers_both_labels_and_all_facts_blocks():
    data = eval_guard.load()
    labels = {c["grounded"] for c in data["cases"]}
    assert labels == {True, False}
    assert {c["facts"] for c in data["cases"]} <= set(data["facts"])


def test_metrics_match_the_case_counts():
    data = eval_guard.load()
    m = eval_guard.metrics(data)
    grounded = sum(1 for c in data["cases"] if c["grounded"])
    assert m["tp"] + m["fn"] == len(data["cases"]) - grounded
    assert m["fp"] + m["tn"] == grounded
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["false_refusal_rate"] <= 1.0
    live, post = m["by_kind"]["live"], m["by_kind"]["post_rep"]
    assert live[1] > 0 and post[1] > 0 and live[1] + post[1] == m["tp"] + m["fn"]


def test_report_includes_the_rates():
    assert "false-refusal rate" in eval_guard.report_offline(eval_guard.load())
