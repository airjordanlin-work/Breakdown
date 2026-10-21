"""Eval the grounding guard against a hand-written, labeled set of coach replies.

    python scripts/eval_guard.py                 # offline, no API key
    python scripts/eval_guard.py --out ../docs/benchmarks/guard_eval.txt
    python scripts/eval_guard.py --model --n 3   # real Haiku on the facts blocks (needs ANTHROPIC_API_KEY)

Offline mode runs app.grounding.ungrounded_claims over every case in
evals/guard_cases.json and reports:
- false accepts: an ungrounded reply the guard let through
- false refusals: a grounded reply the guard refused
- templated fallbacks: every fallback line must pass the guard for its own facts

The case set is small and written by hand, so the numbers describe only these
cases. They are not a rate for real model output. Cases marked known_gap are
claims the guard is known to miss.

Model mode builds a debrief or live prompt for each facts block, asks the real
model, and reports how many replies the guard refused. The replies are not
labeled, so this measures refusal rate only, not correctness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.grounding import ungrounded_claims  # noqa: E402

CASES = ROOT / "evals" / "guard_cases.json"


def load() -> dict:
    return json.loads(CASES.read_text())


def run_offline(data: dict) -> dict:
    """Judge every case. Returns counts and the mismatches."""
    mismatches = []
    counts = {"grounded": 0, "ungrounded": 0, "known_gap": 0}
    for c in data["cases"]:
        flags = ungrounded_claims(c["text"], data["facts"][c["facts"]])
        refused = bool(flags)
        counts["grounded" if c["grounded"] else "ungrounded"] += 1
        if c.get("known_gap"):
            counts["known_gap"] += 1
        if refused == c["grounded"]:   # refused a grounded reply, or accepted an ungrounded one
            mismatches.append({**c, "flags": flags,
                               "kind": "false_refusal" if c["grounded"] else "false_accept"})
    return {"counts": counts, "mismatches": mismatches}


def fallback_failures(data: dict) -> list[str]:
    """Templated lines built from a facts block must pass the guard for that block."""
    from app.coach import fallback_debrief, fallback_cue
    bad = []
    for name, facts in data["facts"].items():
        if "reps" in facts:
            text = fallback_debrief(facts)
            flags = ungrounded_claims(text, facts)
            if flags:
                bad.append(f"{name}: '{text}' -> {flags}")
        elif "focus" in facts:
            focus = facts["focus"]
            result = {"results": [{"joint": focus["joint"], "tier": focus["tier"], "diff": 30.0,
                                   "fix": focus["fix"]}]} if focus["joint"] else \
                     {"results": [{"joint": "left elbow", "tier": "Perfect", "diff": 2.0, "fix": "bend"}]}
            for _ in range(8):
                text = fallback_cue(result if focus["joint"] or focus["tier"] == "Perfect" else None)
                scoped = facts if focus["joint"] else {"move": facts["move"]}
                flags = ungrounded_claims(text, scoped)
                if flags:
                    bad.append(f"{name}: '{text}' -> {flags}")
                    break
    return bad


def report_offline(data: dict) -> str:
    res = run_offline(data)
    c, mm = res["counts"], res["mismatches"]
    fa = [m for m in mm if m["kind"] == "false_accept"]
    fr = [m for m in mm if m["kind"] == "false_refusal"]
    unexpected = [m for m in mm if not m.get("known_gap")]
    fb = fallback_failures(data)
    lines = [
        f"[guard eval] {len(data['cases'])} hand-written cases: {c['grounded']} grounded replies, {c['ungrounded']} ungrounded replies",
        f"  false refusals (grounded reply refused): {len(fr)} of {c['grounded']}",
        f"  false accepts (ungrounded reply passed):  {len(fa)} of {c['ungrounded']}",
        f"  of those, marked known_gap: {len([m for m in mm if m.get('known_gap')])}; unexpected: {len(unexpected)}",
        f"  templated fallbacks that fail the guard: {len(fb)}",
    ]
    for m in mm:
        tag = "known gap" if m.get("known_gap") else "UNEXPECTED"
        lines.append(f"  - {m['kind']} [{tag}] {m['id']}: '{m['text']}' ({m.get('why', '')}) flags={m['flags']}")
    for line in fb:
        lines.append(f"  - fallback fails guard: {line}")
    lines.append("Covers only these cases. Not a rate for real model output.")
    return "\n".join(lines)


def _prompt_for(name: str, facts: dict):
    from app import coach
    if "reps" in facts:
        return coach.DEBRIEF_SYSTEM_PROMPT, coach._build_debrief_prompt(facts), coach.DEBRIEF_MAX_TOKENS, coach.DEBRIEF_TIMEOUT_SECONDS
    focus = facts.get("focus")
    result = None
    if focus and focus["joint"]:
        result = {"results": [{"joint": focus["joint"], "tier": focus["tier"], "diff": 30.0, "fix": focus["fix"]}]}
    elif focus:
        result = {"results": [{"joint": "left elbow", "tier": "Perfect", "diff": 2.0, "fix": "bend"}]}
    return coach.SYSTEM_PROMPT, coach._build_prompt(facts["move"], result, (), facts=facts), coach.MAX_TOKENS, coach.TIMEOUT_SECONDS


def run_model(data: dict, n: int) -> str:
    from app import coach
    if not coach.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set; model mode makes real API calls.")

    async def go():
        total = refused = failed = 0
        lines = []
        for name, facts in data["facts"].items():
            system, prompt, max_tokens, timeout = _prompt_for(name, facts)
            for _ in range(n):
                text = await coach.anthropic_complete(system, prompt, max_tokens, timeout)
                total += 1
                if not text:
                    failed += 1
                    continue
                flags = ungrounded_claims(text.strip(), facts)
                if flags:
                    refused += 1
                    lines.append(f"  - {name}: '{text.strip()}' -> {flags}")
        return total, refused, failed, lines

    total, refused, failed, lines = asyncio.run(go())
    out = [f"[guard eval, model] {coach.MODEL}, {len(data['facts'])} facts blocks x {n} = {total} real API calls",
           f"  no reply (timeout or error): {failed}",
           f"  refused by the guard: {refused} of {total - failed}"]
    out += lines
    out.append("Refusal rate only; replies are not labeled for correctness.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="store_true", help="also run the real model (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--n", type=int, default=3, help="model mode: replies per facts block")
    ap.add_argument("--out", help="write the report to this file")
    args = ap.parse_args()
    data = load()
    text = run_model(data, args.n) if args.model else report_offline(data)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
