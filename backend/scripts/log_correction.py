"""Log a time the coach got something wrong, so it can become an eval case or a knowledge note.

    python scripts/log_correction.py --move "Cross step" --kind wrong_fact \
        --said "Straighten your right knee" --wrong "It was the left knee that was off" \
        --fix "Name the joint from FACTS only"
    python scripts/log_correction.py --list

Kinds:
  wrong_fact     the coach stated something about the body or score that was not true
  bad_priority   it picked the wrong thing to fix first
  bad_wording    the fact was right but the cue was confusing or off-style
  missed_issue   something was clearly wrong and the coach never said it

Each entry is one JSON line in evals/corrections.jsonl. Text only: no audio, no
video (CLAUDE.md rule 7). Optionally pass --facts '<json>' with the facts block
the coach was given, so a wrong_fact entry can be turned into a guard eval case.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "evals" / "corrections.jsonl"
KINDS = ("wrong_fact", "bad_priority", "bad_wording", "missed_issue")


def add(entry: dict, path: Path = LOG) -> dict:
    """Validate and append one entry. Returns the stored entry."""
    if entry.get("kind") not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    for key in ("move", "said", "wrong", "fix"):
        if not str(entry.get(key) or "").strip():
            raise ValueError(f"{key} is required")
    stored = {k: entry[k] for k in ("move", "kind", "said", "wrong", "fix") if k in entry}
    if entry.get("facts") is not None:
        stored["facts"] = entry["facts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(load(path)) + 1
    stored = {"id": f"c{n:03d}", "date": entry.get("date") or date.today().isoformat(), **stored}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stored, ensure_ascii=False) + "\n")
    return stored


def load(path: Path = LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the log and exit")
    ap.add_argument("--move")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--said", help="what the coach said")
    ap.add_argument("--wrong", help="what was actually wrong or true")
    ap.add_argument("--fix", help="what the coach should have said or done")
    ap.add_argument("--facts", help="the facts block the coach had, as JSON (optional)")
    args = ap.parse_args()

    if args.list:
        for e in load():
            print(f'{e["id"]} {e["date"]} [{e["kind"]}] {e["move"]}: said "{e["said"]}" | wrong: {e["wrong"]} | fix: {e["fix"]}')
        return 0
    try:
        facts = json.loads(args.facts) if args.facts else None
        stored = add({"move": args.move, "kind": args.kind, "said": args.said,
                      "wrong": args.wrong, "fix": args.fix, "facts": facts})
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"logged {stored['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
