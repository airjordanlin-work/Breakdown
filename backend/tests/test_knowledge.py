import json
import sys
from pathlib import Path

import pytest

from app import knowledge
from app.chat import CoachChat
from app.coach import _build_debrief_prompt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import log_correction  # noqa: E402


@pytest.mark.parametrize("path", sorted(knowledge.KNOWLEDGE_DIR.glob("*.md")), ids=lambda p: p.name)
def test_every_knowledge_file_passes_lint(path):
    assert knowledge.lint(path) == []


def test_lint_rejects_numbers_and_missing_sections(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("---\nmove: Test\n---\n\n## Key points\n- Bend your knee 90 degrees\n")
    problems = "\n".join(knowledge.lint(bad))
    assert "measurement word or digit" in problems
    assert "missing or empty section 'Fixes'" in problems


def test_block_for_known_and_unknown_move():
    block = knowledge.knowledge_block("cross step")
    assert block.startswith("REFERENCE") and "NOT measured" in block
    assert len(block) <= knowledge.MAX_BLOCK_CHARS
    assert knowledge.knowledge_block("Windmill") == ""
    assert knowledge.knowledge_block(None) == ""


def test_debrief_prompt_includes_reference_only_for_known_moves():
    assert "REFERENCE" in _build_debrief_prompt({"move": "Cross step", "reps": 2})
    assert "REFERENCE" not in _build_debrief_prompt({"move": "Windmill", "reps": 2})


def test_chat_message_includes_reference():
    chat = CoachChat(llm=None)
    msgs = chat._messages("why does it matter?", {"move": "Cross step"})
    assert "REFERENCE" in msgs[-1]["content"]
    assert msgs[-1]["content"].rstrip().endswith("Dancer: why does it matter?")


def test_correction_log_roundtrip(tmp_path):
    path = tmp_path / "c.jsonl"
    e = log_correction.add({"move": "Cross step", "kind": "wrong_fact", "said": "Straighten your right knee",
                            "wrong": "Left knee was off", "fix": "Use the joint from FACTS",
                            "facts": {"move": "Cross step"}}, path)
    assert e["id"] == "c001"
    log_correction.add({"move": "Cross step", "kind": "bad_wording", "said": "x", "wrong": "y", "fix": "z"}, path)
    rows = log_correction.load(path)
    assert [r["id"] for r in rows] == ["c001", "c002"]
    assert json.loads(path.read_text().splitlines()[0])["facts"] == {"move": "Cross step"}


def test_correction_log_validates(tmp_path):
    with pytest.raises(ValueError):
        log_correction.add({"move": "Cross step", "kind": "nope", "said": "a", "wrong": "b", "fix": "c"}, tmp_path / "c.jsonl")
    with pytest.raises(ValueError):
        log_correction.add({"move": "Cross step", "kind": "wrong_fact", "said": "", "wrong": "b", "fix": "c"}, tmp_path / "c.jsonl")
