# Coaching knowledge and correction log

Why: the coach gets better by being given good reference material and by
learning from its mistakes, not by fine-tuning a model on clips of Jordan
coaching. No video or audio is needed for any of this.

## 1. Per-move knowledge file (EXISTS for Cross step)

`backend/knowledge/<move>.md`, one file per move, loaded by `app/knowledge.py`.
Sections: Key points, Common mistakes, Fixes, Cue wording, Priorities.

- It is loaded as a REFERENCE block into the debrief prompt and the chat prompt
  for that move (matched on the move's display name). Moves without a file are
  unchanged. Live cues do not use it yet (PROPOSED).
- It guides wording and priorities only. It never supplies a number or a fact
  about the dancer's body. Three guards:
  1. The prompt tells the model the block is not measured and that it must not
     name a body part that FACTS does not list.
  2. `knowledge.lint` rejects digits and measurement words; a test runs it over
     every knowledge file.
  3. Replies still pass `grounding.ungrounded_claims` against FACTS, so a reply
     that leans on the reference to name an unmeasured joint is refused and the
     templated line is used.
- The Cross step file is a DRAFT written from general toprock knowledge. Jordan
  should correct it. Add a new move by copying the file and changing `move:`.

## 2. Correction log (EXISTS)

When the coach gets something wrong, log it:

```bash
cd backend
python scripts/log_correction.py --move "Cross step" --kind wrong_fact \
  --said "Straighten your right knee" --wrong "The left knee was the one off" \
  --fix "Name the joint from FACTS only"
python scripts/log_correction.py --list
```

Kinds: `wrong_fact`, `bad_priority`, `bad_wording`, `missed_issue`. Entries go to
`backend/evals/corrections.jsonl` as text only. Pass `--facts '<json>'` with the
facts block when you have it.

How entries are used (by hand for now):
- `wrong_fact` with facts: becomes an ungrounded case in `evals/guard_cases.json`.
- `bad_priority`, `bad_wording`: becomes a line in the move's knowledge file.
- `missed_issue`: becomes a labeled rep (below).

## 3. Labeled reps as test data (PROPOSED)

Record a rep, write down what is actually wrong with it, and save landmarks
only (no video). Use them to check that the scorer and coach name the right
thing. Not built. It needs a phone and a real dancer, so it is a manual step for
Jordan.

## Not built yet
- Promoting corrections to eval cases with a script (kept manual so it does not
  collide with the guard eval changes in flight).
- Knowledge for other moves, and use in live cues.
- Any measured claim about the coach getting better. That needs a benchmark
  script and committed output first.
