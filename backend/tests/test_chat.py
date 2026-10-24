"""The dancer's own questions: spend limits, free answers, the model path and the guard."""

import asyncio
import json
import threading
from datetime import date

from fastapi.testclient import TestClient

from api import server
from api.server import CoachingSession, app
from app.chat import CoachChat, chat_facts, quick_answer, summary_line
from app.coach import AICoach
from app.grounding import ungrounded_claims
from app.orchestrator import VoiceOrchestrator
from app.turn_gate import TokenBudget, TurnGate, screen_text


STATS = {"reps": 5, "clean_reps": 3, "clean_streak": 2, "best_streak": 2,
         "focus_joint": "left elbow", "focus_fix": "straighten", "total_points": 900}
SUMMARY = {"type": "rep_summary", "move": "Cross step", "reps": 3, "clean_reps": 2, "best_streak": 2,
           "went_well": ["right knee"],
           "fix": {"joint": "left elbow", "fix": "straighten", "times_off": 1,
                   "checkpoint": 2, "of": 4, "t": 1712.53, "x": 0.41, "y": 0.37}}


def facts():
    return chat_facts("Cross step", STATS, SUMMARY)


class FakeLLM:
    def __init__(self, reply="Nice work, straighten that left elbow next round.", t_in=300, t_out=20):
        self.reply, self.t_in, self.t_out = reply, t_in, t_out
        self.calls = []

    async def __call__(self, system, messages, max_tokens, timeout):
        self.calls.append(messages)
        if self.reply is None:
            return None
        return self.reply, self.t_in, self.t_out


def run(coro):
    return asyncio.run(coro)


# ---- facts --------------------------------------------------------------

def test_facts_leave_out_replay_times_and_positions():
    f = facts()
    fix = f["last_attempt"]["fix"]
    assert "t" not in fix and "x" not in fix and "y" not in fix
    assert f["session"] == {"reps": 5, "clean_reps": 3, "clean_streak": 2, "best_streak": 2}
    assert "total_points" not in json.dumps(f)


def test_facts_with_nothing_scored_yet():
    f = chat_facts("", None, None)
    assert f["move"] == "this move" and f["session"]["reps"] == 0
    assert "last_attempt" not in f and "focus" not in f


# ---- screen and gate ------------------------------------------------------

def test_screen_drops_empty_and_wordless_text():
    assert screen_text("") == "empty"
    assert screen_text("   ...  ") == "empty"
    assert screen_text("123 ???") == "no_words"
    assert screen_text("how was that") is None
    assert screen_text("How was that?", "how was that") == "repeat"


def test_rate_limit_counts_every_turn_in_the_last_minute():
    now = [0.0]
    gate = TurnGate(per_minute=2, per_session=10, clock=lambda: now[0])
    assert gate.admit_turn() is None
    assert gate.admit_turn() is None
    assert gate.admit_turn() == "rate"
    now[0] = 61.0
    assert gate.admit_turn() is None


def test_session_cap_counts_only_model_calls():
    gate = TurnGate(per_minute=100, per_session=2)
    assert gate.admit_model() is None
    assert gate.admit_model() is None
    assert gate.admit_model() == "session_cap"


def test_budget_fails_closed_and_resets_next_day():
    day = [date(2026, 10, 23)]
    budget = TokenBudget(daily=1000, today=lambda: day[0])
    assert budget.can_spend(900)
    budget.spend(900)
    assert not budget.can_spend(200)
    day[0] = date(2026, 10, 24)
    assert budget.can_spend(1000)


# ---- free answers ------------------------------------------------------------

def test_hello_gets_a_reply_without_the_model():
    llm = FakeLLM()
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=TokenBudget(daily=10_000))
    reply = run(chat.answer("hello", facts()))
    assert reply.source == "quick" and reply.text.startswith("Hey")
    assert llm.calls == []


def test_common_questions_are_answered_from_the_facts():
    f = facts()
    assert "3 reps, 2 clean" in quick_answer("how did I do?", f)
    assert "left elbow" in quick_answer("what should I fix", f)
    assert "right knee" in quick_answer("what went well?", f)
    assert "2" in quick_answer("what's my streak", f)
    assert quick_answer("tell me about the history of breaking", f) is None


def test_free_answers_pass_the_grounding_guard():
    f = facts()
    for q in ("hi", "how did I do", "what should I fix", "what went well", "streak", "help", "thanks", "show me"):
        text = quick_answer(q, f)
        assert text and ungrounded_claims(text, f) == [], (q, text)
    empty = chat_facts("", None, None)
    for q in ("hi", "how did I do", "what should I fix", "what went well", "streak"):
        text = quick_answer(q, empty)
        assert text and ungrounded_claims(text, empty) == [], (q, text)


def test_twist_fix_reads_naturally_and_is_grounded():
    f = chat_facts("Cross step", {"reps": 2, "focus_joint": "hips", "focus_fix": "twist"}, None)
    line = summary_line(f)
    assert "hips" in line and "twist" in line
    assert ungrounded_claims(line, f) == []


def test_same_question_same_facts_is_answered_again_for_free():
    llm = FakeLLM()
    chat = CoachChat(llm=llm, gate=TurnGate(per_minute=2), budget=TokenBudget(daily=10_000))
    first = run(chat.answer("why does a straight elbow matter", facts()))
    again = run(chat.answer("Why does a straight elbow matter?", facts()))
    assert first.source == "model" and again.source == "cache"
    assert again.text == first.text and len(llm.calls) == 1


# ---- the model path ------------------------------------------------------------

def test_open_question_goes_to_the_model_with_facts_and_counts_tokens():
    llm = FakeLLM(t_in=310, t_out=22)
    budget = TokenBudget(daily=10_000)
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=budget)
    reply = run(chat.answer("why does the elbow matter here", facts()))
    assert reply.source == "model" and reply.tokens_in == 310 and reply.tokens_out == 22
    assert budget.spent == 332
    sent = llm.calls[0][-1]["content"]
    assert "FACTS" in sent and "left elbow" in sent


def test_history_keeps_only_the_last_turns():
    llm = FakeLLM()
    chat = CoachChat(llm=llm, gate=TurnGate(per_minute=100), budget=TokenBudget(daily=1_000_000))
    for i in range(7):
        run(chat.answer(f"question number {'abcdefg'[i]} about the elbow", facts()))
    # 4 earlier pairs plus the new question
    assert len(llm.calls[-1]) == 9


def test_ungrounded_model_answer_is_replaced():
    llm = FakeLLM(reply="Your right shoulder dropped 15 degrees.")
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=TokenBudget(daily=10_000))
    reply = run(chat.answer("anything about my shoulder", facts()))
    assert reply.source == "fallback" and reply.reason == "guard"
    assert "shoulder" not in reply.text and chat.refused == 1


def test_long_model_answer_is_replaced():
    llm = FakeLLM(reply="word " * 60)
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=TokenBudget(daily=10_000))
    assert run(chat.answer("talk to me about practice", facts())).source == "fallback"


def test_failed_call_uses_a_template_and_still_counts_the_estimate():
    llm = FakeLLM(reply=None)
    budget = TokenBudget(daily=10_000)
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=budget)
    reply = run(chat.answer("talk to me about practice", facts()))
    assert reply.source == "fallback" and reply.reason == "llm_failed"
    assert budget.spent > 0


def test_no_model_call_once_the_daily_budget_is_spent():
    llm = FakeLLM()
    budget = TokenBudget(daily=100)
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=budget)
    reply = run(chat.answer("talk to me about practice", facts()))
    assert reply.source == "limit" and reply.reason == "daily_budget"
    assert llm.calls == []
    assert "3 reps, 2 clean" in reply.text


def test_session_cap_stops_model_calls_but_free_answers_still_work():
    llm = FakeLLM()
    chat = CoachChat(llm=llm, gate=TurnGate(per_minute=100, per_session=1), budget=TokenBudget(daily=10_000))
    assert run(chat.answer("talk to me about practice", facts())).source == "model"
    capped = run(chat.answer("tell me something else about practice", facts()))
    assert capped.source == "limit" and capped.reason == "session_cap"
    assert run(chat.answer("what should I fix", facts())).source == "quick"
    assert len(llm.calls) == 1


def test_rate_limit_answers_with_a_template():
    chat = CoachChat(llm=FakeLLM(), gate=TurnGate(per_minute=1), budget=TokenBudget(daily=10_000))
    run(chat.answer("hi", facts()))
    reply = run(chat.answer("what should I fix", facts()))
    assert reply.source == "limit" and reply.reason == "rate"


def test_long_typed_text_is_cut_before_the_model():
    llm = FakeLLM()
    chat = CoachChat(llm=llm, gate=TurnGate(), budget=TokenBudget(daily=100_000))
    run(chat.answer("explain " + "x" * 5000, facts()))
    assert len(llm.calls[0][-1]["content"]) < 1500


def test_without_an_api_key_open_questions_get_a_template():
    chat = CoachChat(gate=TurnGate(), budget=TokenBudget(daily=10_000))
    chat._available = False
    reply = run(chat.answer("talk to me about practice", facts()))
    assert reply.source == "fallback" and reply.reason == "no_api_key"


# ---- orchestrator ------------------------------------------------------------

def make_orchestrator(llm=None):
    sent, spoken = [], []

    async def emit(msg):
        sent.append(msg)

    async def speak(text):
        spoken.append(text)

    chat = CoachChat(llm=llm or FakeLLM(), gate=TurnGate(per_minute=100), budget=TokenBudget(daily=100_000))
    return VoiceOrchestrator(emit, speak, facts, chat=chat), sent, spoken


def test_a_turn_sends_transcript_states_and_speaks_the_reply():
    orch, sent, spoken = make_orchestrator()
    reply = run(orch.handle_text("hello"))
    assert reply.source == "quick" and spoken == [reply.text]
    assert sent[0] == {"type": "transcript", "text": "hello", "source": "text"}
    states = [m["state"] for m in sent if m["type"] == "voice_state"]
    assert states == ["thinking", "speaking", "idle"]
    assert not orch.busy


def test_a_second_turn_while_busy_is_refused():
    class SlowLLM(FakeLLM):
        async def __call__(self, *a):
            await asyncio.sleep(0.05)
            return await super().__call__(*a)

    orch, sent, spoken = make_orchestrator(SlowLLM())

    async def both():
        return await asyncio.gather(
            orch.handle_text("talk to me about practice"),
            orch.handle_text("and another thing about practice"),
        )
    first, second = run(both())
    assert first is not None and second is None
    assert len(spoken) == 1
    assert any(m.get("busy") for m in sent)


# ---- over the socket ------------------------------------------------------------

def make_session():
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.voice = None
    s.ai_coach = AICoach(llm=lambda *a: None)
    s.speak_lock = asyncio.Lock()
    s.coach_tasks = set()
    s.last_stats, s.last_summary, s.last_move_name = STATS, SUMMARY, "Cross step"
    s.chat = CoachChat(llm=FakeLLM(), gate=TurnGate(), budget=TokenBudget(daily=10_000))
    return s


def test_socket_chat_hello_gets_a_spoken_reply():
    server._sessions["sess-chat"] = make_session()
    try:
        with TestClient(app).websocket_connect("/ws/sess-chat") as ws:
            ws.receive_text()   # voice status
            ws.send_text(json.dumps({"type": "chat", "text": "hello"}))
            msgs = [json.loads(ws.receive_text()) for _ in range(6)]
        types = [m["type"] for m in msgs]
        assert types[0] == "transcript"
        cue = next(m for m in msgs if m["type"] == "coach_cue")
        assert cue["mode"] == "reply" and cue["cue"].startswith("Hey")
        assert {"type": "voice_state", "state": "idle"} in msgs
    finally:
        server._sessions.pop("sess-chat", None)


def test_socket_ignores_chat_without_text():
    server._sessions["sess-chat2"] = make_session()
    try:
        with TestClient(app).websocket_connect("/ws/sess-chat2") as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "chat", "text": 5}))
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text()) == {"type": "pong"}
    finally:
        server._sessions.pop("sess-chat2", None)
