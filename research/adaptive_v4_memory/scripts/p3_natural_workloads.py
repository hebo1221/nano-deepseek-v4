from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_chat(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer returned an empty or non-text chat prompt.")
    return rendered


def build_longbench_v2_prompt(row: dict[str, Any], template: str) -> str:
    fields = {
        "$DOC$": "context",
        "$Q$": "question",
        "$C_A$": "choice_A",
        "$C_B$": "choice_B",
        "$C_C$": "choice_C",
        "$C_D$": "choice_D",
    }
    prompt = template
    for placeholder, field in fields.items():
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"LongBench v2 field {field} must be text.")
        prompt = prompt.replace(placeholder, value.strip())
    if any(placeholder in prompt for placeholder in fields):
        raise ValueError("LongBench v2 prompt retains an unresolved placeholder.")
    return prompt


def build_longmem_full_history_prompt(row: dict[str, Any]) -> str:
    dates = row.get("haystack_dates")
    sessions = row.get("haystack_sessions")
    if not isinstance(dates, list) or not isinstance(sessions, list) or len(dates) != len(sessions):
        raise ValueError("LongMemEval history dates and sessions must be aligned lists.")

    chunks: list[tuple[str, list[dict[str, Any]]]] = []
    for date, session in zip(dates, sessions, strict=True):
        if not isinstance(date, str) or not isinstance(session, list):
            raise ValueError("LongMemEval contains an invalid dated session.")
        clean_session = deepcopy(session)
        for turn in clean_session:
            if not isinstance(turn, dict):
                raise ValueError("LongMemEval session turns must be objects.")
            turn.pop("has_answer", None)
        chunks.append((date, clean_session))
    chunks.sort(key=lambda item: item[0])

    history = ""
    for index, (date, session) in enumerate(chunks, start=1):
        session_text = ""
        for turn in session:
            role, content = turn.get("role"), turn.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("LongMemEval turn role and content must be text.")
            session_text += f"\n\n{role}: {content.strip()}"
        history += (
            f"\n### Session {index}:\nSession Date: {date}\n"
            f"Session Content:\n{session_text}\n"
        )
    if not history:
        raise ValueError("LongMemEval full-history protocol requires at least one session.")

    question_date, question = row.get("question_date"), row.get("question")
    if not isinstance(question_date, str) or not isinstance(question, str):
        raise ValueError("LongMemEval question and question_date must be text.")
    return (
        "I will give you several history chats between you and a user. Please answer the "
        "question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
        f"{history}\n\nCurrent Date: {question_date}\nQuestion: {question}\nAnswer:"
    )


def parse_mrcr_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt")
    messages: Any = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(messages, list) or not messages:
        raise ValueError("MRCR prompt must be a non-empty JSON message list.")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("MRCR messages must be objects.")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("MRCR messages require a supported role and text content.")
        normalized.append({"role": role, "content": content})
    prefix = row.get("random_string_to_prepend")
    answer = row.get("answer")
    if not isinstance(prefix, str) or not prefix.isalnum():
        raise ValueError("MRCR random_string_to_prepend must be alphanumeric.")
    if not isinstance(answer, str) or not answer.startswith(prefix):
        raise ValueError("MRCR answer must start with its registered prefix.")
    return normalized


def load_official_scbench_module(source_root: Path, expected_sha256: str) -> Any:
    module_path = source_root / "scbench" / "eval_utils.py"
    if sha256(module_path) != expected_sha256:
        raise ValueError("Pinned SCBench eval_utils.py SHA-256 drifted.")

    import transformers

    had_sink_cache = hasattr(transformers, "SinkCache")
    old_sink_cache = getattr(transformers, "SinkCache", None)
    if not had_sink_cache:
        transformers.SinkCache = object
    try:
        spec = importlib.util.spec_from_file_location("adaptive_v4_scbench_eval_utils", module_path)
        if spec is None or spec.loader is None:
            raise ValueError("Could not construct the pinned SCBench module loader.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if not had_sink_cache:
            del transformers.SinkCache
        else:
            transformers.SinkCache = old_sink_cache
    return module


def build_scbench_workload(
    *,
    row: dict[str, Any],
    task: str,
    mode: str,
    tokenizer: ChatTokenizer,
    official_module: Any,
) -> dict[str, Any]:
    if mode == "multi-turn":
        create = official_module.create_multiturn_prompt
        built = create(row, task, tokenizer, True, disable_golden_context=False)
        shared_context = None
    elif mode == "multi-request":
        create = official_module.create_scdq_prompt
        built = create(row, task, tokenizer, True)
        shared_context = built["prompts"][0]
        built = {**built, "prompts": built["prompts"][1:]}
    else:
        raise ValueError(f"Unsupported SCBench mode: {mode}.")

    prompts, answers = built.get("prompts"), built.get("ground_truth")
    turns = row.get("multi_turns")
    if not isinstance(prompts, list) or not isinstance(answers, list) or not isinstance(turns, list):
        raise ValueError("Official SCBench prompt builder returned an invalid structure.")
    if len(prompts) != len(answers) or len(prompts) != len(turns):
        raise ValueError("Official SCBench prompt, answer, and turn counts diverged.")

    max_tokens = official_module.DATA_NAME_TO_MAX_NEW_TOKENS[task]
    subtasks = built.get("task")
    records: list[dict[str, Any]] = []
    for index, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
        subtask = subtasks[index] if isinstance(subtasks, list) else None
        reserve = max_tokens[subtask] if isinstance(max_tokens, dict) else max_tokens
        if not isinstance(prompt, str) or not isinstance(reserve, int) or reserve <= 0:
            raise ValueError("Official SCBench prompt or generation reserve is invalid.")
        records.append(
            {
                "turn_index": index,
                "prompt_segment": prompt,
                "ground_truth": answer,
                "generation_reserve_tokens": reserve,
                "subtask": subtask,
            }
        )
    return {"mode": mode, "task": task, "shared_context": shared_context, "turns": records}
