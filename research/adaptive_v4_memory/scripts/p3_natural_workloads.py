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


class TextEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


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


def render_chat_split_last_user(
    tokenizer: ChatTokenizer, messages: list[dict[str, str]]
) -> tuple[str, str]:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Split chat rendering requires a final user message.")
    payload = deepcopy(messages)
    content = payload[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("Final user message content must be text.")
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    separator = f"<adaptive-v4-memory-query-{digest}>"
    if any(separator in str(message.get("content", "")) for message in messages):
        raise ValueError("Chat split separator collides with message content.")
    payload[-1]["content"] = separator + content
    rendered = render_chat(tokenizer, payload)
    parts = rendered.split(separator)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Chat template did not preserve the final-query separator exactly once.")
    return parts[0], parts[1]


def render_chat_split_generation_suffix(
    tokenizer: ChatTokenizer, messages: list[dict[str, str]]
) -> tuple[str, str]:
    """Split after all user content while preserving the exact chat template.

    This lets a press consume the complete benchmark prompt during prefill; the
    second segment contains only the stable boundary retreat plus the assistant
    generation suffix emitted by the tokenizer template.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Generation-suffix rendering requires a final user message.")
    payload = deepcopy(messages)
    content = payload[-1].get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Final user message content must be non-empty text.")
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    separator = f"<adaptive-v4-memory-generation-{digest}>"
    if any(separator in str(message.get("content", "")) for message in messages):
        raise ValueError("Generation suffix separator collides with message content.")
    payload[-1]["content"] = content + separator
    rendered = render_chat(tokenizer, payload)
    parts = rendered.split(separator)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Chat template did not preserve the generation separator exactly once.")
    return parts[0], parts[1]


def render_chat_split_user_content(
    tokenizer: Any,
    context: str,
    query: str,
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, str]:
    if not context or not query:
        raise ValueError("Chat context and query segments must both be non-empty.")
    digest = hashlib.sha256((context + "\0" + query).encode()).hexdigest()
    separator = f"<adaptive-v4-memory-segment-{digest}>"
    if separator in context or separator in query:
        raise ValueError("Chat content separator collides with prompt text.")
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator + query}],
        **template_kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer returned an empty or non-text chat prompt.")
    parts = rendered.split(separator)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Chat template did not preserve the content separator exactly once.")
    return parts[0], parts[1]


def render_chat_split_system_context_query(
    tokenizer: Any,
    system_prefix: str,
    context: str,
    query: str,
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, str, str]:
    """Render one system message and split the exact user context/query text.

    The first returned segment contains the complete rendered system message and
    the user-message header.  It is therefore the causal protected-prefix span,
    rather than merely a string embedded in an untrusted user document.
    """
    if not system_prefix or not context or not query:
        raise ValueError("System prefix, context, and query must all be non-empty.")
    digest = hashlib.sha256(
        (system_prefix + "\0" + context + "\0" + query).encode()
    ).hexdigest()
    context_separator = f"<adaptive-v4-memory-context-{digest}>"
    query_separator = f"<adaptive-v4-memory-query-{digest}>"
    if any(
        separator in value
        for separator in (context_separator, query_separator)
        for value in (system_prefix, context, query)
    ):
        raise ValueError("System safety separator collides with prompt text.")
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prefix},
            {
                "role": "user",
                "content": context_separator + context + query_separator + query,
            },
        ],
        **template_kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer returned an empty or non-text chat prompt.")
    prefix_parts = rendered.split(context_separator)
    if len(prefix_parts) != 2 or not prefix_parts[0] or not prefix_parts[1]:
        raise ValueError("Chat template did not preserve the context separator exactly once.")
    query_parts = prefix_parts[1].split(query_separator)
    if len(query_parts) != 2 or not query_parts[0] or not query_parts[1]:
        raise ValueError("Chat template did not preserve the query separator exactly once.")
    return prefix_parts[0], query_parts[0], query_parts[1]


def encode_rendered_system_context_query_exact(
    tokenizer: Any,
    rendered_system_prefix: str,
    rendered_context: str,
    rendered_query: str,
) -> tuple[Any, Any, int, int, int]:
    """Tokenize once and return context/query plus an exact protected boundary.

    Both string boundaries can cross a BPE merge.  The returned protected length
    and context/query retreat are based only on stable prefixes of the one full
    tokenization, so every protected position indexes the cache actually used.
    """
    rendered_prefix_and_context = rendered_system_prefix + rendered_context
    context_ids, query_ids, query_retreat = encode_rendered_segments_exact(
        tokenizer, rendered_prefix_and_context, rendered_query
    )
    full_ids = tokenizer.encode(
        rendered_prefix_and_context + rendered_query,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prefix_only_ids = tokenizer.encode(
        rendered_system_prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if getattr(prefix_only_ids, "ndim", None) != 2 or prefix_only_ids.shape[0] != 1:
        raise ValueError("Tokenizer must return a rank-two system-prefix tensor.")
    protected_length = 0
    maximum = min(int(full_ids.shape[1]), int(prefix_only_ids.shape[1]))
    while (
        protected_length < maximum
        and int(full_ids[0, protected_length])
        == int(prefix_only_ids[0, protected_length])
    ):
        protected_length += 1
    if protected_length <= 0 or protected_length > int(context_ids.shape[1]):
        raise ValueError("Could not form a non-empty protected system-prefix token span.")
    protected_retreat = int(prefix_only_ids.shape[1]) - protected_length
    return (
        context_ids,
        query_ids,
        protected_length,
        protected_retreat,
        query_retreat,
    )


def encode_rendered_segments_exact(
    tokenizer: Any, rendered_context: str, rendered_query: str
) -> tuple[Any, Any, int]:
    """Return context/query slices of one exact full-prompt tokenization.

    Independent BPE tokenization can merge across the string boundary.  We
    therefore split the full token tensor at its stable prefix with the
    context-only tokenization and report any boundary retreat explicitly.
    """
    if not rendered_context or not rendered_query:
        raise ValueError("Rendered context and query must both be non-empty.")
    full_ids = tokenizer.encode(
        rendered_context + rendered_query,
        return_tensors="pt",
        add_special_tokens=False,
    )
    context_only_ids = tokenizer.encode(
        rendered_context,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if getattr(full_ids, "ndim", None) != 2 or getattr(context_only_ids, "ndim", None) != 2:
        raise ValueError("Tokenizer must return rank-two token tensors.")
    if full_ids.shape[0] != 1 or context_only_ids.shape[0] != 1:
        raise ValueError("Natural prompt tokenization requires batch size one.")

    stable = 0
    maximum = min(int(full_ids.shape[1]), int(context_only_ids.shape[1]))
    while stable < maximum and int(full_ids[0, stable]) == int(context_only_ids[0, stable]):
        stable += 1
    if stable <= 0 or stable >= int(full_ids.shape[1]):
        raise ValueError("Could not form non-empty exact context/query token slices.")
    retreat = int(context_only_ids.shape[1]) - stable
    return full_ids[:, :stable], full_ids[:, stable:], retreat


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


def build_longbench_v2_segments(row: dict[str, Any], template: str) -> tuple[str, str]:
    if template.count("$DOC$") != 1:
        raise ValueError("LongBench v2 template must contain one document placeholder.")
    before, after = template.split("$DOC$")
    context = row.get("context")
    if not isinstance(context, str):
        raise ValueError("LongBench v2 field context must be text.")
    query = build_longbench_v2_prompt({**row, "context": ""}, after)
    context_segment = before + context.strip()
    if context_segment + query != build_longbench_v2_prompt(row, template):
        raise ValueError("LongBench v2 segmented prompt does not reconstruct exactly.")
    return context_segment, query


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
            f"\n### Session {index}:\nSession Date: {date}\nSession Content:\n{session_text}\n"
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


def mrcr_official_token_count(
    row: dict[str, Any], messages: list[dict[str, str]], encoder: TextEncoder
) -> int:
    answer = row.get("answer")
    if not isinstance(answer, str):
        raise ValueError("MRCR answer must be text.")
    return sum(len(encoder.encode(message["content"])) for message in messages) + len(
        encoder.encode(answer)
    )


def mrcr_bin_index(total_tokens: int, boundaries: list[list[int]]) -> int:
    if total_tokens < 0:
        raise ValueError("MRCR token count must be non-negative.")
    for index, boundary in enumerate(boundaries):
        if len(boundary) != 2:
            raise ValueError("Every MRCR bin boundary must contain two integers.")
        lower, upper = boundary
        if not isinstance(lower, int) or not isinstance(upper, int) or lower > upper:
            raise ValueError("MRCR bin boundary is invalid.")
        if lower <= total_tokens <= upper:
            return index
    raise ValueError(f"MRCR example with {total_tokens} tokens is outside all frozen bins.")


def select_mrcr_primary_rows(
    *,
    rows: list[dict[str, Any]],
    encoder: TextEncoder,
    boundaries: list[list[int]],
    primary_bins: int,
    samples_per_bin: int,
) -> list[dict[str, Any]]:
    if primary_bins <= 0 or primary_bins > len(boundaries) or samples_per_bin <= 0:
        raise ValueError("MRCR primary-bin selection contract is invalid.")
    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in range(primary_bins)}
    for row in rows:
        messages = parse_mrcr_messages(row)
        count = mrcr_official_token_count(row, messages, encoder)
        index = mrcr_bin_index(count, boundaries)
        if index < primary_bins:
            grouped[index].append(
                {
                    **row,
                    "official_o200k_prompt_plus_answer_tokens": count,
                    "official_bin_index": index,
                    "messages": messages,
                }
            )
    observed = {index: len(grouped[index]) for index in grouped}
    if any(count != samples_per_bin for count in observed.values()):
        raise ValueError(
            f"MRCR primary bins do not contain exactly {samples_per_bin} samples: {observed}."
        )
    return [row for index in range(primary_bins) for row in grouped[index]]


def mrcr_generation_reserve(row: dict[str, Any], model_tokenizer: TextEncoder) -> int:
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer:
        raise ValueError("MRCR answer must be non-empty text.")
    return len(model_tokenizer.encode(answer)) + 32


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
    if (
        not isinstance(prompts, list)
        or not isinstance(answers, list)
        or not isinstance(turns, list)
    ):
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
