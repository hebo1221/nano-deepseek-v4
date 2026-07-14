from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from p3_natural_workloads import (
    encode_rendered_segments_exact,
    render_chat_split_generation_suffix,
)


@dataclass(frozen=True)
class NaturalSafetyCase:
    example_id: str
    source_id: int
    prompt_position: str
    prompt: str
    metadata: dict[str, Any]


def longsafety_prompt(row: dict[str, Any], position: str) -> str:
    instruction, context = row.get("instruction"), row.get("context")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("LongSafety instruction must be non-empty text.")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("LongSafety context must be non-empty text.")
    if position == "front":
        return f"Based on the following long context, {instruction}\n\n{context}"
    if position == "end":
        return f"{context}\n\nBased on the long context above, {instruction}"
    raise ValueError(f"Unsupported LongSafety prompt position: {position}.")


def generation_cases(
    benchmark: str, rows: list[dict[str, Any]]
) -> list[NaturalSafetyCase]:
    cases: list[NaturalSafetyCase] = []
    if benchmark == "LongSafety":
        for row in rows:
            identifier = row.get("id")
            if not isinstance(identifier, int):
                raise ValueError("LongSafety id must be an integer.")
            for position in ("front", "end"):
                cases.append(
                    NaturalSafetyCase(
                        example_id=f"longsafety:{identifier}:{position}",
                        source_id=identifier,
                        prompt_position=position,
                        prompt=longsafety_prompt(row, position),
                        metadata={
                            "safety_type": row.get("safety_type"),
                            "task_type": row.get("task_type"),
                            "source_word_length": row.get("length"),
                            "source_doc_count": row.get("doc_num"),
                        },
                    )
                )
    elif benchmark == "IFEval":
        for row in rows:
            key, prompt = row.get("key"), row.get("prompt")
            if not isinstance(key, int) or not isinstance(prompt, str) or not prompt:
                raise ValueError("IFEval key and prompt are invalid.")
            cases.append(
                NaturalSafetyCase(
                    example_id=f"ifeval:{key}",
                    source_id=key,
                    prompt_position="official-short",
                    prompt=prompt,
                    metadata={
                        "instruction_id_list": row.get("instruction_id_list"),
                        "kwargs": row.get("kwargs"),
                    },
                )
            )
    else:
        raise ValueError(f"Unsupported natural safety benchmark: {benchmark}.")
    identities = [case.example_id for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Natural safety generation case identities are duplicated.")
    return cases


def rendered_case(tokenizer: Any, case: NaturalSafetyCase) -> dict[str, Any]:
    rendered_prompt, rendered_suffix = render_chat_split_generation_suffix(
        tokenizer, [{"role": "user", "content": case.prompt}]
    )
    context_ids, suffix_ids, boundary_retreat = encode_rendered_segments_exact(
        tokenizer, rendered_prompt, rendered_suffix
    )
    return {
        "context_ids": context_ids,
        "question_ids": suffix_ids,
        "exact_input_tokens": int(context_ids.shape[1] + suffix_ids.shape[1]),
        "raw_prompt_sha256": hashlib.sha256(
            (rendered_prompt + rendered_suffix).encode()
        ).hexdigest(),
        "token_boundary_retreat": boundary_retreat,
    }
