from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def normalize_zh_answer(value: str) -> str:
    chinese_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》"
        "「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    punctuation = set(string.punctuation + chinese_punctuation)
    without_punctuation = "".join(
        character for character in value.lower() if character not in punctuation
    )
    return "".join(without_punctuation.split())


def token_f1(prediction: list[str], reference: list[str]) -> float:
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def qa_f1_english(prediction: str, references: Iterable[str]) -> float:
    scores = []
    for reference in references:
        predicted = normalize_answer(prediction).split()
        expected = normalize_answer(reference).split()
        scores.append(token_f1(predicted, expected) if predicted and expected else 0.0)
    return max(scores, default=0.0)


def qa_f1_chinese(prediction: str, references: Iterable[str]) -> float:
    scores = []
    for reference in references:
        predicted = list(normalize_zh_answer(prediction))
        expected = list(normalize_zh_answer(reference))
        scores.append(token_f1(predicted, expected) if predicted and expected else 0.0)
    return max(scores, default=0.0)


def first_integer(prediction: str) -> str:
    return next((part for part in re.split("[^0-9]", prediction) if part), "")


def score_choice(prediction: str, labels: list[str]) -> float:
    value = prediction.strip()
    if not value:
        return 0.0
    if value[0] in "ABCD":
        return float(value[0] in labels)
    if value in labels:
        return 1.0
    for character in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        value = value.replace(character, " ")
    value = " ".join(value.split())
    for prefix in ["answer is:", "answer:", "answer is", "option is"]:
        index = value.find(prefix)
        if index < 0:
            continue
        if len(value) < index + len(prefix) + 1:
            return 0.0
        suffix = value[index + len(prefix) + 1 :]
        return float(any(suffix.startswith(label) for label in labels))
    return float(any(word in labels for word in value.split() if word in "ABCD"))


def score_math_find(prediction: str, label: int | float | list[int | float]) -> float:
    expected: int | float = label[0] if isinstance(label, list) else label
    match = re.search(r"\d+\.\d+|\d+", prediction)
    if match is None:
        return 0.0
    observed = match.group(0).strip()
    if isinstance(expected, int):
        return float(int(float(observed)) == expected)
    if isinstance(expected, float):
        return float(float(observed) == expected)
    raise TypeError(f"Unsupported math-find label: {type(expected)}")


def score_substring_all(prediction: str, references: Iterable[str]) -> float:
    values = list(references)
    if not values:
        raise ValueError("SCBench substring-all scorer requires references.")
    return sum(float(reference.lower() in prediction.lower()) for reference in values) / len(values)


def official_ground_truth(task: str, turn: dict[str, Any]) -> Any:
    answer = turn["answer"]
    if task == "scbench_choice_eng":
        options = turn["options"]
        if answer not in options:
            raise ValueError("SCBench choice answer is absent from its options.")
        return [answer, "ABCD"[options.index(answer)]]
    if task == "scbench_qa_eng":
        return [answer]
    return answer


def score_turn(
    *,
    task: str,
    prediction: str,
    ground_truth: Any,
    subtask: str | None = None,
    rouge_lsum: Callable[[str, str], float] | None = None,
) -> float:
    effective = subtask or task
    if effective == "scbench_choice_eng":
        return score_choice(prediction, list(ground_truth))
    if effective == "scbench_qa_eng":
        return float(any(str(item).upper() in prediction.upper() for item in ground_truth))
    if effective == "scbench_qa_chn":
        references = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        return qa_f1_chinese(prediction, [str(item) for item in references])
    if effective in {"scbench_kv", "scbench_prefix_suffix"}:
        return float(str(ground_truth) in prediction)
    if effective == "scbench_mf":
        return score_math_find(prediction, ground_truth)
    if effective == "scbench_many_shot":
        return float(any(str(item).upper() in prediction.upper() for item in ground_truth))
    if effective == "scbench_vt":
        references = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        return score_substring_all(prediction, [str(item) for item in references])
    if effective == "scbench_passkey":
        reference = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return float(str(reference) == first_integer(prediction))
    if effective == "scbench_summary":
        if rouge_lsum is None:
            raise ValueError("SCBench summary scoring requires the pinned ROUGE-Lsum scorer.")
        return float(rouge_lsum(prediction, str(ground_truth)))
    if effective in {"scbench_repoqa", "scbench_repoqa_and_kv"}:
        raise ValueError("RepoQA turns require the repository-aware official scorer.")
    raise ValueError(f"Unsupported SCBench scoring task: {effective}.")
