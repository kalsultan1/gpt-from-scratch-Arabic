from typing import Dict, List


GENERIC_OPENINGS = [
    "في يوم من الأيام",
    "كان يا ما كان",
    "في ليلة",
    "في صباح",
    "ذات يوم"
]


def detect_failure_modes(text: str) -> Dict[str, bool]:
    """
    Detect common failure modes in generated Arabic text.
    """
    stripped = text.strip()
    words = stripped.split()

    arabic_chars = sum(1 for ch in stripped if "\u0600" <= ch <= "\u06FF")
    weird_chars = sum(1 for ch in stripped if ch in ["⁇", "�"])

    repeated_word = False
    if len(words) >= 4:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                repeated_word = True
                break

    generic_opening = any(stripped.startswith(opening) for opening in GENERIC_OPENINGS)

    failure_modes = {
        "too_short": len(words) < 10,
        "contains_weird_symbols": weird_chars > 0,
        "low_arabic_content": arabic_chars < max(10, len(stripped) * 0.2),
        "word_repetition": repeated_word,
        "generic_opening": generic_opening,
        "possible_context_loss": len(words) > 0 and stripped.endswith(("...", "؟؟", "!!")),
    }

    return failure_modes


def summarize_failure_modes(samples: List[Dict]) -> Dict[str, int]:
    """
    samples format:
    [
        {"prompt": "...", "output": "..."},
        ...
    ]
    """
    summary = {
        "too_short": 0,
        "contains_weird_symbols": 0,
        "low_arabic_content": 0,
        "word_repetition": 0,
        "generic_opening": 0,
        "possible_context_loss": 0,
    }

    for sample in samples:
        result = detect_failure_modes(sample["output"])
        for key, value in result.items():
            if value:
                summary[key] += 1

    return summary


def format_error_report(samples: List[Dict]) -> str:
    """
    Return a readable text report for README/demo use.
    """
    summary = summarize_failure_modes(samples)

    lines = []
    lines.append("Error Analysis Report")
    lines.append("====================")
    lines.append(f"Total samples: {len(samples)}")
    lines.append("")

    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Sample-level details:")
    lines.append("")

    for i, sample in enumerate(samples, 1):
        failures = detect_failure_modes(sample["output"])
        active = [k for k, v in failures.items() if v]
        if not active:
            active = ["no obvious failure detected"]

        lines.append(f"Sample {i}")
        lines.append(f"Prompt: {sample['prompt']}")
        lines.append(f"Failures: {', '.join(active)}")
        lines.append("")

    return "\n".join(lines)