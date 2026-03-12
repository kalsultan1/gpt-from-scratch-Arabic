import math
from typing import List, Dict

import torch


def compute_loss_and_perplexity(model, dataloader, device):
    """
    Compute average loss and perplexity over a dataloader.
    Assumes each batch is (x, y).
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }


def generate_text(model, sp, prompt, device, max_new_tokens=80, temperature=0.9, top_k=40):
    """
    Generate text from a prompt using the trained GPT model.
    """
    model.eval()

    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    return sp.decode(out[0].tolist())


def simple_instruction_following_score(instruction: str, output: str) -> Dict[str, int]:
    """
    Very simple heuristic score for instruction-following.
    Returns a few binary indicators and a total score.

    This is not perfect, but it is enough for a student project.
    """
    score = 0
    details = {
        "non_empty": 0,
        "reasonable_length": 0,
        "mentions_story_words": 0,
        "looks_like_arabic": 0,
    }

    output = output.strip()

    if len(output) > 0:
        details["non_empty"] = 1
        score += 1

    if len(output.split()) >= 15:
        details["reasonable_length"] = 1
        score += 1

    story_keywords = ["قصة", "طفل", "باب", "قرية", "رحلة", "مفتاح", "بيت", "مدينة", "ليلة", "صباح"]
    if any(word in output for word in story_keywords):
        details["mentions_story_words"] = 1
        score += 1

    arabic_chars = sum(1 for ch in output if "\u0600" <= ch <= "\u06FF")
    if arabic_chars >= max(10, len(output) * 0.2):
        details["looks_like_arabic"] = 1
        score += 1

    details["total_score"] = score
    return details


def repetition_score(text: str, n: int = 3) -> float:
    """
    Compute simple repetition ratio using word n-grams.
    Higher means more repetition.
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)

    if len(ngrams) == 0:
        return 0.0

    repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
    return repetition_ratio


def evaluate_generation_set(samples: List[Dict]) -> List[Dict]:
    """
    samples format:
    [
        {"prompt": "...", "output": "..."},
        ...
    ]
    """
    results = []

    for sample in samples:
        prompt = sample["prompt"]
        output = sample["output"]

        score_info = simple_instruction_following_score(prompt, output)
        rep_score = repetition_score(output, n=3)

        results.append({
            "prompt": prompt,
            "output": output,
            "instruction_score": score_info["total_score"],
            "instruction_details": score_info,
            "repetition_score": rep_score
        })

    return results