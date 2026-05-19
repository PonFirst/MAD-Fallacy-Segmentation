import os
import sys
import json
import re
import time
import random
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_dataset, get_train_test_split

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FALLACY_CLASSES = [
    "AppealtoEmotion",
    "AdHominem",
    "AppealtoAuthority",
    "FalseCause",
    "Slipperyslope",
    "Slogans",
]

# Fixed seed ensures the example selection is fully reproducible across runs.
STRATIFIED_SEED = 42

# Reasoning templates used only inside in-context examples (never for predictions).
_REASONING_TEMPLATES = {
    "AppealtoEmotion": (
        "This span uses emotionally charged language to exploit the audience's "
        "instinct rather than making a rational argument."
    ),
    "AdHominem": (
        "This span attacks the speaker's character or personal qualities rather "
        "than addressing the substance of their argument."
    ),
    "AppealtoAuthority": (
        "This span invokes an authority figure or group consensus as a substitute "
        "for evidence, without establishing the authority's relevance."
    ),
    "FalseCause": (
        "This span implies a causal relationship between events based on correlation "
        "or temporal sequence rather than established causation."
    ),
    "Slipperyslope": (
        "This span implies an improbable chain of consequences will inevitably "
        "follow from an action without justifying each step."
    ),
    "Slogans": (
        "This brief striking phrase is designed to provoke excitement rather than "
        "advance a substantive argument."
    ),
}

BASE_SYSTEM_PROMPT = """You are an expert in logical fallacy detection and argumentation theory.
Your job is to analyze political debate dialogues and identify ALL logical fallacies present.

Fallacies are defined following Walton (1987) and Da San Martino et al. (2019):

- AdHominem: an excessive attack on the arguer's position, character, or personal
  qualities rather than addressing the substance of their argument
- AppealtoAuthority: relying on the endorsement of an authority figure or group
  consensus without providing sufficient evidence; may involve non-experts or
  majority opinion
- AppealtoEmotion: the unessential loading of the argument with emotional language
  to exploit the audience's emotional instinct rather than appeal to reason;
  spans may be as short as a single emotionally loaded word or as long as a sentence
- FalseCause: misinterpreting the correlation of two events as causation
- Slipperyslope: implying that an improbable or exaggerated consequence will
  inevitably result from a particular action
- Slogans: a brief and striking phrase used to provoke excitement in the audience,
  often accompanied by argument by repetition; by definition short phrases

You must respond ONLY with a valid JSON array. No text before or after. No markdown. No backticks.

[
  {
    "suspected_fallacy": "one of the six fallacy names above",
    "span_text": "the exact verbatim words from the dialogue containing the fallacy",
    "reasoning": "one or two sentences explaining why this is a fallacy"
  }
]

Rules:
- Find ALL fallacies present in the dialogue
- span_text must be copied verbatim from the dialogue — do not paraphrase or summarize
- span_text should be the minimal text that contains the fallacy — spans range
  from single words to full sentences depending on where the fallacy is concentrated
- suspected_fallacy must be exactly one of: AdHominem, AppealtoAuthority,
  AppealtoEmotion, FalseCause, Slipperyslope, Slogans
- if no fallacy is found, return an empty array: []
- return between 1 and 10 fallacies
- only include predictions you are confident about — do not pad with weak detections"""


# ---------------------------------------------------------------------------
# Example selection — stratified, one clean dialogue per fallacy class
# ---------------------------------------------------------------------------

def select_stratified_examples(train_df, seed=STRATIFIED_SEED):
    """
    Select one representative training dialogue per fallacy class.

    Selection priority:
      1. Dialogues where the target fallacy is the ONLY class present
         (clean, unambiguous signal for the model).
      2. If no single-class dialogue exists, fall back to any dialogue
         containing the target fallacy.

    The random.Random instance with a fixed seed makes the selection fully
    deterministic and reproducible — seed=42 is reported in the paper.

    Returns a list of (dialogue: str, annotations: list[dict]) tuples,
    one per entry in FALLACY_CLASSES (6 total).
    """
    rng = random.Random(seed)
    examples = []

    for fallacy in FALLACY_CLASSES:
        # All train dialogues that contain at least one annotation of this type
        candidates = [
            row
            for _, row in train_df.iterrows()
            if any(a["fallacy"] == fallacy for a in row["annotations"])
        ]
        if not candidates:
            raise ValueError(
                f"No training dialogue found for fallacy '{fallacy}'. "
                "Check that the train split contains all six classes."
            )

        # Prefer dialogues that contain ONLY this fallacy class (clean examples)
        clean = [
            row for row in candidates
            if len({a["fallacy"] for a in row["annotations"]}) == 1
        ]
        pool = clean if clean else candidates

        chosen = rng.choice(pool)
        examples.append((chosen["dialogue"], chosen["annotations"]))

    return examples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_example_block(idx, dialogue, annotations):
    """Format one labelled training dialogue as a numbered EXAMPLE block."""
    gold_json = [
        {
            "suspected_fallacy": ann["fallacy"],
            "span_text": ann["snippet"],
            "reasoning": _REASONING_TEMPLATES.get(
                ann["fallacy"], "This is a logical fallacy."
            ),
        }
        for ann in annotations
    ]
    return (
        f"EXAMPLE {idx}:\n"
        f"DIALOGUE:\n{dialogue}\n\n"
        f"FALLACIES:\n{json.dumps(gold_json, indent=2)}"
    )


def build_few_shot_user_message(examples, test_dialogue):
    """
    Construct the full user message: labelled examples followed by the
    unlabelled test dialogue.
    """
    blocks = [
        build_example_block(i + 1, dialogue, annotations)
        for i, (dialogue, annotations) in enumerate(examples)
    ]
    examples_section = "\n\n---\n\n".join(blocks)
    return (
        f"Below are {len(examples)} annotated examples, one per fallacy class. "
        f"Study them carefully, then analyze the final dialogue in the same format.\n\n"
        f"{examples_section}\n\n"
        f"---\n\n"
        f"Now analyze this dialogue and identify ALL logical fallacies:\n\n"
        f"DIALOGUE:\n{test_dialogue}"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_few_shot(dialogue: str, examples: list) -> list:
    """
    Call the model with the few-shot prompt and return a parsed prediction list.

    On JSON parse failure, returns a single-element list with an error dict
    so the main loop can log it without crashing.
    On API failure, retries up to 3 times with exponential back-off.
    """
    user_message = build_few_shot_user_message(examples, dialogue)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()

            if response.choices[0].finish_reason == "length":
                print(f"  WARNING: output truncated at {len(raw)} chars")

            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        except json.JSONDecodeError as e:
            # Do not retry — the model returned something; it just isn't valid JSON.
            return [{"error": "json_parse_error", "raw_response": raw, "details": str(e)}]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [{"error": "api_error", "details": str(e)}]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results_list, path="results/few_shot_full.jsonl"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        for record in results_list:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(results_list)} records → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_dataset()
    train, test = get_train_test_split(df)

    # --- Example selection ---
    examples = select_stratified_examples(train, seed=STRATIFIED_SEED)

    print(f"\nFew-shot examples (stratified, seed={STRATIFIED_SEED}):")
    for i, (dialogue, anns) in enumerate(examples):
        labels = [a["fallacy"] for a in anns]
        n_words = len(dialogue.split())
        print(f"  Example {i + 1}: {n_words:>5} words | classes={labels}")

    # Sanity check: all six classes represented exactly once
    selected_classes = [a["fallacy"] for _, anns in examples for a in anns]
    assert set(selected_classes) == set(FALLACY_CLASSES), (
        f"Not all fallacy classes covered: {set(FALLACY_CLASSES) - set(selected_classes)}"
    )
    print(f"  ✓ All {len(FALLACY_CLASSES)} fallacy classes represented\n")

    # --- Main evaluation loop ---
    total = len(test)
    print(f"Running Few-Shot baseline on {total} test dialogues...")
    print(f"Estimated time: {total * 30 / 60:.1f} min\n")

    all_results = []

    for i in range(total):
        sample = test.iloc[i]
        dialogue = sample["dialogue"]
        gold_annotations = sample["annotations"]

        print(
            f"[{i + 1:>2}/{total}] {len(dialogue.split()):>5} words | "
            f"gold={len(gold_annotations)}"
        )

        predictions = run_few_shot(dialogue, examples)

        record = {
            "dialogue_id": int(sample.name),   # original DataFrame index
            "dialogue": dialogue,
            "gold_annotations": gold_annotations,
            "predictions": predictions,
        }
        all_results.append(record)

        if predictions and "error" in predictions[0]:
            print(f"  ERROR: {predictions[0]['error']} — {predictions[0].get('details', '')}")
        else:
            pred_labels = [p.get("suspected_fallacy", "?") for p in predictions]
            print(f"  Predicted {len(predictions)} fallacies: {pred_labels}")

        # Incremental save every 10 dialogues
        if (i + 1) % 10 == 0:
            save_results(all_results, "results/few_shot_full.jsonl")

        if i < total - 1:
            time.sleep(30)

    save_results(all_results, "results/few_shot_full.jsonl")
    print(f"\nDone. {total} dialogues evaluated.")