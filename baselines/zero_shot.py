import os
import sys
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_dataset, get_train_test_split

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert in logical fallacy detection and argumentation theory.
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


def run_zero_shot(dialogue: str) -> list:
    user_message = f"""Analyze this political debate dialogue and identify ALL logical fallacies:

DIALOGUE:
{dialogue}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        except json.JSONDecodeError as e:
            return [{"error": "json_parse_error", "raw_response": raw, "details": str(e)}]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [{"error": "api_error", "details": str(e)}]


def save_results(results_list, path="results/zero_shot_full.jsonl"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        for record in results_list:
            f.write(json.dumps(record) + "\n")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    df = load_dataset()
    train, test = get_train_test_split(df)

    total = len(test)
    print(f"\nRunning Zero-Shot baseline on all {total} test dialogues...\n")
    print(f"Estimated time: {total * 25 / 60:.1f} minutes\n")

    all_results = []

    for i in range(total):
        sample = test.iloc[i]
        dialogue = sample["dialogue"]
        gold_annotations = sample["annotations"]

        print(f"[{i+1}/{total}] length={len(dialogue.split())} words | "
              f"gold={len(gold_annotations)} annotations")

        predictions = run_zero_shot(dialogue)

        record = {
            "dialogue_id": i,
            "dialogue": dialogue,
            "gold_annotations": gold_annotations,
            "predictions": predictions
        }
        all_results.append(record)

        if predictions and "error" in predictions[0]:
            print(f"  ERROR: {predictions[0]['error']}")
        else:
            print(f"  Predicted: {len(predictions)} fallacies")

        # save incrementally every 10 dialogues
        # so progress is not lost if something crashes
        if (i + 1) % 10 == 0:
            save_results(all_results, "results/zero_shot_full.jsonl")
            print(f"  Progress saved ({i+1}/{total})")

        if i < total - 1:
            time.sleep(25)

    save_results(all_results, "results/zero_shot_full.jsonl")
    print(f"\nDone. {total} dialogues evaluated.")