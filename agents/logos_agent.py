import os
import sys
import json
import re
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Logos, an expert in formal logic and argumentation theory.
Your job is to analyze political debate dialogues and identify ALL logical fallacies
based purely on the STRUCTURE of the argument — not its emotional tone or the
speaker's credibility.

Fallacies are defined following Walton (1987) and Da San Martino et al. (2019):

- AdHominem: an excessive attack on the arguer's position, character, or personal
  qualities rather than addressing the substance of their argument
- AppealtoAuthority: relying on the endorsement of an authority figure or group
  consensus without providing sufficient evidence; may involve non-experts or
  majority opinion
- AppealtoEmotion: loading the argument with emotional language to exploit the
  audience's emotional instinct rather than appeal to reason
- FalseCause: misinterpreting the correlation of two events as causation
- Slipperyslope: implying that an improbable or exaggerated consequence will
  inevitably result from a particular action
- Slogans: a brief and striking phrase used to provoke excitement in the audience,
  often accompanied by argument by repetition

You must respond ONLY with a valid JSON array. No text before or after. No markdown. No backticks.

[
  {
    "suspected_fallacy": "one of the six fallacy names above",
    "span_text": "the exact verbatim words from the dialogue containing the fallacy",
    "reasoning": "one or two sentences explaining the structural flaw in argument terms"
  }
]

Rules:
- Find ALL fallacies present in the dialogue based on argument structure alone
- span_text must be copied verbatim from the dialogue — do not paraphrase or summarize
- span_text should be the minimal text that contains the fallacy — per the annotation
  guidelines, fallacious spans range from single words to full sentences depending
  on where the faulty reasoning is concentrated
- suspected_fallacy must be exactly one of: AdHominem, AppealtoAuthority,
  AppealtoEmotion, FalseCause, Slipperyslope, Slogans
- reasoning must focus on the structural argument flaw, not emotional tone or
  speaker credibility
- if no fallacy is found, return an empty array: []
- return between 1 and 10 fallacies
- only include predictions you are confident about — do not pad with weak detections"""


def run_logos(dialogue: str) -> list:
    """
    Run the Logos agent on a single dialogue.

    Args:
        dialogue: the full debate speech turn text

    Returns:
        list of dicts, each with keys: suspected_fallacy, span_text, reasoning
        or a list with one error dict if something fails
    """
    user_message = f"""Analyze this political debate dialogue and identify ALL logical fallacies based on argument structure alone:

DIALOGUE:
{dialogue}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)

        # ensure it's a list
        if isinstance(parsed, dict):
            parsed = [parsed]

        return parsed

    except json.JSONDecodeError as e:
        return [{"error": "json_parse_error", "raw_response": raw, "details": str(e)}]
    except Exception as e:
        return [{"error": "api_error", "details": str(e)}]


if __name__ == "__main__":
    from data_loader import load_dataset, get_train_test_split

    df = load_dataset()
    train, test = get_train_test_split(df)

    print("\nTesting Logos agent on 3 dialogues...\n")

    for i in range(3):
        sample = test.iloc[i]
        dialogue = sample["dialogue"]
        gold_annotations = sample["annotations"]

        print(f"{'='*60}")
        print(f"Dialogue {i+1}")
        print(f"Dialogue length: {len(dialogue.split())} words")
        print(f"Gold annotations ({len(gold_annotations)}):")
        for ann in gold_annotations:
            print(f"  - {ann['fallacy']}: {ann['snippet'][:60]}...")

        print(f"\nRunning Logos...")
        results = run_logos(dialogue)

        if results and "error" in results[0]:
            print(f"ERROR: {results[0]}")
        else:
            print(f"Predicted ({len(results)}):")
            for r in results:
                print(f"  - {r.get('suspected_fallacy')}: {str(r.get('span_text',''))[:60]}...")
                print(f"    Reasoning: {r.get('reasoning')}")