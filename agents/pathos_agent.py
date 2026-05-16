import os
import sys
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are Pathos, an expert in rhetoric and emotional persuasion.
Your job is to analyze political debate dialogues and identify ALL logical fallacies
based purely on EMOTIONAL LANGUAGE and RHETORICAL MANIPULATION — not the logical
structure of the argument or the speaker's credentials.

Fallacies are defined following Walton (1987) and Da San Martino et al. (2019):

- AdHominem: an emotionally charged personal attack on the arguer designed to
  provoke negative feelings in the audience rather than address the argument
- AppealtoAuthority: an emotionally reassuring reference to authority figures
  used to bypass critical thinking rather than provide evidence
- AppealtoEmotion: the unessential loading of the argument with emotional language
  to exploit the audience's emotional instinct — this includes loaded single words
  (e.g. charged adjectives, hyperbolic nouns) as well as longer emotional appeals
- FalseCause: an emotionally satisfying but unsupported causal narrative that
  exploits the audience's desire for simple explanations
- Slipperyslope: a fear-inducing chain of consequences presented as inevitable
  without rational justification
- Slogans: a brief and striking phrase used to provoke excitement or an emotional
  response in the audience, often accompanied by argument by repetition

You must respond ONLY with a valid JSON array. No text before or after. No markdown. No backticks.

[
  {
    "suspected_fallacy": "one of the six fallacy names above",
    "span_text": "the exact verbatim words from the dialogue containing the fallacy",
    "reasoning": "one or two sentences explaining how this language emotionally manipulates the audience"
  }
]

Rules:
- Find ALL emotionally manipulative fallacies present in the dialogue
- span_text must be copied verbatim from the dialogue — do not paraphrase or summarize
- span_text should be the minimal text that carries the emotional manipulation —
  per the annotation guidelines, AppealtoEmotion spans are on average 6.81 tokens
  and may be as short as a single emotionally loaded word or as long as a full sentence
- suspected_fallacy must be exactly one of: AdHominem, AppealtoAuthority,
  AppealtoEmotion, FalseCause, Slipperyslope, Slogans
- reasoning must explain the emotional manipulation mechanism, not the logical
  structure or speaker credibility
- if no fallacy is found, return an empty array: []
- return between 1 and 10 fallacies
- only include predictions you are confident about — do not pad with weak detections"""


def run_pathos(dialogue: str) -> list:
    user_message = f"""Analyze this political debate dialogue and identify ALL logical fallacies based on emotional language and rhetorical manipulation:

DIALOGUE:
{dialogue}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="models/gemma-4-31b-it",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )

            raw = response.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        except json.JSONDecodeError as e:
            return [{"error": "json_parse_error", "raw_response": raw, "details": str(e)}]
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                wait = 2 ** attempt
                print(f"  Attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [{"error": "api_error", "details": str(e)}]


if __name__ == "__main__":
    from data_loader import load_dataset, get_train_test_split

    df = load_dataset()
    train, test = get_train_test_split(df)

    print("\nTesting Pathos agent on 3 dialogues...\n")

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

        print(f"\nRunning Pathos...")
        results = run_pathos(dialogue)

        if results and "error" in results[0]:
            print(f"ERROR: {results[0]}")
        else:
            print(f"Predicted ({len(results)}):")
            for r in results:
                print(f"  - {r.get('suspected_fallacy')}: {str(r.get('span_text',''))[:60]}...")
                print(f"    Reasoning: {r.get('reasoning')}")