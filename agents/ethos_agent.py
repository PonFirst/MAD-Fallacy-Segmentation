import os
import sys
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Ethos, an expert in credibility, authority, and source evaluation.
Your job is to analyze political debate dialogues and identify ALL logical fallacies
based purely on how CREDIBILITY and AUTHORITY are being used or abused — not the
emotional tone or the logical structure of the argument.

Fallacies are defined following Walton (1987) and Da San Martino et al. (2019):

- AdHominem: attacking a person's credibility, character, or personal qualities
  instead of addressing the substance of their argument
- AppealtoAuthority: relying on the endorsement of an authority figure or group
  consensus without providing sufficient evidence; includes citing non-experts,
  unverified sources, or majority opinion as proof
- AppealtoEmotion: using the speaker's own credibility or status to emotionally
  reassure the audience rather than provide substantive evidence
- FalseCause: using authoritative-sounding statistics or credentials to imply
  a causal relationship that has not been established
- Slipperyslope: citing expert predictions or authoritative sources to justify
  an unsupported chain of consequences
- Slogans: using the speaker's status or position to lend credibility to a brief
  striking phrase that substitutes for substantive argument

You must respond ONLY with a valid JSON array. No text before or after. No markdown. No backticks.

[
  {
    "suspected_fallacy": "one of the six fallacy names above",
    "span_text": "the exact verbatim words from the dialogue containing the fallacy",
    "reasoning": "one or two sentences explaining how credibility or authority is being abused"
  }
]

Rules:
- Find ALL credibility-based fallacies present in the dialogue
- span_text must be copied verbatim from the dialogue — do not paraphrase or summarize
- span_text should be the minimal text containing the credibility abuse — spans
  may range from a short phrase identifying an authority figure to a full sentence
  depending on where the abuse is concentrated
- suspected_fallacy must be exactly one of: AdHominem, AppealtoAuthority,
  AppealtoEmotion, FalseCause, Slipperyslope, Slogans
- reasoning must explain the credibility or authority abuse specifically, not the
  emotional content or logical structure
- if no fallacy is found, return an empty array: []
- return between 1 and 10 fallacies
- only include predictions you are confident about — do not pad with weak detections"""


def run_ethos(dialogue: str) -> list:
    """
    Run the Ethos agent on a single dialogue.

    Args:
        dialogue: the full debate speech turn text

    Returns:
        list of dicts, each with keys: suspected_fallacy, span_text, reasoning
        or a list with one error dict if something fails
    """
    user_message = f"""Analyze this political debate dialogue and identify ALL logical fallacies based on credibility and authority abuse:

DIALOGUE:
{dialogue}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            raw = response.choices[0].message.content.strip()

            # strip Qwen3 thinking blocks — handle both complete and partial blocks
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

            # strip markdown code blocks
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


if __name__ == "__main__":
    from data_loader import load_dataset, get_train_test_split

    df = load_dataset()
    train, test = get_train_test_split(df)

    print("\nTesting Ethos agent on 3 dialogues...\n")

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

        print(f"\nRunning Ethos...")
        results = run_ethos(dialogue)

        if results and "error" in results[0]:
            print(f"ERROR: {results[0]}")
        else:
            print(f"Predicted ({len(results)}):")
            for r in results:
                print(f"  - {r.get('suspected_fallacy')}: {str(r.get('span_text',''))[:60]}...")
                print(f"    Reasoning: {r.get('reasoning')}")