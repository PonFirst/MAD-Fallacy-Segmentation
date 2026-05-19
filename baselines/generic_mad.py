import os
import sys
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_dataset, get_train_test_split
from agents.judge_agent import run_judge

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Generic agents use LLaMA 3.3 70B (same as Logos in the full MAD system) to
# preserve GPT-OSS-120B's 200K TPD quota for the Judge across all MAD variants.
# All three agents share the same non-specialized system prompt — the only
# structural difference from zero-shot is three independent calls + Judge synthesis.
# This isolates the contribution of rhetorical specialization.
GENERIC_AGENT_MODEL = "llama-3.3-70b-versatile"
INTER_AGENT_SLEEP = 10  # seconds — LLaMA TPM is 12K; 3 calls × ~3K tokens needs ~45s/dialogue

GENERIC_SYSTEM_PROMPT = """You are an expert in logical fallacy detection and argumentation theory.
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


def run_generic_agent(dialogue: str, agent_id: str) -> list:
    """Run one generic (non-specialized) agent on a dialogue."""
    user_message = (
        f"Analyze this political debate dialogue and identify ALL logical fallacies:\n\n"
        f"DIALOGUE:\n{dialogue}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GENERIC_AGENT_MODEL,
                messages=[
                    {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"  WARNING: agent {agent_id} output truncated")

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
                print(f"  Agent {agent_id} attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [{"error": "api_error", "details": str(e)}]


def run_generic_mad(dialogue: str) -> list:
    """
    Run three identical generic agents (LLaMA 3.3 70B) sequentially, then
    synthesize with the Judge (GPT-OSS 120B). Sequential to stay within LLaMA's
    30 RPM limit. No debate trigger — this baseline tests specialization value.

    Returns the Judge's output with fallacy_label renamed to suspected_fallacy
    for compatibility with evaluate.py.
    """
    report_a = run_generic_agent(dialogue, "A")
    time.sleep(INTER_AGENT_SLEEP)
    report_b = run_generic_agent(dialogue, "B")
    time.sleep(INTER_AGENT_SLEEP)
    report_c = run_generic_agent(dialogue, "C")
    time.sleep(5)  # brief pause before Judge (different model, separate quota)

    judge_output = run_judge(dialogue, report_a, report_b, report_c)

    # Normalize Judge output key to match evaluate.py expectations
    normalized = []
    for item in judge_output:
        if "error" in item:
            normalized.append(item)
        else:
            normalized.append({
                "suspected_fallacy": item.get("fallacy_label", ""),
                "span_text": item.get("span_text", ""),
                "reasoning": item.get("reasoning", ""),
            })
    return normalized


def save_results(results_list, path="results/generic_mad_full.jsonl"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        for record in results_list:
            f.write(json.dumps(record) + "\n")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    df = load_dataset()
    train, test = get_train_test_split(df)

    total = len(test)
    print(f"\nRunning Generic MAD baseline on all {total} test dialogues...\n")
    # 3 LLaMA agent calls (10s sleep each) + Judge + 20s between dialogues
    secs_per_dialogue = 2 * INTER_AGENT_SLEEP + 5 + 20
    print(f"Estimated time: {total * secs_per_dialogue / 60:.1f} minutes "
          f"(~{total * 3 * 3000 // 1000}K LLaMA tokens, TPD limit = 100K)\n")

    all_results = []

    for i in range(total):
        sample = test.iloc[i]
        dialogue = sample["dialogue"]
        gold_annotations = sample["annotations"]

        print(f"[{i+1}/{total}] length={len(dialogue.split())} words | "
              f"gold={len(gold_annotations)} annotations")

        predictions = run_generic_mad(dialogue)

        record = {
            "dialogue_id": i,
            "dialogue": dialogue,
            "gold_annotations": gold_annotations,
            "predictions": predictions,
        }
        all_results.append(record)

        if predictions and "error" in predictions[0]:
            print(f"  ERROR: {predictions[0]['error']}")
        else:
            print(f"  Predicted: {len(predictions)} fallacies")

        if (i + 1) % 10 == 0:
            save_results(all_results, "results/generic_mad_full.jsonl")
            print(f"  Progress saved ({i+1}/{total})")

        if i < total - 1:
            time.sleep(20)

    save_results(all_results, "results/generic_mad_full.jsonl")
    print(f"\nDone. {total} dialogues evaluated.")
