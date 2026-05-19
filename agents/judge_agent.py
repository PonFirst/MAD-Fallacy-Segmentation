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

SYSTEM_PROMPT = """You are the Judge in a multi-agent debate system for logical fallacy detection.

You receive analysis reports from three specialized agents:
  - Logos: analyzes logical argument structure
  - Pathos: analyzes emotional language and rhetorical manipulation
  - Ethos: analyzes credibility and authority abuse

All span_text values in the reports are verbatim excerpts from the original dialogue.
Do not modify, paraphrase, or extend any span_text — copy them exactly as they appear
in the agent reports when emitting your final predictions.

WEIGHTING RULES — apply these before making your decision:
  - AppealtoEmotion, Slogans: Pathos's report carries the most weight.
    These fallacies are defined by their emotional mechanism, so Pathos's
    perspective is most relevant.
  - AppealtoAuthority, AdHominem: Ethos's report carries the most weight.
    These fallacies are defined by how credibility is invoked or attacked.
  - FalseCause, Slipperyslope: Logos's report carries the most weight.
    These fallacies are defined by flaws in logical inference structure.
  - If agents predict DIFFERENT fallacy labels for overlapping text spans:
    explicitly state which agent's perspective is most relevant to that
    span and why, then arbitrate to a single label.

SPAN REFINEMENT RULE:
  - Always prefer the minimal span that contains the core fallacious element.
  - For AppealtoEmotion and Slogans especially, if agents return long spans,
    select the shorter version that captures the emotionally loaded language.
  - span_text must be copied verbatim from an agent report — never invent new text.

DEDUPLICATION RULE:
  - If multiple agents flag the same span with the same label, emit it once.
  - Merge near-identical spans (same label, overlapping text) into one entry.

OUTPUT: respond ONLY with a valid JSON array. No text before or after. No markdown. No backticks.

[
  {
    "fallacy_label": "one of the six fallacy names",
    "span_text": "verbatim from an agent report",
    "reasoning": "one or two sentences explaining the final decision and which agent's perspective was most relevant"
  }
]

Valid fallacy labels: AdHominem, AppealtoAuthority, AppealtoEmotion, FalseCause, Slipperyslope, Slogans

If no fallacy is supported by the evidence across the reports, return an empty array: []"""


def run_judge(
    dialogue: str,
    logos_report: list,
    pathos_report: list,
    ethos_report: list,
    rebuttal_report: dict | None = None,
) -> list:
    """
    Synthesize agent reports into a final fallacy prediction list.
    The full dialogue is NOT sent to the Judge — only agent reports.
    span_text values in agent reports are already verbatim from the dialogue.

    Args:
        dialogue: kept for API compatibility but not sent to the model
        logos_report: predictions from the Logos agent
        pathos_report: predictions from the Pathos agent
        ethos_report: predictions from the Ethos agent
        rebuttal_report: optional dict with keys 'logos', 'pathos', 'ethos'
                         containing rebuttal responses (None if no debate round)

    Returns:
        list of dicts with keys: fallacy_label, span_text, reasoning
        or a list with one error dict if something fails
    """
    user_parts = [
        f"LOGOS REPORT (logical structure analysis):\n{json.dumps(logos_report, indent=2)}",
        f"\nPATHOS REPORT (emotional manipulation analysis):\n{json.dumps(pathos_report, indent=2)}",
        f"\nETHOS REPORT (credibility and authority analysis):\n{json.dumps(ethos_report, indent=2)}",
    ]

    if rebuttal_report is not None:
        user_parts.append(
            f"\nREBUTTAL ROUND (agents responded to each other's disagreements):\n"
            f"{json.dumps(rebuttal_report, indent=2)}"
        )

    user_parts.append(
        "\nSynthesize the above reports into a final list of fallacy predictions. "
        "Apply the weighting rules and span refinement rules before deciding. "
        "Copy span_text verbatim from the agent reports."
    )

    user_message = "\n".join(user_parts)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"  WARNING: judge output truncated — increase max_tokens")

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
                print(f"  Judge attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [{"error": "api_error", "details": str(e)}]


if __name__ == "__main__":
    from data_loader import load_dataset, get_train_test_split

    df = load_dataset()
    train, test = get_train_test_split(df)

    print("\nTesting Judge agent with mock reports on 2 dialogues...\n")

    for i in range(2):
        sample = test.iloc[i]
        dialogue = sample["dialogue"]
        gold_annotations = sample["annotations"]

        print(f"{'='*60}")
        print(f"Dialogue {i+1}")
        print(f"Dialogue length: {len(dialogue.split())} words")
        print(f"Gold annotations ({len(gold_annotations)}):")
        for ann in gold_annotations:
            print(f"  - {ann['fallacy']}: {ann['snippet'][:60]}...")

        mock_logos = [
            {"suspected_fallacy": ann["fallacy"], "span_text": ann["snippet"],
             "reasoning": "Mock Logos reasoning."}
            for ann in gold_annotations[:2]
        ]
        mock_pathos = [
            {"suspected_fallacy": ann["fallacy"], "span_text": ann["snippet"],
             "reasoning": "Mock Pathos reasoning."}
            for ann in gold_annotations[:1]
        ]
        mock_ethos = []

        print(f"\nRunning Judge with mock reports...")
        results = run_judge(dialogue, mock_logos, mock_pathos, mock_ethos)

        if results and "error" in results[0]:
            print(f"ERROR: {results[0]}")
        else:
            print(f"Judge predicted ({len(results)}):")
            for r in results:
                print(f"  - {r.get('fallacy_label')}: {str(r.get('span_text',''))[:60]}...")
                print(f"    Reasoning: {r.get('reasoning')}")

        if i < 1:
            time.sleep(20)