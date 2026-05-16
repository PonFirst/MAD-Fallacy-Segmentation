# config.py
LOGOS_MODEL = "llama-3.3-70b-versatile"
PATHOS_MODEL = "models/gemma-4-31b-it"
ETHOS_MODEL = "qwen/qwen3-32b"
JUDGE_MODEL = "openai/gpt-oss-120b"

TEMPERATURE = 0.1
MAX_TOKENS = 2048
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 20

FALLACY_LABELS = [
    "AdHominem",
    "AppealtoAuthority",
    "AppealtoEmotion",
    "FalseCause",
    "Slipperyslope",
    "Slogans"
]