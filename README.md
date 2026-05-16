# Logos, Pathos, Ethos: A Rhetorically-Grounded Multi-Agent Framework for Fallacy Span Detection

A Multi-Agent Debate (MAD) system for span-level logical fallacy detection in political debate transcripts, grounded in Aristotle's rhetorical triad.

## Overview

This system jointly detects and classifies logical fallacies in full political debate speech turns using a Multi-Agent Debate (MAD) framework. Three specialized agents — Logos (logical structure), Pathos (emotional manipulation), Ethos (credibility abuse) — independently analyze each dialogue and produce structured reports. A Judge agent synthesizes these reports into a final prediction list, where each prediction includes a fallacy category label and exact character-level span boundaries marking where the fallacy occurs in the text.

Unlike prior work that classifies pre-segmented fallacy snippets into a single label per input, our system operates on full unsegmented speech turns averaging 1,600 words and predicts multiple co-occurring fallacies per dialogue across six categories: AdHominem, AppealtoAuthority, AppealtoEmotion, FalseCause, Slipperyslope, and Slogans.

## Dataset

ElecDeb60To20 (Goffredo et al., EMNLP 2023) — US presidential debate transcripts 1960–2020 annotated with six fallacy categories: AdHominem, AppealtoAuthority, AppealtoEmotion, FalseCause, Slipperyslope, Slogans.

## Agent Stack

| Agent  | Model                   | Provider | Role                       |
|--------|-------------------------|----------|----------------------------|
| Logos  | llama-3.3-70b-versatile | Groq     | Logical structure analysis |
| Pathos | gemma-4-31b-it          | Google   | Emotional manipulation     |
| Ethos  | qwen/qwen3-32b          | Groq     | Credibility and authority  |
| Judge  | openai/gpt-oss-120b     | Groq     | Multi-report synthesis     |

## Repository Structure

```
agents/          — Logos, Pathos, Ethos, Judge agent implementations
baselines/       — Zero-shot, few-shot, generic MAD baselines
data/            — ElecDeb60To20 dataset
exploratory/     — Data analysis scripts
data_loader.py   — Dataset loading and preprocessing
evaluate.py      — Greedy one-to-one span matching evaluation
pipeline.py      — Full MAD pipeline with asyncio
config.py        — Model names and hyperparameters
```
## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY and GEMINI_API_KEY to .env
python3 pipeline.py
python3 evaluate.py
```