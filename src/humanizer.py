"""Second LLM: turns the package into a readable answer. It gets ONLY the package."""
from src.config import *
import json
from openai import OpenAI

DEFAULT_MODEL = "openrouter/free"        # pin one specific model if output quality varies
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    return _client

SYSTEM = """You write a short, plain-English answer to a financial question using ONLY the JSON package you are given.

Rules:
1. Use no outside knowledge. Do not use anything you know about this company or period.
2. Copy numbers exactly as they appear in the package. Never calculate, round, convert or estimate new numbers.
3. Cite every factual sentence. For an article use [doc_id, YYYY-MM-DD]. For weekly sentiment numbers use [weekly]. For event numbers use [event YYYY-MM-DD].
4. If a window has "available": false, or an event status is not "ok", say that figure is not available and give the reason. Never fill the gap.
5. An event with kind "large_move" is the largest price move in the window. Do not call it earnings.
6. Articles with "multi_ticker_roundup": true cover many companies. Do not use them as evidence about this one company.
7. Sentiment is the tone of coverage, not a price move. Do not suggest sentiment caused the price reaction.
8. Explain "car_pct" in plain words, as the return beyond what the market model expected. A t-stat or p-value only supports "statistically significant" if the package gives it.
9. Mention the as-of date. This is a retrospective analysis, not a trading signal or investment advice. Keep it to about 200 words, in short paragraphs, with no headings."""

def build_messages(question, package, feedback=None):
    user = f"Question:\n{question}\n\nPackage (JSON):\n{json.dumps(package, indent=1, ensure_ascii=False, default=str)}"
    if feedback:
        user += ("\n\nYour previous answer failed these checks. Rewrite it and fix every item, "
                 "using only the package:\n- " + "\n- ".join(feedback))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def write(question, package, model=DEFAULT_MODEL, feedback=None, client=None):
    client = client or get_client()
    resp = client.chat.completions.create(model=model, temperature=0,
                                          messages=build_messages(question, package, feedback))
    return resp.choices[0].message.content.strip()