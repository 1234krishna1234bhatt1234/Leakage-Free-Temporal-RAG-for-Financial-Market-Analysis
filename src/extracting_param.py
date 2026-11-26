import os
from typing import Literal
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])      # no key in the code

class RetrievalInput(BaseModel):
    ticker: str = Field(description="Uppercase stock ticker, e.g. NVDA")
    search_query: str = Field(description="Short semantic search phrase, no dates or instructions")
    start: str = Field(description="Start date, YYYY-MM-DD")
    end: str = Field(description="End date, YYYY-MM-DD")
    as_of: str = Field(description="Knowledge cutoff date, YYYY-MM-DD")
    tasks: list[Literal["trend", "event_impact", "point_lookup"]] = Field(
        description="Every task the question asks for. A question can need more than one.")
    event_hint: str = Field(description="Event phrase such as 'May earnings', or empty string if none")

SYSTEM_PROMPT = f"""Extract structured arguments from the user's financial question.
Rules:
- ticker: map company names to tickers. Valid tickers: {', '.join(keep)}.
- Convert month or quarter ranges to exact dates ('Jan to Jun 2023' -> start 2023-01-01, end 2023-06-30).
- If the user says 'as of <date>', set as_of to that date. Otherwise set as_of equal to end.
- tasks: 'trend' for how sentiment or coverage changed over time, 'event_impact' for how the market
  reacted to an event, 'point_lookup' for what was the latest news at a date. Include all that apply.
- event_hint: the event phrase only (for example 'May earnings'); empty string if none.
- search_query: the financial topic only."""

def extract_params(user_text, model="openrouter/free"):
    schema = RetrievalInput.model_json_schema()
    schema["additionalProperties"] = False
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_text}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "retrieval_input", "strict": True, "schema": schema}},
        extra_body={"provider": {"require_parameters": True},
                    "plugins": [{"id": "response-healing"}]},
    )
    return RetrievalInput.model_validate_json(resp.choices[0].message.content)