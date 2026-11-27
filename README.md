# Leakage-Free Temporal RAG for Financial Market Intelligence

A retrieval-augmented system that answers questions like *"How did sentiment around NVDA change from Jan to Jun 2023, and how did the market react to its May earnings? Answer as of 31 May 2023."*

- **Point-in-time retrieval:** every question has an as-of date. Articles published after it are excluded inside the database query, so they never reach the LLM.
- **Tool-computed numbers:** weekly FinBERT sentiment and event-study abnormal returns are computed by code, not by the LLM.
- **Verifier:** checks every number and citation in the answer against the tool outputs.

It is a retrospective research tool. It is not a trading signal or investment advice.

## 1. Requirements

- Python 3.10+
- About 5 GB of free disk space (data, vector database)
- An OpenRouter API key (for the question parser and the answer writer)
- A GPU is **not** needed to run the project. It is only needed if you regenerate the embeddings and sentiment scores from scratch (Section 6).

## 2. Setup

```bash
git clone https://github.com/1234krishna1234bhatt1234/Leakage-Free-Temporal-RAG-for-Financial-Market-Analysis
cd Leakage-Free-Temporal-RAG-for-Financial-Market-Analysis

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install pandas pyarrow numpy tqdm yfinance scipy lxml chromadb \
    sentence-transformers transformers torch openai pydantic python-dotenv jupyter
```

Create a file named `.env` in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` or the `data/` folder.

## 3. Get the data (from Google Drive)

Download these from the shared Drive folder: **`<YOUR_DRIVE_LINK>`** (`ie624_project/data`).

Place them in `data/` so the project looks like this:

```
data/
  raw/
    cleaned_data.parquet        # cleaned articles (one row per article)
  emb_shards/
    shard_000.npz ...           # saved embeddings (portable backup, about 320 MB)
  finbert_results.parquet       # FinBERT scores per article
  prices.parquet                # daily adjusted closes, 30 tickers + ^GSPC
  earnings.parquet              # earnings dates
```

You do **not** need to download `chroma_full/` from Drive. A copied Chroma folder often fails to load on another machine, so it is rebuilt from the shards in the next step.

`raw/combined.parquet` is only needed if you want to rerun the cleaning (Section 6).

## 4. Rebuild the vector database (once per machine)

Run from the project root:

```bash
python -c "from src.embed import rebuild_chroma_from_shards; rebuild_chroma_from_shards()"
```

This takes a while on CPU and prints progress per shard. It creates `data/chroma_full/`. The last printed line shows two numbers (database size and number of articles); they must be equal. No re-embedding happens.

You do not repeat this step unless the folder is deleted, `cleaned_data.parquet` is regenerated, or `chromadb` is upgraded to an incompatible version.

## 5. Run

Open `main.ipynb` from the project root and run the cells in order:

1. **Load everything**
   ```python
   from src.loaders import load_all
   ctx = load_all()
   docs, sent, prices, col, model, keep = ctx.docs, ctx.sent, ctx.prices, ctx.col, ctx.model, ctx.tickers
   ```
   It checks that the database size equals the number of articles. Rerun this cell after every kernel restart.
2. **Parser cell** (OpenRouter client, schema, `extract_params`).
3. **Ask a question**
   ```python
   question = ("How did sentiment around NVDA change from Jan to Jun 2023, "
               "and how did the market react to its May earnings? Answer as of 31 May 2023.")
   params = extract_params(question)
   ```
4. **Run the pipeline**
   ```python
   from src.pipeline import run
   from src.humanizer import write
   from src.verifier import verify_and_rewrite

   pkg = run(params, ctx)                         # retrieval + sentiment + event study
   answer = write(question, pkg)                  # LLM writes from the package only
   final, report = verify_and_rewrite(question, pkg, answer, col=col)
   print(final); print(report["passed"], report["failures"])
   ```
5. **Compare with baselines**
   ```python
   from src.baselines import rag_baseline
   plain = rag_baseline(question, params, ctx, variant="plain")
   print(plain["answer"]); print("articles after as_of:", plain["n_after_as_of"])
   ```

Supported companies are the 30 tickers in `keep`. Questions must fall inside the data range (2009 to 2024), and the as-of date must be on or after the start of the window.

### Quick sanity checks

```python
from src.retrieval import retrieve
res = retrieve("Nvidia earnings", "NVDA", "2023-01-01", "2023-06-30", "2023-05-31",
               col=col, model=model, sent=sent)
print(res[["date", "sim", "score", "sentiment", "title"]])   # all dates <= 2023-05-31
```

## 6. iF Regenerating everything from scratch 

Only needed to rebuild the data pipeline. Steps 3 and 4 need a GPU (we used a Colab T4).

1. Download the parquet files of the Hugging Face dataset `benstaf/FNSPID-filtered-nasdaq-100`, concatenate them, drop duplicate rows, and save `data/raw/combined.parquet`.
2. `python -m src.preprocess` creates `data/raw/cleaned_data.parquet` (the script refuses to overwrite an existing file, because embeddings and scores depend on its row order).
3. On a GPU machine: `from src.embed import embed_all; embed_all()` creates the embedding shards.
4. On a GPU machine: `from src.sentiment import score_all, merge_parts; score_all(); merge_parts()` creates `finbert_results.parquet`.
5. `from src.prices import download_prices; download_prices()` and `from src.events import fetch_earnings; fetch_earnings(keep)`.
6. Rebuild Chroma (Section 4).

On Colab, mount Drive first (`drive.mount('/content/drive')`). `src/config.py` detects Colab and switches paths and device automatically.

## 7. Project structure

```
src/
  config.py        paths, constants, Colab/local switch
  preprocess.py    raw articles -> cleaned documents
  embed.py         embeddings in shards; rebuild Chroma from shards
  sentiment.py     FinBERT scoring; weekly sentiment tool
  prices.py        price download and returns
  retrieval.py     point-in-time retrieval (switches give the baselines)
  events.py        event dates (earnings, large-move fallback)
  event_study.py   market model, abnormal returns, CAR, t-statistic
  pipeline.py      question -> evidence package
  humanizer.py     package -> written answer (LLM)
  verifier.py      checks numbers and citations of an answer
  baselines.py     plain and recency RAG
  loaders.py       load_all()
main.ipynb         parser, pipeline runs, comparisons
```

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run from the project root (the folder containing `src/`). |
| `Error loading hnsw index` | The Chroma folder is not usable on this machine. Delete `data/chroma_full` and rebuild (Section 4). |
| `chroma ... != docs ...` assertion in `load_all()` | The rebuild is incomplete or a shard is missing. Check `data/emb_shards` and rebuild. |
| `Ticker ... not recognised` | The parser returned a company outside the 30 supported tickers. Rephrase with the company name. |
| `KeyError: 'OPENROUTER_API_KEY'` | Add the key to `.env` in the project root. |
| Event study returns `insufficient_history` or `window_extends_past_as_of` | Expected: the company lacks 100+ prior trading days, or the window would run past the as-of date. |
| Answers differ between runs | The free OpenRouter router may pick different models. Pin one model in the parser and humanizer. |

## 9. Limitations

The LLM's own training knowledge can still leak and is not controlled by retrieval filtering. Ticker tags are noisy (roundup articles carry one ticker). FinBERT reads at most 512 tokens, and long articles are scored from their summary. The verifier checks answers against the evidence package, not against the real world.