"""Orchestrator: parsed question -> ONE package of evidence and tool numbers.
Everything numeric is computed here. The humanizer LLM only sees this package."""
from src.config import *
from src.retrieval import retrieve
from src.sentiment import weekly_sentiment
from src.events import load_earnings, resolve_events
from src.event_study import event_study

LISTING = re.compile(r"options?\s+begin\s+trading", re.I)            # listing notices, no content
ROUNDUP = re.compile(r"after-hours earnings report|pre-market most active|most active for|stocks? making", re.I)
WINDOWS = {(-1, 1): "days_-1_to_+1", (0, 5): "days_0_to_+5"}

def pick_articles(res, max_articles):
    if res.empty:
        return []
    res = res[~res.title.str.contains(LISTING)].copy()
    res = res.sort_values("score", ascending=False).head(max_articles).sort_values("publish_time")
    return [{"doc_id": r.doc_id, "date": str(r.date), "title": r.title,
             "sentiment": round(float(r.sentiment), 3) if pd.notna(r.sentiment) else None,
             "multi_ticker_roundup": bool(ROUNDUP.search(r.title)),
             "snippet": r.text[:500]} for r in res.itertuples()]

def weekly_block(docs, p, eff_end):
    wk = weekly_sentiment(docs, p.ticker, p.start, eff_end, p.as_of)
    if wk.empty:
        return [], {}
    wk["partial"] = ((wk.week_start < pd.Timestamp(p.start)) |
                     (wk.week_start + pd.Timedelta(days=6) > pd.Timestamp(eff_end)))
    d = lambda x: x.strftime("%Y-%m-%d")
    rows = [{"week_start": d(r.week_start), "mean_sentiment": round(float(r.mean_sentiment), 3),
             "n_docs": int(r.n_docs), "low_confidence": bool(r.low_confidence),
             "partial_week": bool(r.partial)} for r in wk.itertuples()]

    tot = int(wk.n_docs.sum())
    summ = {"n_articles_total": tot,
            "overall_mean_sentiment": round(float((wk.mean_sentiment * wk.n_docs).sum() / tot), 3)}
    busy = wk.loc[wk.n_docs.idxmax()]
    summ["busiest_week"] = {"week_start": d(busy.week_start), "n_docs": int(busy.n_docs)}
    ok = wk[(~wk.low_confidence) & (~wk.partial)]               # complete, well-covered weeks only
    if len(ok):
        pk, tr = ok.loc[ok.mean_sentiment.idxmax()], ok.loc[ok.mean_sentiment.idxmin()]
        summ["peak_week"] = {"week_start": d(pk.week_start), "mean_sentiment": round(float(pk.mean_sentiment), 3)}
        summ["trough_week"] = {"week_start": d(tr.week_start), "mean_sentiment": round(float(tr.mean_sentiment), 3)}
        k = min(4, len(ok) // 2)
        if k >= 1:
            early, late = float(ok.head(k).mean_sentiment.mean()), float(ok.tail(k).mean_sentiment.mean())
            summ.update({"weeks_in_early_and_late_groups": k,
                         "early_mean_sentiment": round(early, 3),
                         "late_mean_sentiment": round(late, 3),
                         "change_late_minus_early": round(late - early, 3)})
    return rows, summ

def event_record(prices, ticker, ev, as_of):
    s = event_study(prices, ticker, ev["date"], as_of)
    rec = {"kind": ev["kind"], "date": str(pd.Timestamp(ev["date"]).date()), "status": s["status"]}
    if ev["kind"] == "large_move":
        rec["day_return_pct"] = round(100 * ev["ret"], 2)
    if s["status"] != "ok":
        return rec
    rec.update({"beta": round(s["beta"], 2), "n_estimation_days": s["n_est"]})
    for (a, b), name in WINDOWS.items():
        car = s.get(f"car_{a}_{b}")
        if car is None:
            rec[name] = {"available": False, "reason": s.get(f"note_{a}_{b}", "unavailable")}
        else:
            p_ = s[f"p_{a}_{b}"]
            rec[name] = {"available": True, "car_pct": round(100 * car, 2),
                         "t_stat": round(s[f"t_{a}_{b}"], 2),
                         "p_value": "<0.001" if p_ < 0.001 else round(p_, 3),
                         "market_adjusted_pct": round(100 * s[f"mktadj_{a}_{b}"], 2)}
    return rec

def run(params, ctx, earnings=None, max_articles=15):
    p = params
    if p.ticker not in ctx.tickers:
        raise ValueError(f"Unsupported ticker {p.ticker}. Supported: {ctx.tickers}")
    eff_end = min(pd.Timestamp(p.end), pd.Timestamp(p.as_of)).strftime("%Y-%m-%d")
    if pd.Timestamp(p.start) > pd.Timestamp(eff_end):
        raise ValueError("The window starts after the as-of date, so nothing is allowed to be used.")

    notes, tasks = [], set(p.tasks)
    if eff_end != p.end:
        notes.append(f"Window end {p.end} was cut to the as-of date {eff_end}. Nothing after it is used.")

    # evidence articles (always), via point-in-time retrieval
    only_lookup = tasks == {"point_lookup"}
    res = retrieve(p.search_query, p.ticker, p.start, eff_end, p.as_of,
                   col=ctx.col, model=ctx.model, sent=ctx.sent,
                   use_buckets=not only_lookup, k_total=max_articles)
    pkg = {"as_of": p.as_of, "ticker": p.ticker, "window": {"start": p.start, "end": eff_end},
           "tasks": list(p.tasks), "articles": pick_articles(res, max_articles),
           "weekly": [], "weekly_summary": {}, "events": [], "notes": notes}

    if "trend" in tasks:
        pkg["weekly"], pkg["weekly_summary"] = weekly_block(ctx.docs, p, eff_end)
        if not pkg["weekly"]:
            notes.append("No articles in the window, so there is no sentiment timeline.")

    if "event_impact" in tasks:
        if earnings is None:
            earnings = load_earnings()
        evs = resolve_events(ctx.prices, p.ticker, p.start, eff_end, p.as_of,
                             hint=p.event_hint, earnings=earnings)[-4:]
        pkg["events"] = [event_record(ctx.prices, p.ticker, e, p.as_of) for e in evs]
        if not evs:
            notes.append("No matching event was found in the window.")
        if any(e["kind"] == "large_move" for e in evs):
            notes.append("Some events are the largest price-move days, not confirmed earnings dates.")
        notes.append("Event day 0 is the first trading day the market could react. "
                     "After-close announcements count from the next trading day.")
    return pkg