"""Verifier: checks an answer against the evidence package. No LLM involved.

  numbers   every non-trivial number must match a package value (within the rounding
            of the number as written) or appear in an article cited in the same sentence
  citations [doc_id, date] must name a package article with that exact date, on or before
            as_of (and exist in Chroma if `col` is given); [weekly] needs a weekly table;
            [event date] needs an event on that date
  cutoff    explicit dates after as_of fail; later years / month-years only warn
It checks the answer against the package, not against the real world, and it cannot
judge qualitative claims.
"""
import re
from src.config import ts, DAY

DASHES = re.compile(r"[\u2010-\u2015\u2212]")          # non-breaking hyphens, en/em dashes, minus
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}

DOC_CIT = re.compile(r"\[\s*(?:doc_id\s*:?\s*)?([0-9a-f]{16})\s*,\s*(\d{4}-\d{2}-\d{2})\s*\]", re.I)
EVENT_CIT = re.compile(r"\[\s*event\s+(\d{4}-\d{2}-\d{2})\s*\]", re.I)
WEEKLY_CIT = re.compile(r"\[\s*weekly\s*\]", re.I)
ANY_CIT = re.compile(r"\[[^\]]*\]")
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
HEX16 = re.compile(r"\b[0-9a-f]{16}\b")
MD1 = re.compile(rf"\b(?:{MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I)   # May 25
MD2 = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})[a-z]*\b", re.I)      # 25 May
QTR = re.compile(r"\b[QH][1-4]\b|\bFY\d{2,4}\b", re.I)
NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
MONTH_YEAR = re.compile(rf"\b({MONTHS})[a-z]*\.?\s+(\d{{4}})\b", re.I)
DISCLAIMER = re.compile(r"retrospective|investment advice|trading signal|as of", re.I)
STRUCTURAL = {20, 120}                                   # estimation-window labels


def _clean(text):
    t = DASHES.sub("-", text)
    for rx in (ANY_CIT, ISO, HEX16, MD1, MD2, QTR):
        t = rx.sub(" ", t)
    return t


def numbers(text):
    """List of (raw, abs value, decimals, is_percent) for numbers in the text."""
    out = []
    for m in NUM.finditer(_clean(text)):
        raw = m.group(0)
        body = raw.rstrip("%").replace(",", "")
        try:
            v = abs(float(body))
        except ValueError:
            continue
        dec = len(body.split(".")[1]) if "." in body else 0
        out.append((raw, v, dec, raw.endswith("%")))
    return out


def _benign(v, dec, pct):
    """Small counts, years and window labels are not checked."""
    return (not pct and dec == 0 and (v <= 12 or 1900 <= v <= 2100 or v in STRUCTURAL))


def _matches(v, dec, allowed):
    tol = 0.5 * 10 ** (-dec) + 1e-9                      # rounding of the number as written
    return any(abs(v - a) <= tol for a in allowed)


def package_numbers(pkg):
    out = set()

    def walk(o, key=""):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            out.add(abs(float(o)))
        elif isinstance(o, str) and key == "p_value":    # e.g. "<0.001"
            out.update(v for _, v, _, _ in numbers(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key)
    walk(pkg)
    return out


def verify(answer, package, col=None):
    as_of = package["as_of"]
    as_of_ts = ts(as_of) + DAY - 1
    text = DASHES.sub("-", answer)
    arts = {a["doc_id"]: a for a in package.get("articles", [])}
    art_nums = {i: {v for _, v, _, _ in numbers(a["title"] + " " + a.get("snippet", ""))}
                for i, a in arts.items()}
    allowed = package_numbers(package)
    events = {e["date"]: e for e in package.get("events", [])}
    has_weekly = bool(package.get("weekly"))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[\"(*])", text) if s.strip()]
    failures, warnings, bad = [], [], set()
    n_num = n_num_bad = n_cit = n_cit_bad = 0

    def add_fail(i, msg):
        failures.append(msg)
        if i is not None:
            bad.add(i)

    for i, s in enumerate(sentences):
        doc_c, ev_c, wk = DOC_CIT.findall(s), EVENT_CIT.findall(s), bool(WEEKLY_CIT.search(s))

        # ---- citations ----
        for did, d in doc_c:
            n_cit += 1
            msg = None
            if did not in arts:
                msg = f"Citation {did} is not in the package."
            elif arts[did]["date"] != d:
                msg = f"Citation {did} has date {d}, but the article is dated {arts[did]['date']}."
            elif d > as_of:
                msg = f"Citation {did} is dated {d}, after the as-of date {as_of}."
            elif col is not None:
                r = col.get(ids=[did], include=["metadatas"])
                if not r["ids"]:
                    msg = f"Citation {did} does not exist in the database."
                elif r["metadatas"][0]["publish_time"] > as_of_ts:
                    msg = f"Citation {did} was published after the as-of date."
            if msg:
                n_cit_bad += 1
                add_fail(i, msg)
        for d in ev_c:
            n_cit += 1
            if d not in events:
                n_cit_bad += 1
                add_fail(i, f"Citation [event {d}] does not match any event in the package.")
            elif events[d].get("kind") == "large_move" and re.search(r"\bearnings\b", s, re.I):
                add_fail(i, f"The event on {d} is a large price move, not confirmed earnings.")
        if wk:
            n_cit += 1
            if not has_weekly:
                n_cit_bad += 1
                add_fail(i, "Citation [weekly] used, but the package has no weekly table.")

        # ---- numbers ----
        cited_nums = set().union(*[art_nums.get(d, set()) for d, _ in doc_c])
        nums = [x for x in numbers(s) if not _benign(x[1], x[2], x[3])]
        for raw, v, dec, pct in nums:
            n_num += 1
            if not (_matches(v, dec, allowed) or _matches(v, dec, cited_nums)):
                n_num_bad += 1
                add_fail(i, f"Number '{raw}' matches no tool output or cited article "
                            f"(sentence: \"{s[:80]}...\").")
        if nums and not (doc_c or ev_c or wk):
            add_fail(i, f"Sentence with numbers has no citation: \"{s[:80]}...\"")
        elif not (doc_c or ev_c or wk) and len(s.split()) > 8 and not DISCLAIMER.search(s):
            warnings.append(f"Uncited sentence: \"{s[:80]}...\"")

    # ---- cutoff ----
    for d in sorted(set(ISO.findall(text))):
        if d > as_of:
            add_fail(None, f"The answer mentions the date {d}, after the as-of date {as_of}.")
    body = ANY_CIT.sub(" ", text)
    ay = int(as_of[:4])
    for y in sorted({int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", body)}):
        if y > ay:
            warnings.append(f"Mentions the year {y}, after the as-of year (check it comes from a cited article).")
    for m, y in MONTH_YEAR.findall(body):
        if (int(y), MONTH_NUM[m.lower()[:3]], 1) > tuple(int(x) for x in as_of.split("-")):
            warnings.append(f"Mentions {m} {y}, after the as-of date.")

    return {"passed": not failures, "failures": failures, "warnings": warnings,
            "numbers_checked": n_num, "numbers_unsupported": n_num_bad,
            "citations_checked": n_cit, "citations_invalid": n_cit_bad,
            "numeric_accuracy": None if not n_num else 1 - n_num_bad / n_num,
            "citation_validity": None if not n_cit else 1 - n_cit_bad / n_cit,
            "bad_sentences": sorted(bad), "sentences": sentences}


def flag_unsupported(report):
    """Re-join the answer, marking sentences that failed a check."""
    bad = set(report["bad_sentences"])
    return " ".join(s + (" [UNVERIFIED]" if i in bad else "") for i, s in enumerate(report["sentences"]))


def verify_and_rewrite(question, package, answer, col=None, max_retries=1, **write_kw):
    """Verify; on failure regenerate (default once) with the failures listed, then flag leftovers."""
    from src.humanizer import write
    rep, tries = verify(answer, package, col), 0
    while not rep["passed"] and tries < max_retries:
        answer = write(question, package, feedback=rep["failures"][:10], **write_kw)
        rep, tries = verify(answer, package, col), tries + 1
    rep["regenerations"] = tries
    return (answer if rep["passed"] else flag_unsupported(rep)), rep