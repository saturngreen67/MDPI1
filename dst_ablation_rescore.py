

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

INPUT_DIR  = Path(r"D:\DST_Results\ablation\inputs")
OUTPUT_DIR = Path(r"D:\DST_Results\ablation")

HAZARD_TABLE_FILE = "Hazard index.md"
PRI_TABLE_FILE    = "PRI.md"
HAZARD_REF_FILE   = "Hazard report.txt"
PRI_REF_FILE      = "PRI report.txt"

GEMINI_API_KEY_OVERRIDE = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_MODEL_VERSION    = "gemini-2.5-flash-lite"
GEMINI_PACING_SEC       = 7

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:latest"

NLI_MODEL_NAME = "roberta-large-mnli"
NLI_BATCH_SIZE = 8

N_REPETITIONS = 3

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_inputs(input_dir: Path):
    def read(fn, required=True):
        p = input_dir / fn
        if not p.exists():
            if required:
                logging.error("Missing required input: %s", p)
                sys.exit(1)
            return ""
        return p.read_text(encoding="utf-8")

    return (
        read(HAZARD_TABLE_FILE),
        read(PRI_TABLE_FILE),
        read(HAZARD_REF_FILE, required=False),
        read(PRI_REF_FILE, required=False),
    )


def load_existing_reports(reports_dir: Path):
    """Discover all generated reports and parse their tags.

    Tag format: {rtype}_{cond}_rep{N}
    rtype in {hazard, pri}
    cond  in {full, noexample, llama_full, llama_noexample}
    """
    if not reports_dir.exists():
        logging.error("Reports dir does not exist: %s", reports_dir)
        sys.exit(1)

    pattern = re.compile(r"^(hazard|pri)_(full|noexample|llama_full|llama_noexample)_rep(\d+)\.md$")
    rows = []
    for f in sorted(reports_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        rtype, cond, rep = m.group(1), m.group(2), int(m.group(3))
        text = f.read_text(encoding="utf-8")
        if len(text.split()) < 40:
            logging.warning("Skipping %s (too short: %d words)", f.name, len(text.split()))
            continue
        model = "llama" if cond.startswith("llama") else "gemini"
        rows.append({
            "report_id": f.stem,
            "type":      rtype,
            "condition": cond,
            "repetition": rep,
            "model":     model,
            "path":      f,
            "text":      text,
        })
    logging.info("Discovered %d reports", len(rows))
    return rows

NUMERIC_RE_FULL = re.compile(r"\b\d+(?:[,]\d{3})*(?:\.\d+)?\b")

SCALE_OK = {"0", "1", "2", "3", "4", "5"}


def numeric_hallucinations_with_context(generated: str, allowed_text: str,
                                        report_id: str, window: int = 40):
    """Return list of dicts: every flagged numeric token with surrounding context.

    A token is flagged if it appears in `generated` but:
      - is not in SCALE_OK (whitelisted index values)
      - does NOT appear in `allowed_text` (the input tables)

    Note: `allowed_text` is the INPUT tables only, never the expert reference.
    The reference is for similarity scoring, not grounding.
    """
    allowed_nums = set(NUMERIC_RE_FULL.findall(allowed_text))
    flagged = []
    for m in NUMERIC_RE_FULL.finditer(generated):
        tok = m.group(0)
        if tok in SCALE_OK:
            continue
        if tok in allowed_nums:
            continue
        start = max(0, m.start() - window)
        end = min(len(generated), m.end() + window)
        ctx = generated[start:end].replace("\n", " ")
        flagged.append({
            "report_id": report_id,
            "token": tok,
            "position": m.start(),
            "context": f"...{ctx}...",
        })
    return flagged

_NLI_PIPE = None


def get_nli_pipe():
    """Lazy-load roberta-large-mnli. ~1.4GB download on first run."""
    global _NLI_PIPE
    if _NLI_PIPE is not None:
        return _NLI_PIPE
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logging.error("transformers + torch required for NLI. "
                      "pip install transformers torch")
        return None
    device = 0 if torch.cuda.is_available() else -1
    logging.info("Loading NLI model %s on %s...", NLI_MODEL_NAME,
                 "GPU" if device == 0 else "CPU")
    _NLI_PIPE = pipeline("text-classification", model=NLI_MODEL_NAME,
                         device=device, top_k=None)
    return _NLI_PIPE


def md_table_to_premise(md_table: str) -> str:
    """Convert a markdown table into NLI-friendly prose.

    NLI models are trained on natural language, not markdown. We flatten each
    data row into a sentence using the header names as field labels.
    """
    lines = [l.strip() for l in md_table.strip().splitlines() if l.strip()]
    rows = [l for l in lines if l.startswith("|")]
    if len(rows) < 3:
        return md_table

    def split_row(r):
        parts = [c.strip() for c in r.strip("|").split("|")]
        return parts

    headers = split_row(rows[0])
    data = [split_row(r) for r in rows[2:] if "---" not in r]

    sentences = []
    for row in data:
        if len(row) != len(headers):
            continue
        parts = [f"{h} is {v}" for h, v in zip(headers, row) if v and v != "-"]
        sentences.append(". ".join(parts) + ".")
    return " ".join(sentences)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_into_sentences(text: str):
    text = re.sub(r"^[#*\-]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


def nli_score_report(text: str, premise: str, report_id: str):
    """Run NLI on every sentence of `text` against `premise`.

    Returns (list of per-sentence dicts, summary dict).
    """
    pipe = get_nli_pipe()
    if pipe is None:
        return [], {"entailed": 0, "neutral": 0, "contradicted": 0,
                    "entailment_rate": float("nan")}

    sents = split_into_sentences(text)
    if not sents:
        return [], {"entailed": 0, "neutral": 0, "contradicted": 0,
                    "entailment_rate": float("nan")}
    inputs = [{"text": premise, "text_pair": s} for s in sents]
    rows = []
    counts = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
    for i in range(0, len(inputs), NLI_BATCH_SIZE):
        batch = inputs[i:i+NLI_BATCH_SIZE]
        out = pipe(batch, truncation=True, max_length=512)
        for inp, scores in zip(batch, out):
            best = max(scores, key=lambda d: d["score"])
            label = best["label"]
            counts[label] = counts.get(label, 0) + 1
            rows.append({
                "report_id": report_id,
                "sentence": inp["text_pair"],
                "label": label.lower(),
                "score": round(best["score"], 4),
            })

    ent = counts["ENTAILMENT"]
    con = counts["CONTRADICTION"]
    denom = ent + con
    rate = ent / denom if denom > 0 else float("nan")
    summary = {
        "entailed": counts["ENTAILMENT"],
        "neutral": counts["NEUTRAL"],
        "contradicted": counts["CONTRADICTION"],
        "entailment_rate": round(rate, 4) if denom > 0 else float("nan"),
    }
    return rows, summary
def load_gemini_key():
    if GEMINI_API_KEY_OVERRIDE:
        return GEMINI_API_KEY_OVERRIDE
    return os.getenv("GEMINI_API_KEY")


def gemini_call(client, prompt: str, system: str, temperature: float = 0.0):
    try:
        from google.genai.errors import APIError
    except ImportError:
        from google.api_core.exceptions import GoogleAPIError as APIError

    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_VERSION,
                contents=[prompt],
                config={"system_instruction": system, "temperature": temperature},
            )
            return resp.text
        except APIError as e:
            msg = getattr(e, "message", str(e))
            code = getattr(e, "code", None)
            if code == 429 and attempt < 3:
                wait = 30 * (attempt + 1)
                logging.warning("Gemini 429, waiting %ds (attempt %d/4)", wait, attempt + 1)
                time.sleep(wait)
                continue
            return f"[Gemini API error: {msg}]"
        except Exception as e:
            return f"[Unexpected error: {e}]"
    return "[Gemini API error: exhausted retries]"


def ollama_call(prompt: str, system: str, temperature: float = 0.0):
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"[Ollama error: {e}]"


DECOMPOSE_SYSTEM = (
    "You are a careful analyst. Your job is to decompose a report into atomic "
    "factual claims. An atomic claim is a single, indivisible statement that "
    "can be evaluated as true or false against source data. "
    "Each claim should be self-contained and contain at most one fact."
)

DECOMPOSE_PROMPT = """Decompose the following climate risk report into atomic claims.

Rules:
- Each claim must be a single, self-contained, evaluable statement.
- Skip rhetorical sentences, transitions, and pure recommendations.
- Quote specific values (HI, EI, VI, PRI scores) verbatim if mentioned.
- Output ONLY a JSON array of strings, nothing else. No markdown, no preamble.

REPORT:
{report}

JSON array of atomic claims:"""


JUDGE_SYSTEM = (
    "You are a strict fact-checker. You verify each claim against ONLY the "
    "source table provided. You do not use outside knowledge. If a claim "
    "cannot be confirmed or refuted by the table alone, label it 'unverifiable'."
)

JUDGE_PROMPT = """SOURCE TABLE (the only ground truth):
{table}

CLAIM TO VERIFY:
{claim}

Classify the claim into exactly one category:
- supported: every fact in the claim is directly stated or trivially derivable from the table
- partial: part of the claim is supported but part adds detail not in the table
- contradicted: the claim asserts something the table contradicts
- unverifiable: the claim is about something the table does not cover (style, recommendations, general knowledge)

Respond with ONLY a JSON object like:
{{"verdict": "supported", "reason": "Row 3 shows HI=2 for landslides"}}

JSON:"""


def parse_json_loose(text: str):
    """Strip markdown fences and parse JSON. Returns None on failure."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    starts = [i for i in [s.find("["), s.find("{")] if i != -1]
    ends   = [i for i in [s.rfind("]"), s.rfind("}")] if i != -1]
    if not starts or not ends:
        return None
    s = s[min(starts):max(ends) + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def judge_report(report_text: str, table_md: str, report_id: str,
                 judge_model: str, gemini_client=None):
    """Decompose report into claims, then judge each claim against the table.

    Returns (list of per-claim dicts, summary dict).
    """
    call = (lambda p, s, t=0.0: gemini_call(gemini_client, p, s, t)) \
        if judge_model == "gemini" \
        else ollama_call
    raw = call(DECOMPOSE_PROMPT.format(report=report_text), DECOMPOSE_SYSTEM, 0.0)
    if judge_model == "gemini":
        time.sleep(GEMINI_PACING_SEC)
    claims = parse_json_loose(raw)
    if not isinstance(claims, list):
        logging.warning("[%s] claim decomposition failed; raw=%r", report_id, raw[:200])
        return [], {"n_claims": 0, "supported": 0, "partial": 0,
                    "contradicted": 0, "unverifiable": 0,
                    "support_rate": float("nan")}
    claims = [c for c in claims if isinstance(c, str) and len(c.split()) >= 3]
    rows = []
    counts = {"supported": 0, "partial": 0, "contradicted": 0, "unverifiable": 0}
    for claim in claims:
        raw = call(JUDGE_PROMPT.format(table=table_md, claim=claim), JUDGE_SYSTEM, 0.0)
        if judge_model == "gemini":
            time.sleep(GEMINI_PACING_SEC)
        verdict_obj = parse_json_loose(raw)
        if not isinstance(verdict_obj, dict) or "verdict" not in verdict_obj:
            verdict = "parse_error"
            reason = (raw or "")[:200]
        else:
            verdict = str(verdict_obj.get("verdict", "")).lower().strip()
            reason = str(verdict_obj.get("reason", ""))[:300]
        if verdict in counts:
            counts[verdict] += 1
        rows.append({
            "report_id": report_id,
            "claim": claim,
            "verdict": verdict,
            "reason": reason,
        })

    total = sum(counts.values())
    summary = {
        "n_claims": total,
        **counts,
        "support_rate": round(counts["supported"] / total, 4) if total else float("nan"),
    }
    return rows, summary
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nli", action="store_true",
                    help="Skip NLI metric (avoids transformers/torch)")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip LLM-judge metric (avoids API calls)")
    ap.add_argument("--judge-model", choices=["gemini", "llama"], default="gemini",
                    help="Which model to use as judge (default: gemini)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N reports (for debugging)")
    args = ap.parse_args()

    setup_logging(OUTPUT_DIR / "rescore_log.txt")
    logging.info("=" * 70)
    logging.info("Faithfulness re-scoring run %s", datetime.now().isoformat())
    logging.info("skip_nli=%s skip_judge=%s judge_model=%s",
                 args.skip_nli, args.skip_judge, args.judge_model)
    logging.info("=" * 70)
    hazard_md, pri_md, hazard_ref, pri_ref = load_inputs(INPUT_DIR)
    reports = load_existing_reports(OUTPUT_DIR / "reports")
    if args.limit:
        reports = reports[:args.limit]
        logging.info("Limited to first %d reports", len(reports))
    allowed_text = hazard_md + "\n" + pri_md
    nli_premises = {
        "hazard": md_table_to_premise(hazard_md),
        "pri":    md_table_to_premise(pri_md),
    }
    if not args.skip_nli:
        logging.info("Hazard NLI premise (%d chars): %s...",
                     len(nli_premises["hazard"]), nli_premises["hazard"][:200])
        logging.info("PRI NLI premise (%d chars): %s...",
                     len(nli_premises["pri"]), nli_premises["pri"][:200])

    judge_tables = {"hazard": hazard_md, "pri": pri_md}
    gemini_client = None
    if not args.skip_judge and args.judge_model == "gemini":
        key = load_gemini_key()
        if not key:
            logging.error("GEMINI_API_KEY not found")
            sys.exit(1)
        from google import genai
        gemini_client = genai.Client(api_key=key)
        logging.info("Gemini judge client initialised")
    elif not args.skip_judge and args.judge_model == "llama":
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
            if r.status_code != 200:
                logging.error("Ollama not available at %s", OLLAMA_BASE)
                sys.exit(1)
        except Exception:
            logging.error("Ollama not available at %s", OLLAMA_BASE)
            sys.exit(1)
        logging.info("Llama judge ready")

    summary_rows  = []
    flagged_rows  = []
    nli_rows      = []
    claim_rows    = []
    summary_csv = OUTPUT_DIR / "rescore_per_report.csv"
    done_ids = set()
    if summary_csv.exists():
        try:
            prev = pd.read_csv(summary_csv)
            required_cols = ["numeric_hallucinations_improved"]
            if not args.skip_nli:
                required_cols.append("nli_entailment_rate")
            if not args.skip_judge:
                required_cols.append("judge_support_rate")
            mask = prev[required_cols].notna().all(axis=1)
            done_ids = set(prev.loc[mask, "report_id"].tolist())
            summary_rows = prev.to_dict("records")
            logging.info("RESUME: %d reports already fully scored; will skip",
                         len(done_ids))

            for sidecar, accum in [
                ("rescore_flagged_numbers.csv", flagged_rows),
                ("rescore_nli_sentences.csv",   nli_rows),
                ("rescore_claims.csv",          claim_rows),
            ]:
                p = OUTPUT_DIR / sidecar
                if p.exists():
                    accum.extend(pd.read_csv(p).to_dict("records"))
                    logging.info("RESUME: loaded %d rows from %s",
                                 len(accum), sidecar)
        except Exception as e:
            logging.warning("Could not resume from existing CSVs: %s", e)
            summary_rows = []
            done_ids = set()

    for i, r in enumerate(reports, 1):
        rid    = r["report_id"]
        rtype  = r["type"]
        text   = r["text"]

        if rid in done_ids:
            logging.info("\n[%d/%d] %s — SKIP (already scored)", i, len(reports), rid)
            continue

        logging.info("\n[%d/%d] %s (%s, %s, %s, rep%d)",
                     i, len(reports), rid, rtype, r["condition"],
                     r["model"], r["repetition"])

        flagged = numeric_hallucinations_with_context(text, allowed_text, rid)
        flagged_rows.extend(flagged)
        n_flagged = len(flagged)
        logging.info("  Numeric hallucinations (improved): %d", n_flagged)
        if flagged:
            for f in flagged[:3]:
                logging.info("    flagged: %r in '%s'", f["token"], f["context"][:80])
        if args.skip_nli:
            nli_summary = {"entailed": float("nan"), "neutral": float("nan"),
                           "contradicted": float("nan"),
                           "entailment_rate": float("nan")}
        else:
            sent_rows, nli_summary = nli_score_report(
                text, nli_premises[rtype], rid)
            nli_rows.extend(sent_rows)
            logging.info("  NLI: entailed=%s contradicted=%s entailment_rate=%s",
                         nli_summary["entailed"], nli_summary["contradicted"],
                         nli_summary["entailment_rate"])

        if args.skip_judge:
            judge_summary = {"n_claims": float("nan"), "supported": float("nan"),
                             "partial": float("nan"), "contradicted": float("nan"),
                             "unverifiable": float("nan"),
                             "support_rate": float("nan")}
        else:
            claim_recs, judge_summary = judge_report(
                text, judge_tables[rtype], rid, args.judge_model, gemini_client)
            claim_rows.extend(claim_recs)
            logging.info("  Judge: %d claims, supported=%s support_rate=%s",
                         judge_summary["n_claims"], judge_summary["supported"],
                         judge_summary["support_rate"])

        summary_rows.append({
            "report_id":  rid,
            "type":       rtype,
            "condition":  r["condition"],
            "repetition": r["repetition"],
            "model":      r["model"],
            "n_words":    len(text.split()),
            "numeric_hallucinations_improved":  n_flagged,
            "nli_entailed":           nli_summary["entailed"],
            "nli_neutral":            nli_summary["neutral"],
            "nli_contradicted":       nli_summary["contradicted"],
            "nli_entailment_rate":    nli_summary["entailment_rate"],
            "judge_n_claims":         judge_summary["n_claims"],
            "judge_supported":        judge_summary["supported"],
            "judge_partial":          judge_summary["partial"],
            "judge_contradicted":     judge_summary["contradicted"],
            "judge_unverifiable":     judge_summary["unverifiable"],
            "judge_support_rate":     judge_summary["support_rate"],
        })

        if i % 4 == 0 or i == len(reports):
            pd.DataFrame(summary_rows).to_csv(
                OUTPUT_DIR / "rescore_per_report.csv", index=False)
            if flagged_rows:
                pd.DataFrame(flagged_rows).to_csv(
                    OUTPUT_DIR / "rescore_flagged_numbers.csv", index=False)
            if nli_rows:
                pd.DataFrame(nli_rows).to_csv(
                    OUTPUT_DIR / "rescore_nli_sentences.csv", index=False)
            if claim_rows:
                pd.DataFrame(claim_rows).to_csv(
                    OUTPUT_DIR / "rescore_claims.csv", index=False)
    df = pd.DataFrame(summary_rows)
    agg = df.groupby(["type", "condition", "model"]).agg(
        n=("report_id", "count"),
        num_halluc_mean=("numeric_hallucinations_improved", "mean"),
        num_halluc_sum=("numeric_hallucinations_improved", "sum"),
        nli_ent_rate_mean=("nli_entailment_rate", "mean"),
        nli_contradicted_mean=("nli_contradicted", "mean"),
        judge_support_rate_mean=("judge_support_rate", "mean"),
        judge_contradicted_mean=("judge_contradicted", "mean"),
    ).round(3).reset_index()
    agg.to_csv(OUTPUT_DIR / "rescore_summary_by_condition.csv", index=False)

    logging.info("\n%s", "=" * 70)
    logging.info("AGGREGATE RESULTS (by type x condition x model)")
    logging.info("%s", "=" * 70)
    logging.info("\n%s", agg.to_string(index=False))
    logging.info("\nOutputs in %s:", OUTPUT_DIR)
    logging.info("  rescore_per_report.csv             — one row per report")
    logging.info("  rescore_summary_by_condition.csv   — aggregated for paper")
    logging.info("  rescore_flagged_numbers.csv        — every flagged number with context")
    logging.info("  rescore_nli_sentences.csv          — per-sentence NLI verdicts")
    logging.info("  rescore_claims.csv                 — per-claim judge verdicts")
    logging.info("DONE %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
