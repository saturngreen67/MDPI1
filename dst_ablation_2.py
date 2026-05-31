
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

import pandas as pd

INPUT_DIR  = Path(r"D:\DST_Results\ablation\inputs")
OUTPUT_DIR = Path(r"D:\DST_Results\ablation")

HAZARD_TABLE_FILE = "Hazard index.md"
PRI_TABLE_FILE    = "PRI.md"
HAZARD_REF_FILE   = "Hazard report.txt"
PRI_REF_FILE      = "PRI report.txt"

OPENAI_API_KEY_OVERRIDE = "xxxxxxxxxxxxxxxx"
OPENAI_MODEL  = "gpt-5.4-mini"

OPENAI_PACING_SEC = 1

NLI_MODEL_NAME = "roberta-large-mnli"
NLI_BATCH_SIZE = 8
N_REPETITIONS  = 3
MIN_VALID_WORDS = 80
HAZARD_TEMP = 0.7
PRI_TEMP    = 0.3
JUDGE_TEMP  = 0.0


EXAMPLE_HAZARD_TABLE = """
| Infrastructure | Climate driver | Impact model | Hazard Index | Hazard Level |
| :--- | :--- | :--- | :--- | :--- |
| Railway | Heat wave | Track buckling and equipment failure due to extreme heat | 5 | Extreme |
| Railway | Droughts | Soil desiccation affecting embankment stability | 5 | Extreme |
| Railway | Water stress | Reduced water availability for maintenance and cleaning | 4 | Very High |
| Railway | Changing temperature (Chronic) | Thermal expansion of rail components over time | 2 | Medium |
| Railway | Heavy precipitation | Flash flooding affecting track drainage systems | 1 | Low |
| Railway | Landslide | Slope instability triggered by saturation | 2 | Medium |
| Railway | Changing wind patterns | Crosswinds affecting train stability | 0 | No variation |
"""

EXAMPLE_HAZARD_REPORT = """
It can be concluded that droughts and heat waves constitute the most critical climate hazards for the analyzed railway line, consistently registering EXTREME hazard index scores (Hazard Index: 5). This underscores their high likelihood and severity, reflecting the growing influence of temperature-related acute events on the structural and operational integrity of the infrastructure.

In addition, water stress emerges as a significant chronic hazard, with VERY HIGH hazard levels (Index: 4). This indicates potential challenges in water availability for maintenance operations, ecological balance, and indirect effects such as soil desiccation and vegetation loss on embankments.

Conversely, chronic temperature changes, such as mean annual temperature rise, register MEDIUM scores (Index: 2). While their direct impacts may be moderate, they may act cumulatively or serve as enabling conditions for more severe events (e.g., prolonged high temperatures amplifying drought effects).

Precipitation-related hazards, including heavy rainfall, present LOW hazard levels (Index: 1). Notably, there is no significant variation projected in mean annual rainfall, which suggests limited change in total precipitation but potential shifts in intensity. This may still pose operational risks if drainage systems are overwhelmed.

Landslides, classified as acute and linked to precipitation, show MEDIUM scores (Index: 2). Despite their moderate hazard classification, their localized impact potential is high, especially in mountainous terrain with critical slope infrastructure.

Finally, wind-related hazards are generally assessed as showing NO VARIATION (Index: 0). Although currently not prioritized, these hazards should be monitored as part of a comprehensive risk strategy.

In summary, while droughts, heat waves, and water stress represent the most immediate and severe threats, the analysis highlights the importance of considering the full spectrum of hazards, including those with medium or low scores, as their risk contribution is ultimately shaped by the exposure and vulnerability profile of the infrastructure.
"""

EXAMPLE_PRI_TABLE = """
| Infrastructure | Climate driver | Impact model | Hazard Index | Exposure Index | Vulnerability Index | PRI scores |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Railway | Heavy precipitation | Reactive CAPEX due to damages associated to heavy rains | 2 | 4 | 4 | 2 |
| Railway | Storm (winds) | Reactive CAPEX due to damages associated to strong winds | 1 | 4 | 3 | 1 |
| Railway | Landslides | Reactive CAPEX due to damages associated to landslides | 2 | 4 | 5 | 2 |
| Railway | Changing precipitation | Increased maintenance due to increase in precipitation | 0 | 4 | 3 | 0 |
| Railway | Temperature variability | Increased maintenance due to temperature increase | 1 | 4 | 3 | 1 |
| Railway | Heavy precipitation | Stop of operations due to heavy rain | 2 | 4 | 3 | 1 |
"""

EXAMPLE_PRI_REPORT = """
**Potential Risk Index (PRI) Assessment Report**

The analysis integrates the Hazard Index (HI), Exposure Index (EI), and Vulnerability Index (VI) to compute the Potential Risk Index (PRI) for the analyzed infrastructure assets.

**High and Moderate Risks (PRI 2)**
The results indicate that the highest computed risk levels (PRI = 2) are driven by **Heavy Precipitation** and **Landslides**.
* **Drivers:** These risks are characterized by a moderate Hazard Index (HI: 2) combined with high Exposure (EI: 4) and high Vulnerability (VI: 4-5).
* **Consequences:** The impacts include reactive CAPEX due to structural damages and potential operational stoppages.

**Low Risks (PRI 1)**
**Temperature variability** and **Storms (winds)** present a low risk (PRI = 1). While the Exposure is high (EI: 4), the Hazard Index is low (HI: 1), limiting the overall risk score.

**No Risk (PRI 0)**
Impacts related to general **Changing precipitation patterns** (chronic changes) registered a PRI of 0.

**Conclusion**
"""

HAZARD_SYSTEM = (
    "You are an expert climate hazard analyst. Generate a professional report "
    "interpreting a climate hazard table.\n"
    "STRICT PROTOCOLS: 1. Zero Hallucination — base analysis EXCLUSIVELY on "
    "provided table. 2. Output Format — narrative Markdown. DO NOT reconstruct "
    "the table. 3. Follow the depth and structure of the provided example.")

PRI_SYSTEM = (
    "You are a senior infrastructure risk analyst. Write a formal PRI Assessment "
    "Report.\nSTRICT PROTOCOLS: 1. Zero Hallucination. 2. Narrative Markdown "
    "output. 3. Use abbreviations HI, EI, VI, PRI. "
    "4. PRI is derived from HI x EI x VI, normalised to a 0-4 scale. "
    "5. EI = 3 is a conservative default applied uniformly due to data unavailability "
    "where no economic asset valuation is present. "
    "6. A dash (- or --) in HI means no relevant hazard variation exists for that "
    "driver; treat as not applicable and do not speculate.")

_DASH_NOTE = (
    "Note: '-' or '--' in Hazard Index means no relevant hazard variation "
    "for that driver -- classify as Not applicable, do not speculate.\n\n")

def hazard_prompt(table_md: str, with_example: bool) -> str:
    if with_example:
        return f"""
### REFERENCE EXAMPLE
{EXAMPLE_HAZARD_TABLE}
{EXAMPLE_HAZARD_REPORT}

### ACTUAL TASK
{_DASH_NOTE}Hazard Data Table:
{table_md}

INSTRUCTIONS: Narrative Markdown. Group hazards logically. Mention Hazard Level and Index exactly as in data.
"""
    return f"""
### TASK
{_DASH_NOTE}Hazard Data Table:
{table_md}

INSTRUCTIONS: Narrative Markdown. Group hazards logically. Mention Hazard Level and Index exactly as in data.
"""


def pri_prompt(table_md: str, with_example: bool) -> str:
    if with_example:
        return f"""
### REFERENCE EXAMPLE
{EXAMPLE_PRI_TABLE}
{EXAMPLE_PRI_REPORT}

### ACTUAL TASK
PRI Data Table:
{table_md}

INSTRUCTIONS: Narrative Markdown. Categorise risks highest first. Explain drivers (HI, EI, VI). No speculation on engineering strategies.
"""
    return f"""
### TASK
PRI Data Table:
{table_md}

INSTRUCTIONS: Narrative Markdown. Categorise risks highest first. Explain drivers (HI, EI, VI). No speculation on engineering strategies.
"""

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
    return (read(HAZARD_TABLE_FILE), read(PRI_TABLE_FILE),
            read(HAZARD_REF_FILE, required=False), read(PRI_REF_FILE, required=False))

def get_openai_client():
    key = OPENAI_API_KEY_OVERRIDE or os.getenv("OPENAI_API_KEY")
    if not key:
        logging.error("No OpenAI API key. Set OPENAI_API_KEY env var or "
                      "OPENAI_API_KEY_OVERRIDE in the script.")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        logging.error("openai package required: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=key)


def openai_chat(client, system: str, user: str, temperature: float):
    """Call GPT-5.4 mini with retry on rate-limit, and graceful fallback if
    the model rejects a custom temperature."""
    from openai import APIError, RateLimitError, APIStatusError

    def _do(use_temp):
        kwargs = dict(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if use_temp:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    for attempt in range(5):
        try:
            return _do(use_temp=True)
        except (RateLimitError,) as e:
            wait = 20 * (attempt + 1)
            logging.warning("OpenAI rate limit, waiting %ds (attempt %d/5)", wait, attempt + 1)
            time.sleep(wait)
        except (APIStatusError, APIError) as e:
            msg = str(e)
            if "temperature" in msg.lower() and ("unsupported" in msg.lower()
                                                  or "does not support" in msg.lower()
                                                  or "only the default" in msg.lower()):
                logging.warning("Model rejected custom temperature; retrying with default.")
                try:
                    return _do(use_temp=False)
                except Exception as e2:
                    return f"[OpenAI error: {e2}]"
            if attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            return f"[OpenAI error: {msg}]"
        except Exception as e:
            return f"[Unexpected error: {e}]"
    return "[OpenAI error: exhausted retries]"

NUMERIC_RE_FULL = re.compile(r"\b\d+(?:[,]\d{3})*(?:\.\d+)?\b")
SCALE_OK = {"0", "1", "2", "3", "4", "5"}


def numeric_hallucinations_with_context(generated, allowed_text, report_id, window=40):
    allowed_nums = set(NUMERIC_RE_FULL.findall(allowed_text))
    flagged = []
    for m in NUMERIC_RE_FULL.finditer(generated):
        tok = m.group(0)
        if tok in SCALE_OK or tok in allowed_nums:
            continue
        s = max(0, m.start() - window)
        e = min(len(generated), m.end() + window)
        flagged.append({"report_id": report_id, "token": tok, "position": m.start(),
                        "context": f"...{generated[s:e]}...".replace("\n", " ")})
    return flagged

_NLI_PIPE = None


def get_nli_pipe():
    global _NLI_PIPE
    if _NLI_PIPE is not None:
        return _NLI_PIPE
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logging.error("transformers + torch required for NLI.")
        return None
    device = 0 if torch.cuda.is_available() else -1
    logging.info("Loading NLI model %s on %s...", NLI_MODEL_NAME, "GPU" if device == 0 else "CPU")
    _NLI_PIPE = pipeline("text-classification", model=NLI_MODEL_NAME, device=device, top_k=None)
    return _NLI_PIPE


def md_table_to_premise(md_table: str) -> str:
    lines = [l.strip() for l in md_table.strip().splitlines() if l.strip()]
    rows = [l for l in lines if l.startswith("|")]
    if len(rows) < 3:
        return md_table

    def split_row(r):
        return [c.strip() for c in r.strip("|").split("|")]

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
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.split()) >= 4]


def nli_score_report(text, premise, report_id):
    pipe = get_nli_pipe()
    if pipe is None:
        return [], {"entailed": 0, "neutral": 0, "contradicted": 0, "entailment_rate": float("nan")}
    sents = split_into_sentences(text)
    if not sents:
        return [], {"entailed": 0, "neutral": 0, "contradicted": 0, "entailment_rate": float("nan")}
    inputs = [{"text": premise, "text_pair": s} for s in sents]
    rows, counts = [], {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
    for i in range(0, len(inputs), NLI_BATCH_SIZE):
        batch = inputs[i:i + NLI_BATCH_SIZE]
        out = pipe(batch, truncation=True, max_length=512)
        for inp, scores in zip(batch, out):
            best = max(scores, key=lambda d: d["score"])
            counts[best["label"]] = counts.get(best["label"], 0) + 1
            rows.append({"report_id": report_id, "sentence": inp["text_pair"],
                         "label": best["label"].lower(), "score": round(best["score"], 4)})
    ent, con = counts["ENTAILMENT"], counts["CONTRADICTION"]
    denom = ent + con
    return rows, {"entailed": ent, "neutral": counts["NEUTRAL"], "contradicted": con,
                  "entailment_rate": round(ent / denom, 4) if denom else float("nan")}

DECOMPOSE_SYSTEM = (
    "You are a careful analyst. Your job is to decompose a report into atomic "
    "factual claims. An atomic claim is a single, indivisible statement that "
    "can be evaluated as true or false against source data. "
    "Each claim should be self-contained and contain at most one fact.")

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
    "cannot be confirmed or refuted by the table alone, label it 'unverifiable'.")

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
    if not text:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", text.strip())
    s = re.sub(r"\s*```\s*$", "", s)
    starts = [i for i in [s.find("["), s.find("{")] if i != -1]
    ends = [i for i in [s.rfind("]"), s.rfind("}")] if i != -1]
    if not starts or not ends:
        return None
    try:
        return json.loads(s[min(starts):max(ends) + 1])
    except json.JSONDecodeError:
        return None


def judge_report(client, report_text, table_md, report_id):
    raw = openai_chat(client, DECOMPOSE_SYSTEM,
                      DECOMPOSE_PROMPT.format(report=report_text), JUDGE_TEMP)
    time.sleep(OPENAI_PACING_SEC)
    claims = parse_json_loose(raw)
    if not isinstance(claims, list):
        logging.warning("[%s] decomposition failed; raw=%r", report_id, (raw or "")[:160])
        return [], {"n_claims": 0, "supported": 0, "partial": 0,
                    "contradicted": 0, "unverifiable": 0, "support_rate": float("nan")}
    claims = [c for c in claims if isinstance(c, str) and len(c.split()) >= 3]

    rows, counts = [], {"supported": 0, "partial": 0, "contradicted": 0, "unverifiable": 0}
    for claim in claims:
        raw = openai_chat(client, JUDGE_SYSTEM,
                          JUDGE_PROMPT.format(table=table_md, claim=claim), JUDGE_TEMP)
        time.sleep(OPENAI_PACING_SEC)
        obj = parse_json_loose(raw)
        if not isinstance(obj, dict) or "verdict" not in obj:
            verdict, reason = "parse_error", (raw or "")[:200]
        else:
            verdict = str(obj.get("verdict", "")).lower().strip()
            reason = str(obj.get("reason", ""))[:300]
        if verdict in counts:
            counts[verdict] += 1
        rows.append({"report_id": report_id, "claim": claim,
                     "verdict": verdict, "reason": reason})
    total = sum(counts.values())
    return rows, {"n_claims": total, **counts,
                  "support_rate": round(counts["supported"] / total, 4) if total else float("nan")}

def generate_reports(client, hazard_md, pri_md, reports_dir):
    """Generate the 12 GPT-5.4-mini reports, saving each to disk. Resumable:
    skips any report already present and valid."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("hazard", "gpt_full",      True,  HAZARD_SYSTEM, HAZARD_TEMP, hazard_md, hazard_prompt),
        ("hazard", "gpt_noexample", False, HAZARD_SYSTEM, HAZARD_TEMP, hazard_md, hazard_prompt),
        ("pri",    "gpt_full",      True,  PRI_SYSTEM,    PRI_TEMP,    pri_md,    pri_prompt),
        ("pri",    "gpt_noexample", False, PRI_SYSTEM,    PRI_TEMP,    pri_md,    pri_prompt),
    ]
    produced = []
    for rtype, cond, with_ex, system, temp, table, builder in specs:
        for rep in range(1, N_REPETITIONS + 1):
            tag = f"{rtype}_{cond}_rep{rep}"
            path = reports_dir / f"{tag}.md"
            if path.exists():
                txt = path.read_text(encoding="utf-8")
                if len(txt.split()) >= MIN_VALID_WORDS:
                    logging.info("SKIP generation (exists): %s", tag)
                    produced.append((tag, rtype, cond, rep, path, txt))
                    continue
            logging.info("Generating %s ...", tag)
            prompt = builder(table, with_ex)
            text = openai_chat(client, system, prompt, temp)
            time.sleep(OPENAI_PACING_SEC)
            if len(text.split()) < MIN_VALID_WORDS or text.startswith("["):
                logging.warning("  invalid/short response for %s: %r", tag, text[:120])
                (reports_dir / f"{tag}.last_failure.log").write_text(text, encoding="utf-8")
                continue
            path.write_text(text, encoding="utf-8")
            produced.append((tag, rtype, cond, rep, path, text))
    return produced


def load_existing_reports(reports_dir: Path):
    """For --judge-existing: load the 24 Gemini/Llama reports already on disk."""
    pattern = re.compile(r"^(hazard|pri)_(full|noexample|llama_full|llama_noexample)_rep(\d+)\.md$")
    rows = []
    for f in sorted(reports_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        rtype, cond, rep = m.group(1), m.group(2), int(m.group(3))
        text = f.read_text(encoding="utf-8")
        if len(text.split()) < 40:
            continue
        rows.append((f.stem, rtype, cond, rep, f, text))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-existing", action="store_true",
                    help="Skip generation; re-judge the existing 24 reports with GPT-5.4 mini")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    suffix = "gpt54mini_crossjudge" if args.judge_existing else "gpt54mini"
    setup_logging(OUTPUT_DIR / f"rescore_log_{suffix}.txt")
    logging.info("=" * 70)
    logging.info("GPT-5.4 mini arm — %s — %s", suffix, datetime.now().isoformat())
    logging.info("=" * 70)

    hazard_md, pri_md, hazard_ref, pri_ref = load_inputs(INPUT_DIR)
    allowed_text = hazard_md + "\n" + pri_md
    nli_premises = {"hazard": md_table_to_premise(hazard_md),
                    "pri":    md_table_to_premise(pri_md)}
    judge_tables = {"hazard": hazard_md, "pri": pri_md}

    client = None
    need_client = (not args.judge_existing) or (not args.skip_judge)
    if need_client:
        client = get_openai_client()
    reports_dir = OUTPUT_DIR / "reports"
    if args.judge_existing:
        reports = load_existing_reports(reports_dir)
        logging.info("Cross-judge mode: loaded %d existing reports", len(reports))
    else:
        reports = generate_reports(client, hazard_md, pri_md, reports_dir)
        logging.info("Generation mode: %d GPT reports available", len(reports))

    if args.limit:
        reports = reports[:args.limit]

    summary_rows, flagged_rows, nli_rows, claim_rows = [], [], [], []

    for i, (rid, rtype, cond, rep, path, text) in enumerate(reports, 1):
        model = ("gpt54mini" if cond.startswith("gpt")
                 else "llama" if cond.startswith("llama") else "gemini")
        logging.info("\n[%d/%d] %s (%s, %s, %s)", i, len(reports), rid, rtype, cond, model)

        flagged = numeric_hallucinations_with_context(text, allowed_text, rid)
        flagged_rows.extend(flagged)
        n_flagged = len(flagged)
        logging.info("  Numeric hallucinations: %d", n_flagged)

        if args.skip_nli:
            nli_sum = {k: float("nan") for k in
                       ("entailed", "neutral", "contradicted", "entailment_rate")}
        else:
            srows, nli_sum = nli_score_report(text, nli_premises[rtype], rid)
            nli_rows.extend(srows)
            logging.info("  NLI: ent=%s con=%s rate=%s",
                         nli_sum["entailed"], nli_sum["contradicted"], nli_sum["entailment_rate"])

        if args.skip_judge:
            jsum = {k: float("nan") for k in
                    ("n_claims", "supported", "partial", "contradicted",
                     "unverifiable", "support_rate")}
        else:
            crows, jsum = judge_report(client, text, judge_tables[rtype], rid)
            claim_rows.extend(crows)
            logging.info("  Judge(GPT-5.4 mini): %s claims, support_rate=%s",
                         jsum["n_claims"], jsum["support_rate"])

        summary_rows.append({
            "report_id": rid, "type": rtype, "condition": cond,
            "repetition": rep, "model": model, "judge_model": "gpt54mini",
            "n_words": len(text.split()),
            "numeric_hallucinations_improved": n_flagged,
            "nli_entailed": nli_sum["entailed"], "nli_neutral": nli_sum["neutral"],
            "nli_contradicted": nli_sum["contradicted"],
            "nli_entailment_rate": nli_sum["entailment_rate"],
            "judge_n_claims": jsum["n_claims"], "judge_supported": jsum["supported"],
            "judge_partial": jsum["partial"], "judge_contradicted": jsum["contradicted"],
            "judge_unverifiable": jsum["unverifiable"],
            "judge_support_rate": jsum["support_rate"],
        })

        if i % 2 == 0 or i == len(reports):
            pd.DataFrame(summary_rows).to_csv(
                OUTPUT_DIR / f"rescore_per_report_{suffix}.csv", index=False)
            if flagged_rows:
                pd.DataFrame(flagged_rows).to_csv(
                    OUTPUT_DIR / f"rescore_flagged_numbers_{suffix}.csv", index=False)
            if nli_rows:
                pd.DataFrame(nli_rows).to_csv(
                    OUTPUT_DIR / f"rescore_nli_sentences_{suffix}.csv", index=False)
            if claim_rows:
                pd.DataFrame(claim_rows).to_csv(
                    OUTPUT_DIR / f"rescore_claims_{suffix}.csv", index=False)

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        df["verifiable_total"] = (df["judge_supported"] + df["judge_partial"]
                                  + df["judge_contradicted"])
        df["support_rate_verifiable"] = df["judge_supported"] / df["verifiable_total"]
        agg = df.groupby(["type", "condition", "model"]).agg(
            n=("report_id", "count"),
            num_halluc_mean=("numeric_hallucinations_improved", "mean"),
            nli_ent_rate_mean=("nli_entailment_rate", "mean"),
            judge_support_rate_mean=("judge_support_rate", "mean"),
            judge_support_rate_verifiable_mean=("support_rate_verifiable", "mean"),
            judge_unverifiable_mean=("judge_unverifiable", "mean"),
        ).round(3).reset_index()
        agg.to_csv(OUTPUT_DIR / f"rescore_summary_by_condition_{suffix}.csv", index=False)
        logging.info("\n%s\n%s", "=" * 70, agg.to_string(index=False))

    logging.info("\nDONE %s — outputs suffixed _%s in %s", datetime.now().isoformat(),
                 suffix, OUTPUT_DIR)


if __name__ == "__main__":
    main()
