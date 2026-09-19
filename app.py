import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from langchain_groq import ChatGroq


# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="E-commerce AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path("data/processed/ecommerce_clean.db").resolve()
CODEBOOK_PATH = Path("ecommerce_agent_codebook.md")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

if not CODEBOOK_PATH.exists():
    st.error(f"Codebook not found: {CODEBOOK_PATH}")
    st.stop()

codebook_text = CODEBOOK_PATH.read_text(encoding="utf-8")


# ============================================================
# 2. SQL SAFETY
# ============================================================

FORBIDDEN_SQL = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
    "PRAGMA",
}


def validate_readonly_sql(sql: str):
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL query is empty."

    cleaned = sql.strip()
    first_word = cleaned.split()[0].upper()

    if first_word not in {"SELECT", "WITH"}:
        return False, "Only SELECT or WITH queries are allowed."

    for keyword in FORBIDDEN_SQL:
        if re.search(
            rf"\b{keyword}\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return False, f"Forbidden SQL keyword: {keyword}"

    without_last_semicolon = cleaned.rstrip(";")

    if ";" in without_last_semicolon:
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"


def validate_sql_query(sql: str) -> dict:
    """
    Deterministic validator.
    It does NOT decide what the insight/paradox should be.
    """
    safe, reason = validate_readonly_sql(sql)

    if not safe:
        return {
            "status": "BLOCKED",
            "messages": [reason],
        }

    normalized = " ".join(sql.lower().split())

    # Compile against the actual SQLite database without executing.
    try:
        db_uri = f"file:{DB_PATH}?mode=ro"

        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.execute(
                "EXPLAIN QUERY PLAN " + sql
            ).fetchall()

    except Exception as e:
        return {
            "status": "BLOCKED",
            "messages": [
                f"SQL does not compile: {type(e).__name__}: {e}"
            ],
        }

    warnings = []

    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)",
        normalized,
    ):
        warnings.append(
            "SUM(price) is the sum of price on retained order-item rows, "
            "not guaranteed complete original-order revenue."
        )

    if re.search(
        r"\b(sum|avg)\s*\(\s*(?:\w+\.)?payment_value\s*\)",
        normalized,
    ):
        warnings.append(
            "payment_value comes from one retained payment row per order; "
            "do not automatically interpret it as complete original-order revenue."
        )

    if (
        "product_category_name" in normalized
        and "payment_value" in normalized
    ):
        warnings.append(
            "Category × payment_value is high-risk because category belongs "
            "to the retained product row while payment_value belongs to the "
            "retained payment row."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "order_items" in normalized
    ):
        warnings.append(
            "Counting order_items rows does not recover the original item count; "
            "this transformed table contains one retained row per order."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "payments" in normalized
    ):
        warnings.append(
            "Counting payments rows does not recover the original number of "
            "payment transactions; one retained payment row remains per order."
        )

    if (
        "order_delivered_timestamp" in normalized
        and "is not null" not in normalized
    ):
        warnings.append(
            "Delivery analysis references order_delivered_timestamp without "
            "explicitly requiring a non-NULL actual-delivery timestamp."
        )

    if warnings:
        return {
            "status": "WARNING",
            "messages": warnings,
        }

    return {
        "status": "SAFE",
        "messages": ["SQL passed validation."],
    }


def execute_sql_dataframe(
    sql: str,
    max_rows: int = 200,
):
    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        raise ValueError(
            "SQL BLOCKED: "
            + " ".join(validation["messages"])
        )

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
    ) as conn:
        df = pd.read_sql_query(
            sql,
            conn,
        )

    return df.head(max_rows), validation


# ============================================================
# 3. DATABASE CONTEXT
# ============================================================

@st.cache_data(show_spinner=False)
def build_schema_context() -> str:
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
    ) as conn:
        rows = conn.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name;
            """
        ).fetchall()

    parts = []

    for name, kind, sql in rows:
        parts.append(
            f"{kind.upper()}: {name}\n{sql}"
        )

    return "\n\n".join(parts)


SCHEMA_CONTEXT = build_schema_context()


# ============================================================
# 4. LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0,
    )


def llm_text(prompt: str) -> str:
    """
    Plain model call with NO tool binding.
    This avoids malformed tool-call JSON from the provider.
    """
    llm = get_llm()
    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content.strip()

    # Some providers may return structured content blocks.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(
                        str(item["text"])
                    )
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts).strip()

    return str(content).strip()


def extract_json_payload(text: str):
    """
    Parse JSON robustly from:
    - raw JSON
    - ```json ... ```
    - prose followed by a JSON object/array
    """
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = text.strip()

    # Fenced JSON
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        candidate = fenced.group(1).strip()

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Raw JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # First object/array span.
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if (
        object_start != -1
        and object_end > object_start
    ):
        candidate = cleaned[
            object_start:object_end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")

    if (
        array_start != -1
        and array_end > array_start
    ):
        candidate = cleaned[
            array_start:array_end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(
        "Could not parse valid JSON from LLM response."
    )


def llm_json(
    prompt: str,
    required_key: Optional[str] = None,
    retries: int = 2,
):
    """
    Ask the model for JSON using ordinary text generation.

    Important:
    - No function/tool calling is used.
    - If the first response is malformed, a second small repair call
      receives the malformed text and converts it to valid JSON only.
    """
    last_error = None
    raw = ""

    for attempt in range(retries + 1):
        try:
            if attempt == 0:
                raw = llm_text(prompt)
            else:
                repair_prompt = f"""
Repair the following malformed model output into VALID JSON only.

MALFORMED OUTPUT:
{raw}

Rules:
- Preserve the intended information.
- Return JSON only.
- No markdown fences.
- No commentary before or after JSON.
"""
                raw = llm_text(repair_prompt)

            data = extract_json_payload(raw)

            if (
                required_key is not None
                and not (
                    isinstance(data, dict)
                    and required_key in data
                )
            ):
                raise ValueError(
                    f"Missing required JSON key: {required_key}"
                )

            return data

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"LLM JSON generation failed after repair attempts: {last_error}"
    )


# ============================================================
# 5. COMMON HELPERS
# ============================================================

def format_history_for_prompt(
    history,
    max_messages=8,
):
    if not history:
        return "(no prior conversation)"

    lines = []

    for msg in history[-max_messages:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(lines)


def same_language_instruction(question: str):
    return (
        "Respond in the same language as the user's question."
    )


def normalize_sql_plan_item(
    item: Dict[str, Any],
    index: int,
):
    normalized = {
        "id": str(
            item.get("id", f"A{index}")
        ),
        "title": str(
            item.get(
                "title",
                f"Analysis {index}",
            )
        ),
        "reason": str(
            item.get("reason", "")
        ),
        "sql": str(
            item.get("sql", "")
        ).strip(),
    }

    sql_lower = normalized["sql"].lower()

    def clean_label(text: str, metric_phrase: str):
        # Replace risky generic labels, then collapse accidental duplication.
        text = re.sub(
            r"\brevenue\b",
            metric_phrase,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bsales\b",
            metric_phrase,
            text,
            flags=re.IGNORECASE,
        )

        duplicate = f"{metric_phrase} {metric_phrase}"
        while duplicate.lower() in text.lower():
            text = re.sub(
                re.escape(duplicate),
                metric_phrase,
                text,
                flags=re.IGNORECASE,
            )

        # Common awkward mixed forms.
        text = re.sub(
            r"retained item-price retained item-price total",
            "retained item-price total",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"retained payment-value retained payment-value total",
            "retained payment-value total",
            text,
            flags=re.IGNORECASE,
        )

        return text

    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)",
        normalized["sql"],
        flags=re.IGNORECASE,
    ):
        normalized["title"] = clean_label(
            normalized["title"],
            "retained item-price total",
        )
        normalized["reason"] = clean_label(
            normalized["reason"],
            "retained item-price total",
        )

    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?payment_value\s*\)",
        normalized["sql"],
        flags=re.IGNORECASE,
    ):
        normalized["title"] = clean_label(
            normalized["title"],
            "retained payment-value total",
        )
        normalized["reason"] = clean_label(
            normalized["reason"],
            "retained payment-value total",
        )

    return normalized


def run_sql_plan(
    plan_items: List[Dict[str, Any]],
    max_queries: int = 6,
):
    results = []

    for index, item in enumerate(
        plan_items[:max_queries],
        start=1,
    ):
        normalized = normalize_sql_plan_item(
            item,
            index,
        )

        sql = normalized["sql"]

        validation = validate_sql_query(sql)

        result = {
            **normalized,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
            "row_count_returned": 0,
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(
                    sql,
                    max_rows=200,
                )

                result["records"] = (
                    df.to_dict(
                        orient="records"
                    )
                )
                result["columns"] = (
                    df.columns.tolist()
                )
                result[
                    "row_count_returned"
                ] = len(df)

            except Exception as e:
                result["status"] = "ERROR"
                result["validation_messages"].append(
                    f"{type(e).__name__}: {e}"
                )

        results.append(result)

    return results


def compact_evidence(
    analyses,
    max_rows_per_query=20,
):
    compact = []

    for item in analyses:
        compact.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "reason": item.get("reason"),
            "status": item.get("status"),
            "validation_messages": item.get(
                "validation_messages",
                [],
            ),
            "sql": item.get("sql"),
            "rows": item.get(
                "records",
                [],
            )[:max_rows_per_query],
        })

    return compact


def clean_string_list(value):
    if not isinstance(value, list):
        return []

    cleaned = []

    for item in value:
        if item is None:
            continue

        text = str(item).strip()

        if text:
            cleaned.append(text)

    return cleaned


def empty_strategy():
    return {
        "short_term": [],
        "medium_term": [],
        "long_term": [],
    }


def normalize_strategy(value):
    if not isinstance(value, dict):
        return empty_strategy()

    return {
        "short_term": clean_string_list(
            value.get("short_term", [])
        ),
        "medium_term": clean_string_list(
            value.get("medium_term", [])
        ),
        "long_term": clean_string_list(
            value.get("long_term", [])
        ),
    }


def no_chart():
    return {
        "type": "none",
        "x": None,
        "y": None,
        "title": None,
        "source_analysis_id": None,
    }


# ============================================================
# 6. STAGE 1 — ANALYST / PRIMARY SQL DISCOVERY
# ============================================================

def parse_tagged_analyses(text: str):
    """
    Parse:
    <ANALYSIS>
    <ID>A1</ID>
    <TITLE>...</TITLE>
    <REASON>...</REASON>
    <SQL>...</SQL>
    </ANALYSIS>
    """
    analyses = []

    blocks = re.findall(
        r"<ANALYSIS>(.*?)</ANALYSIS>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for index, block in enumerate(
        blocks[:4],
        start=1,
    ):
        def tag(name):
            match = re.search(
                rf"<{name}>(.*?)</{name}>",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )

            return (
                match.group(1).strip()
                if match
                else ""
            )

        sql = tag("SQL")

        if not sql:
            continue

        analyses.append({
            "id": tag("ID") or f"A{index}",
            "title": tag("TITLE") or f"Analysis {index}",
            "reason": tag("REASON"),
            "sql": sql,
        })

    return analyses


def plan_primary_analyses(
    question: str,
    history,
):
    """
    Stage 1 Analyst.

    Uses tagged text rather than JSON/function-calling because some Groq
    generations intermittently fail strict JSON formatting even when the
    analytical content itself is valid.
    """
    history_text = format_history_for_prompt(
        history
    )

    prompt = f"""
You are Stage 1: ANALYST in an AI Data Analyst system.

Your task is NOT to answer the user yet.
Your task is to decide what SQL evidence is needed.

USER QUESTION:
{question}

RECENT CONVERSATION:
{history_text}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK / SEMANTIC RULES:
{codebook_text}

RULES:
- Use only the actual schema above.
- Generate only read-only SELECT or WITH SQL.
- Never use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA.
- Every numerical claim later must come from SQL evidence.
- Respect transformed-data limitations in the codebook.
- Do not use unsupported basket-size, repeat-customer, original item-count,
  or original payment-transaction-count metrics.
- Never assume a currency.
- If you use SUM(price), call it "retained item-price total".
  NEVER call it revenue or sales.
- If you use SUM(payment_value), call it "retained payment-value total".
  NEVER call it guaranteed complete revenue.
- Do not describe payment_type counts as customer preference.
- For a narrow factual question, create 1-3 focused analyses.
- For an open-ended business/strategy/insight question, create EXACTLY 4
  complementary analyses chosen by YOU from the available schema.
- Choose analyses because they help answer THIS user's question.
- Do not pre-program a business story or paradox.
- Keep each SQL query reasonably compact.

For an open-ended business question, try to cover complementary dimensions
when supported by the schema, such as time, product/category, geography,
fulfillment/delivery, or payment structure. YOU decide which dimensions are
most useful.

Return ONLY this tagged format.
Do not use markdown fences.
Do not write anything before or after the tags.

<ANALYSIS>
<ID>A1</ID>
<TITLE>short descriptive title</TITLE>
<REASON>why this evidence is useful</REASON>
<SQL>SELECT ...</SQL>
</ANALYSIS>

<ANALYSIS>
<ID>A2</ID>
<TITLE>short descriptive title</TITLE>
<REASON>why this evidence is useful</REASON>
<SQL>SELECT ...</SQL>
</ANALYSIS>

Continue until the required number of analyses is provided.
"""

    raw = llm_text(prompt)

    analyses = parse_tagged_analyses(
        raw
    )

    # One small formatting-repair call only if the first response could
    # not be parsed. This does not change the analytical plan.
    if not analyses:
        repair_prompt = f"""
Reformat the following Analyst response into <ANALYSIS> blocks.

Preserve the intended analyses and SQL.
Do not invent a new business story.
Do not use markdown.
Return only tagged blocks.

ORIGINAL RESPONSE:
{raw}

Required block format:

<ANALYSIS>
<ID>A1</ID>
<TITLE>...</TITLE>
<REASON>...</REASON>
<SQL>SELECT ...</SQL>
</ANALYSIS>
"""

        repaired = llm_text(
            repair_prompt
        )

        analyses = parse_tagged_analyses(
            repaired
        )

    if not analyses:
        raise RuntimeError(
            "Analyst response could not be parsed into tagged SQL analyses."
        )

    return analyses


# ============================================================
# 7. STAGE 2 — PARADOX HUNTER
# ============================================================

def discover_paradox_candidates(
    question: str,
    primary_results,
    prior_attempts=None,
    round_number: int = 1,
):
    """
    Discovery only.
    The model proposes falsifiable hypotheses, but DOES NOT write SQL here.
    This keeps the creative task separate from SQL engineering.
    """
    evidence = compact_evidence(
        primary_results,
        max_rows_per_query=18,
    )

    prior_attempts = prior_attempts or []

    prompt = f"""
You are Stage 2A: PARADOX HUNTER, discovery round {round_number}.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

PRIOR ATTEMPTS:
{json.dumps(prior_attempts, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK:
{codebook_text}

Your job is to propose FALSIFIABLE candidate paradoxes.

A candidate does NOT need to be true. It is a hypothesis to test.

Look for structures such as:
- aggregate success but subgroup weakness,
- rank reversal across two valid metrics,
- high volume paired with weak service performance,
- a segment that breaks an aggregate pattern,
- time-period reversal,
- concentration in one dimension but dispersion in another,
- an operational trade-off visible in the available variables.

STRICT RULES:
- Candidates must arise from THIS database/evidence.
- Do not repeat failed prior attempts.
- Do not manufacture facts that are not yet observed.
- Phrase each candidate as a testable hypothesis.
- Explain exactly what result would verify it.
- Respect transformed-data limitations.
- Never call SUM(price) revenue or sales.
- Do not interpret payment_type frequency as customer preference.

For an open-ended business request, propose 3-5 candidates.
For a narrow request, propose 1-3 relevant candidates.
If absolutely no meaningful candidate can be formulated from the available
schema, return an empty list — but do not choose empty merely because a
candidate is uncertain. Uncertainty is the reason we verify it.

Return ONLY valid JSON:

{{
  "candidates": [
    {{
      "hypothesis": "testable candidate paradox",
      "why_surprising": "why it would be counter-intuitive if true",
      "verification_logic": "the exact comparison/rank/reversal the SQL must establish"
    }}
  ]
}}

No markdown.
No prose outside JSON.
"""

    data = llm_json(
        prompt,
        required_key="candidates",
        retries=2,
    )

    candidates = data.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidates,
        list,
    ):
        return []

    normalized = []

    for index, candidate in enumerate(
        candidates[:5],
        start=1,
    ):
        hypothesis = str(
            candidate.get(
                "hypothesis",
                "",
            )
        ).strip()

        if not hypothesis:
            continue

        normalized.append({
            "id": f"R{round_number}P{index}",
            "hypothesis": hypothesis,
            "why_surprising": str(
                candidate.get(
                    "why_surprising",
                    "",
                )
            ).strip(),
            "verification_logic": str(
                candidate.get(
                    "verification_logic",
                    "",
                )
            ).strip(),
            "round_number": round_number,
        })

    return normalized


def plan_paradox_verification_sql(
    question: str,
    candidate,
    primary_results,
):
    """
    Separate SQL-planning call for one paradox candidate.
    The model is not asked to re-invent the paradox here.
    """
    evidence = compact_evidence(
        primary_results,
        max_rows_per_query=12,
    )

    prompt = f"""
You are Stage 2B: SQL VERIFICATION PLANNER.

USER QUESTION:
{question}

PARADOX CANDIDATE:
{json.dumps(candidate, ensure_ascii=False, default=str)}

PRIMARY SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK:
{codebook_text}

Write ONE read-only SQLite query that can directly verify or falsify the
candidate's verification_logic.

STRICT SQL CONTRACT:
1. SELECT or WITH only.
2. Include the focal entity/group in the returned rows.
3. Include the comparison group(s) needed by the hypothesis.
4. Return BOTH sides of every claimed contrast.
5. If hypothesis says "top X by metric A", construct the top-X set in a
   CTE/subquery, then evaluate metric B inside exactly that set.
6. If hypothesis says highest/lowest/rank reversal, return enough rows or
   explicit ranks to establish that claim.
7. Do not use a LIMIT that could accidentally remove the focal entity.
8. Respect transformed-data limitations.
9. Never call SUM(price) revenue/sales.
10. Prefer a single compact result table that the Judge can read directly.

Return ONLY valid JSON:

{{
  "verification_sql": "WITH ... SELECT ..."
}}

No markdown.
No prose outside JSON.
"""

    data = llm_json(
        prompt,
        required_key="verification_sql",
        retries=2,
    )

    return str(
        data.get(
            "verification_sql",
            "",
        )
    ).strip()


def repair_paradox_sql(
    question: str,
    candidate,
    failed_sql: str,
    validation_messages,
):
    prompt = f"""
You are repairing a SQLite verification query.

USER QUESTION:
{question}

PARADOX CANDIDATE:
{json.dumps(candidate, ensure_ascii=False, default=str)}

FAILED SQL:
{failed_sql}

VALIDATOR FEEDBACK:
{json.dumps(validation_messages, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK:
{codebook_text}

Return a corrected query that tests the SAME candidate.
Do not change the hypothesis.

Rules:
- SELECT/WITH only.
- Fix all validator/schema issues.
- Preserve the candidate's comparison set and verification logic.
- Return all rows/metrics needed to falsify or support the claim.

Return ONLY valid JSON:
{{"verification_sql":"..."}}
"""

    data = llm_json(
        prompt,
        required_key="verification_sql",
        retries=1,
    )

    return str(
        data.get(
            "verification_sql",
            "",
        )
    ).strip()


def verify_paradox_candidates(
    question: str,
    candidates,
    primary_results,
):
    verified = []

    for candidate in candidates[:5]:
        try:
            sql = plan_paradox_verification_sql(
                question=question,
                candidate=candidate,
                primary_results=primary_results,
            )
        except Exception as e:
            verified.append({
                **candidate,
                "verification_sql": "",
                "status": "ERROR",
                "validation_messages": [
                    f"SQL planner failed: {type(e).__name__}: {e}"
                ],
                "records": [],
                "columns": [],
            })
            continue

        validation = validate_sql_query(sql)

        # One autonomous repair attempt if SQL is structurally invalid.
        if validation["status"] == "BLOCKED":
            try:
                repaired_sql = repair_paradox_sql(
                    question=question,
                    candidate=candidate,
                    failed_sql=sql,
                    validation_messages=validation["messages"],
                )

                repaired_validation = validate_sql_query(
                    repaired_sql
                )

                if repaired_validation["status"] != "BLOCKED":
                    sql = repaired_sql
                    validation = repaired_validation

            except Exception:
                pass

        item = {
            **candidate,
            "verification_sql": sql,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(
                    sql,
                    max_rows=150,
                )

                item["records"] = (
                    df.to_dict(
                        orient="records"
                    )
                )
                item["columns"] = (
                    df.columns.tolist()
                )

            except Exception as e:
                item["status"] = "ERROR"
                item[
                    "validation_messages"
                ].append(
                    f"{type(e).__name__}: {e}"
                )

        verified.append(item)

    return verified



# ============================================================
# 7C. FAST PARADOX TEST PLANNER (v7)
# ============================================================

def parse_tagged_paradox_tests(text: str):
    """
    Parse:
    <TEST>
    <ID>T1</ID>
    <QUESTION>...</QUESTION>
    <WHY>...</WHY>
    <SQL>...</SQL>
    </TEST>
    """
    tests = []

    blocks = re.findall(
        r"<TEST>(.*?)</TEST>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for index, block in enumerate(blocks[:3], start=1):
        def tag(name):
            match = re.search(
                rf"<{name}>(.*?)</{name}>",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            return (
                match.group(1).strip()
                if match
                else ""
            )

        sql = tag("SQL")

        if not sql:
            continue

        tests.append({
            "id": tag("ID") or f"T{index}",
            "hypothesis": tag("QUESTION"),
            "why_surprising": tag("WHY"),
            "verification_logic": (
                "Judge the returned comparison table and report only "
                "a contrast/reversal/tension directly supported by its rows."
            ),
            "verification_sql": sql,
            "round_number": 1,
        })

    return tests


def plan_fast_paradox_tests(
    question: str,
    primary_results,
):
    """
    One LLM call proposes exactly 3 exploratory paradox tests + SQL.
    It uses tagged text rather than JSON to avoid provider JSON-format failures.
    """
    evidence = compact_evidence(
        primary_results,
        max_rows_per_query=10,
    )

    prompt = f"""
You are the PARADOX HUNTER in an AI Data Analyst system.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATA LIMITATIONS:
- orders: one row per order.
- order_items: one retained row per order.
- payments: one retained payment row per order.
- products: one metadata row per product_id.
- customers: transformed unique customer rows.
- SUM(price) = retained item-price total only.
- SUM(payment_value) = retained payment-value total only.
- Do not infer original basket size, repeat purchase, profitability,
  customer preference, demand, or causality.
- Never invent currency.

TASK:
Create EXACTLY 3 exploratory, falsifiable paradox tests unless the schema
literally makes three distinct comparisons impossible.

IMPORTANT:
- Do NOT assert a specific entity is paradoxical before seeing verification rows.
- Tests should be broad enough to let the data reveal WHICH group/time/category
  is surprising.
- Each test must return the full comparison table needed to find a reversal,
  tension, subgroup exception, or rank mismatch.
- Prefer tests such as:
  * operational performance vs activity/volume across groups,
  * rank comparison under two legitimate metrics,
  * aggregate pattern vs subgroup pattern,
  * time-period activity vs service outcome,
  * concentration vs performance.
- These are structures only; YOU choose the actual tests from this database.
- SQL must be SELECT/WITH only.
- Avoid LIMIT that could hide the comparison unless the universe is first
  explicitly defined in a CTE.
- Do not use SUM(price) or SUM(payment_value) as complete revenue.

Return EXACTLY this tagged format, with no markdown fences and no text outside it:

<TEST>
<ID>T1</ID>
<QUESTION>broad testable business question</QUESTION>
<WHY>why a surprising result would matter</WHY>
<SQL>WITH ... SELECT ...</SQL>
</TEST>
<TEST>
<ID>T2</ID>
<QUESTION>broad testable business question</QUESTION>
<WHY>why a surprising result would matter</WHY>
<SQL>SELECT ...</SQL>
</TEST>
<TEST>
<ID>T3</ID>
<QUESTION>broad testable business question</QUESTION>
<WHY>why a surprising result would matter</WHY>
<SQL>SELECT ...</SQL>
</TEST>
"""

    raw = llm_text(prompt)
    tests = parse_tagged_paradox_tests(raw)

    # One lightweight format-repair retry only if parsing failed.
    if not tests:
        repair_prompt = f"""
Reformat the following response into EXACTLY 3 <TEST> blocks.
Preserve its intended hypotheses and SQL where possible.
Do not add markdown or commentary.

ORIGINAL RESPONSE:
{raw}

Required format:
<TEST>
<ID>T1</ID>
<QUESTION>...</QUESTION>
<WHY>...</WHY>
<SQL>...</SQL>
</TEST>
<TEST>
<ID>T2</ID>
<QUESTION>...</QUESTION>
<WHY>...</WHY>
<SQL>...</SQL>
</TEST>
<TEST>
<ID>T3</ID>
<QUESTION>...</QUESTION>
<WHY>...</WHY>
<SQL>...</SQL>
</TEST>
"""
        repaired = llm_text(repair_prompt)
        tests = parse_tagged_paradox_tests(
            repaired
        )

    return tests[:3]


def execute_fast_paradox_tests(
    tests,
):
    """
    Python only validates and executes the agent-generated SQL.
    It does not decide what the paradox is.
    """
    results = []

    for item in tests[:3]:
        sql = item.get(
            "verification_sql",
            "",
        )

        validation = validate_sql_query(sql)

        result = {
            **item,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(
                    sql,
                    max_rows=120,
                )

                result["records"] = (
                    df.to_dict(
                        orient="records"
                    )
                )
                result["columns"] = (
                    df.columns.tolist()
                )

            except Exception as e:
                result["status"] = "ERROR"
                result[
                    "validation_messages"
                ].append(
                    f"{type(e).__name__}: {e}"
                )

        results.append(result)

    return results


# ============================================================
# 7D. FAST FINAL JUDGE + STRATEGIST (v7)
# ============================================================

def parse_items_from_block(
    text: str,
    block_name: str,
):
    block_match = re.search(
        rf"<{block_name}>(.*?)</{block_name}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not block_match:
        return []

    return [
        item.strip()
        for item in re.findall(
            r"<ITEM>(.*?)</ITEM>",
            block_match.group(1),
            flags=re.DOTALL | re.IGNORECASE,
        )
        if item.strip()
    ]


def parse_final_tagged_report(
    text: str,
):
    def single_tag(name):
        match = re.search(
            rf"<{name}>(.*?)</{name}>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return (
            match.group(1).strip()
            if match
            else ""
        )

    answer = single_tag("ANSWER")

    basic_insights = parse_items_from_block(
        text,
        "BASIC_INSIGHTS",
    )

    paradoxical_insights = parse_items_from_block(
        text,
        "PARADOXICAL_INSIGHTS",
    )

    short_term = parse_items_from_block(
        text,
        "SHORT_TERM",
    )

    medium_term = parse_items_from_block(
        text,
        "MEDIUM_TERM",
    )

    long_term = parse_items_from_block(
        text,
        "LONG_TERM",
    )

    limitations = parse_items_from_block(
        text,
        "LIMITATIONS",
    )

    judgments = []

    judgments_block = re.search(
        r"<JUDGMENTS>(.*?)</JUDGMENTS>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if judgments_block:
        judgment_blocks = re.findall(
            r"<JUDGMENT>(.*?)</JUDGMENT>",
            judgments_block.group(1),
            flags=re.DOTALL | re.IGNORECASE,
        )

        for block in judgment_blocks:
            def local_tag(name):
                match = re.search(
                    rf"<{name}>(.*?)</{name}>",
                    block,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                return (
                    match.group(1).strip()
                    if match
                    else ""
                )

            supported_text = (
                local_tag("SUPPORTED")
                .strip()
                .lower()
            )

            judgments.append({
                "id": local_tag("ID"),
                "supported": (
                    supported_text
                    in {"true", "yes", "1"}
                ),
                "insight": local_tag("INSIGHT"),
                "reason": local_tag("REASON"),
            })

    if not answer:
        raise ValueError(
            "Missing <ANSWER> in final report."
        )

    return {
        "answer": answer,
        "basic_insights": basic_insights,
        "paradoxical_insights": paradoxical_insights,
        "strategy": {
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
        },
        "limitations": limitations,
        "judgments": judgments,
        "chart": no_chart(),
    }


def fast_final_report(
    question: str,
    primary_results,
    paradox_results,
    chart_requested: bool,
):
    """
    One final call using tagged text rather than JSON.
    It judges tests + writes insights + strategy.
    """
    primary_evidence = compact_evidence(
        primary_results,
        max_rows_per_query=8,
    )

    paradox_evidence = []

    for item in paradox_results:
        paradox_evidence.append({
            "id": item.get("id"),
            "test_question": item.get("hypothesis"),
            "why_interesting": item.get("why_surprising"),
            "status": item.get("status"),
            "validation_messages": item.get(
                "validation_messages",
                [],
            ),
            "sql": item.get("verification_sql"),
            "rows": item.get(
                "records",
                [],
            )[:25],
        })

    prompt = f"""
You are the FINAL DATA JUDGE AND BUSINESS STRATEGIST.

USER QUESTION:
{question}

PRIMARY EXECUTED SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

PARADOX TESTS + EXECUTED RESULTS:
{json.dumps(paradox_evidence, ensure_ascii=False, default=str)}

SEMANTIC RULES:
- SUM(price) = retained item-price total only, never complete revenue/sales.
- SUM(payment_value) = retained payment-value total only, never complete revenue.
- Payment-method frequency is NOT customer preference.
- Geographic totals alone do not prove market opportunity.
- Product/category totals alone do not prove demand or profitability.
- Do not infer causality.
- Never invent numbers or currency.

TASK 1 — BASIC INSIGHTS
Produce 2-4 useful direct insights supported by primary SQL evidence.

TASK 2 — JUDGE EVERY PARADOX TEST
A test is SUPPORTED only if its returned rows contain a genuinely surprising
contrast/reversal/tension/subgroup exception.
A test can reveal a surprising pattern different from the direction initially
imagined, but the pattern must be directly visible in those rows.
If evidence is weak, ambiguous, or ordinary, mark false.

TASK 3 — PARADOXICAL INSIGHTS
Include ONLY supported judgments.
If none are supported, use one item saying no defensible paradox was verified.

TASK 4 — BUSINESS STRATEGY
Return short-, medium-, and long-term actions.
Important:
- Do NOT recommend increasing inventory, advertising, expansion, pricing changes,
  promotions, loyalty programs, or major resource reallocation solely from an
  aggregate ranking.
- When commercial action needs missing information, explicitly request evidence
  such as profitability, stockouts, conversion, cost-to-serve, payment failure,
  or inventory availability.
- Operational actions can be proposed when directly supported by delivery or
  status evidence.

TASK 5 — EXECUTIVE ANSWER
Concise answer in the user's language.

TASK 6 — LIMITATIONS
Only material limitations.

Return ONLY the following tagged format.
Do not use markdown fences.
Do not write anything outside these tags.

<ANSWER>
...
</ANSWER>

<BASIC_INSIGHTS>
<ITEM>...</ITEM>
<ITEM>...</ITEM>
</BASIC_INSIGHTS>

<JUDGMENTS>
<JUDGMENT>
<ID>T1</ID>
<SUPPORTED>true</SUPPORTED>
<INSIGHT>...</INSIGHT>
<REASON>...</REASON>
</JUDGMENT>
<JUDGMENT>
<ID>T2</ID>
<SUPPORTED>false</SUPPORTED>
<INSIGHT></INSIGHT>
<REASON>...</REASON>
</JUDGMENT>
</JUDGMENTS>

<PARADOXICAL_INSIGHTS>
<ITEM>...</ITEM>
</PARADOXICAL_INSIGHTS>

<SHORT_TERM>
<ITEM>...</ITEM>
</SHORT_TERM>

<MEDIUM_TERM>
<ITEM>...</ITEM>
</MEDIUM_TERM>

<LONG_TERM>
<ITEM>...</ITEM>
</LONG_TERM>

<LIMITATIONS>
<ITEM>...</ITEM>
</LIMITATIONS>
"""

    raw = llm_text(prompt)

    try:
        report = parse_final_tagged_report(
            raw
        )

    except Exception:
        # One repair call, preserving content.
        repair_prompt = f"""
Reformat the following analyst response into the exact tagged report format.
Preserve its factual conclusions and judgments.
Do not add new facts.
Do not use markdown.
Return tags only.

ORIGINAL:
{raw}

Required tags:
<ANSWER>...</ANSWER>
<BASIC_INSIGHTS><ITEM>...</ITEM></BASIC_INSIGHTS>
<JUDGMENTS>
<JUDGMENT>
<ID>T1</ID>
<SUPPORTED>true|false</SUPPORTED>
<INSIGHT>...</INSIGHT>
<REASON>...</REASON>
</JUDGMENT>
</JUDGMENTS>
<PARADOXICAL_INSIGHTS><ITEM>...</ITEM></PARADOXICAL_INSIGHTS>
<SHORT_TERM><ITEM>...</ITEM></SHORT_TERM>
<MEDIUM_TERM><ITEM>...</ITEM></MEDIUM_TERM>
<LONG_TERM><ITEM>...</ITEM></LONG_TERM>
<LIMITATIONS><ITEM>...</ITEM></LIMITATIONS>
"""
        repaired = llm_text(
            repair_prompt
        )
        report = parse_final_tagged_report(
            repaired
        )

    # The UI can still use deterministic chart fallback when requested.
    report["chart"] = no_chart()

    return report


# ============================================================
# 8. STAGE 3A — PARADOX JUDGE
# ============================================================

def judge_paradoxes(
    question: str,
    primary_results,
    paradox_results,
):
    if not paradox_results:
        return []

    prompt = f"""
You are Stage 3A: PARADOX JUDGE.

USER QUESTION:
{question}

PRIMARY EVIDENCE:
{json.dumps(compact_evidence(primary_results, 12), ensure_ascii=False, default=str)}

PARADOX CANDIDATES + EXECUTED VERIFICATION RESULTS:
{json.dumps(paradox_results, ensure_ascii=False, default=str)}

Judge each candidate strictly.

A candidate is SUPPORTED only if the returned SQL rows directly demonstrate
the candidate's own verification_logic and hypothesis.

Rules:
- Use only executed SQL rows shown above.
- A plausible story is not enough.
- If the focal entity is absent, supported=false.
- If the comparison set does not match the hypothesis, supported=false.
- If a rank/lowest/highest claim cannot be established from returned rows,
  supported=false.
- If evidence is mixed or ambiguous, supported=false.
- Never invent missing numbers.
- Respect transformed-data limitations.
- If supported, write one concise paradoxical insight in the user's language,
  including the concrete contrast that makes it surprising.

Return ONLY valid JSON:

{{
  "judgments": [
    {{
      "id": "P1",
      "supported": true,
      "insight": "verified paradoxical insight, or empty string",
      "reason": "why the executed rows do or do not verify it"
    }}
  ]
}}

No markdown.
No prose outside JSON.
"""

    data = llm_json(
        prompt,
        required_key="judgments",
    )

    judgments = data.get(
        "judgments",
        [],
    )

    if not isinstance(
        judgments,
        list,
    ):
        return []

    return judgments


# ============================================================
# 9. STAGE 3B — BUSINESS STRATEGIST / FINAL SYNTHESIS
# ============================================================

def synthesize_report(
    question: str,
    primary_results,
    paradox_results,
    paradox_judgments,
    chart_requested: bool,
):
    """
    Final synthesis deliberately uses several SMALL model calls rather than one
    giant structured JSON response. A formatting failure in one component will
    no longer destroy the entire report.
    """
    primary_evidence = compact_evidence(
        primary_results,
        max_rows_per_query=18,
    )

    judgment_map = {
        str(item.get("id")): item
        for item in paradox_judgments
    }

    supported_paradoxes = []

    for item in paradox_results:
        candidate_id = str(item.get("id"))
        judgment = judgment_map.get(candidate_id, {})

        if judgment.get("supported") is True:
            supported_paradoxes.append({
                "id": candidate_id,
                "hypothesis": item.get("hypothesis", ""),
                "verification_logic": item.get("verification_logic", ""),
                "verification_rows": item.get("records", [])[:20],
                "verified_insight": judgment.get("insight", ""),
                "judgment_reason": judgment.get("reason", ""),
            })

    # ---------------------------
    # Executive answer: plain text
    # ---------------------------
    answer_prompt = f"""
You are the final Business Strategist.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

VERIFIED PARADOXICAL INSIGHTS:
{json.dumps(supported_paradoxes, ensure_ascii=False, default=str)}

DATABASE CODEBOOK:
{codebook_text}

Write a concise executive answer in the user's language.

Rules:
- Use only executed SQL evidence above.
- Never invent numbers or currency.
- Never claim causality.
- Never call SUM(price) revenue/sales; call it retained item-price total.
- Never call payment_type distribution customer preference.
- Respect transformed-data limitations.
- 1-2 short paragraphs only.
"""
    try:
        answer = llm_text(answer_prompt)
    except Exception:
        answer = "Primary SQL evidence was collected successfully. See the evidence and verified paradox sections below."

    # ---------------------------
    # Basic insights: small JSON
    # ---------------------------
    basic_prompt = f"""
USER QUESTION:
{question}

EXECUTED PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

Return 1-4 BASIC business insights directly supported by this SQL evidence.

Rules:
- Same language as the user.
- Include concrete values when present.
- No invented numbers.
- No causality.
- Never call SUM(price) revenue/sales; call it retained item-price total.
- Never call payment_type counts customer preference.

Return ONLY:
{{"basic_insights":["..."]}}
"""
    try:
        basic_data = llm_json(
            basic_prompt,
            required_key="basic_insights",
            retries=2,
        )
        basic_insights = clean_string_list(
            basic_data.get("basic_insights", [])
        )
    except Exception:
        basic_insights = []

    # ---------------------------
    # Paradox insight: use judge output DIRECTLY
    # ---------------------------
    paradoxical_insights = [
        item.get("verified_insight", "")
        for item in supported_paradoxes
        if item.get("verified_insight")
    ]

    if not paradoxical_insights:
        paradoxical_insights = [
            "Paradox Hunter đã kiểm tra các candidate trong lượt này nhưng chưa xác minh được một insight nghịch lý đủ chắc bằng SQL."
        ]

    # ---------------------------
    # Strategy: small JSON
    # ---------------------------
    strategy_prompt = f"""
USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

VERIFIED PARADOXICAL INSIGHTS:
{json.dumps(supported_paradoxes, ensure_ascii=False, default=str)}

Generate evidence-linked business strategies.

Rules:
- Same language as the user.
- short_term: 1-3 immediate operational actions.
- medium_term: 1-3 process/analysis/resource-allocation actions.
- long_term: 1-3 structural strategic actions.
- If evidence is insufficient for an action, recommend what to measure next.
- Do not invent profitability, demand, customer preference, or causality.
- Never call SUM(price) revenue/sales.

Return ONLY:
{{
  "strategy": {{
    "short_term": ["..."],
    "medium_term": ["..."],
    "long_term": ["..."]
  }}
}}
"""
    try:
        strategy_data = llm_json(
            strategy_prompt,
            required_key="strategy",
            retries=2,
        )
        strategy = normalize_strategy(
            strategy_data.get(
                "strategy",
                empty_strategy(),
            )
        )
    except Exception:
        strategy = empty_strategy()

    # ---------------------------
    # Limitations: small JSON
    # ---------------------------
    limitation_prompt = f"""
DATABASE CODEBOOK:
{codebook_text}

PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

Return only the material interpretation limitations relevant to these analyses.
Same language as the user.
Maximum 4 items.

Return ONLY:
{{"limitations":["..."]}}
"""
    try:
        limitation_data = llm_json(
            limitation_prompt,
            required_key="limitations",
            retries=1,
        )
        limitations = clean_string_list(
            limitation_data.get(
                "limitations",
                [],
            )
        )
    except Exception:
        limitations = []

    # ---------------------------
    # Chart: only if explicitly requested
    # ---------------------------
    chart = no_chart()

    if chart_requested:
        chart_prompt = f"""
USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

Choose one useful chart ONLY from the primary evidence.

Rules:
- source_analysis_id must match an existing A-id.
- x/y must exactly match returned SQL columns.
- type must be bar, line, scatter, pie, or none.

Return ONLY:
{{
  "chart": {{
    "type": "none|bar|line|scatter|pie",
    "x": null,
    "y": null,
    "title": null,
    "source_analysis_id": null
  }}
}}
"""
        try:
            chart_data = llm_json(
                chart_prompt,
                required_key="chart",
                retries=1,
            )
            if isinstance(chart_data.get("chart"), dict):
                chart = chart_data["chart"]
        except Exception:
            chart = no_chart()

    return {
        "answer": answer,
        "basic_insights": basic_insights,
        "paradoxical_insights": paradoxical_insights,
        "strategy": strategy,
        "limitations": limitations,
        "chart": chart,
        "supported_paradoxes": supported_paradoxes,
    }


# ============================================================
# 10. CHART HELPERS
# ============================================================

def detect_chart_requested(
    question: str,
):
    q = question.lower()

    keywords = [
        "chart",
        "graph",
        "plot",
        "visual",
        "visualization",
        "biểu đồ",
        "vẽ",
    ]

    return any(
        word in q
        for word in keywords
    )


def deterministic_chart_fallback(
    chart,
    primary_results,
    chart_requested,
):
    """
    Generic chart fallback only.
    It chooses columns; it does not create business insights.
    """
    if not chart_requested:
        return no_chart()

    requested_type = str(
        chart.get(
            "type",
            "none",
        )
    ).lower()

    requested_x = chart.get("x")
    requested_y = chart.get("y")
    requested_source = chart.get(
        "source_analysis_id"
    )

    # First try model-specified table.
    if requested_source:
        for item in primary_results:
            if (
                item.get("id")
                == requested_source
            ):
                columns = item.get(
                    "columns",
                    [],
                )

                if (
                    requested_type
                    in {
                        "bar",
                        "line",
                        "scatter",
                        "pie",
                    }
                    and requested_x in columns
                    and requested_y in columns
                ):
                    return {
                        "type": requested_type,
                        "x": requested_x,
                        "y": requested_y,
                        "title": chart.get(
                            "title"
                        ),
                        "source_analysis_id": requested_source,
                    }

    # Generic fallback from any primary table.
    for item in primary_results:
        records = item.get(
            "records",
            [],
        )

        if not records:
            continue

        df = pd.DataFrame(records)

        numeric_cols = (
            df.select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        categorical_cols = [
            col
            for col in df.columns
            if col not in numeric_cols
        ]

        if (
            categorical_cols
            and numeric_cols
        ):
            return {
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": item.get(
                    "title"
                ),
                "source_analysis_id": item.get(
                    "id"
                ),
            }

        if len(numeric_cols) >= 2:
            return {
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": item.get(
                    "title"
                ),
                "source_analysis_id": item.get(
                    "id"
                ),
            }

    return no_chart()


# ============================================================
# 11. CLARIFICATION / ERROR RESULTS
# ============================================================

def clarification_result(
    answer: str,
    limitation: Optional[str] = None,
):
    return {
        "answer": answer,
        "primary_analyses": [],
        "paradox_candidates": [],
        "paradox_judgments": [],
        "basic_insights": [
            "Chưa thể tạo insight định lượng đáng tin cậy trước khi câu hỏi được làm rõ."
        ],
        "paradoxical_insights": [
            "Paradox Hunter chưa chạy vì chưa có primary SQL evidence."
        ],
        "strategy": {
            "short_term": [
                "Làm rõ metric hoặc phạm vi phân tích trước khi ra quyết định."
            ],
            "medium_term": [
                "Chuẩn hóa định nghĩa KPI để các phân tích sau nhất quán."
            ],
            "long_term": [
                "Duy trì semantic layer rõ ràng cho các KPI kinh doanh quan trọng."
            ],
        },
        "limitations": (
            [limitation]
            if limitation
            else []
        ),
        "chart": no_chart(),
        "stage_trace": {
            "analyst": "NOT RUN",
            "paradox_hunter": "NOT RUN",
            "paradox_verification": "NOT RUN",
            "strategist": "NOT RUN",
        },
        "error": False,
    }


def error_result(
    message: str,
    stage_trace=None,
):
    return {
        "answer": message,
        "primary_analyses": [],
        "paradox_candidates": [],
        "paradox_judgments": [],
        "basic_insights": [],
        "paradoxical_insights": [],
        "strategy": empty_strategy(),
        "limitations": [
            "The agent workflow did not complete."
        ],
        "chart": no_chart(),
        "stage_trace": stage_trace or {},
        "error": True,
    }


# ============================================================
# 12. FULL AGENT WORKFLOW
# ============================================================

def ask_data_agent(
    question: str,
    history,
):
    """
    v7 FAST AGENTIC WORKFLOW

    Normal path:
    1 LLM call  -> Analyst SQL plan
    Python      -> execute primary SQL
    1 LLM call  -> Paradox Hunter + verification SQL
    Python      -> execute paradox SQL
    1 LLM call  -> Judge + insights + strategy

    Usually ~3 LLM calls total, not 20-30.
    """

    q = question.lower()

    # Semantic clarification guard only.
    mentions_revenue = (
        "revenue" in q
        or "doanh thu" in q
    )

    explicitly_price = (
        "sum(price)" in q
        or "sum of price" in q
        or "using price" in q
        or "theo price" in q
    )

    explicitly_payment = (
        "sum(payment_value)" in q
        or "sum of payment_value" in q
        or "using payment_value" in q
        or "theo payment_value" in q
    )

    if (
        mentions_revenue
        and not explicitly_price
        and not explicitly_payment
    ):
        return clarification_result(
            answer=(
                "Revenue/doanh thu là metric mơ hồ trong transformed dataset này. "
                "Bạn muốn dùng **SUM(price)** (retained item-price total) "
                "hay **SUM(payment_value)** (retained payment-value total)?"
            ),
            limitation=(
                "Dataset hiện tại không bảo đảm một complete original-order revenue metric."
            ),
        )

    stage_trace = {
        "analyst": "RUNNING",
        "paradox_hunter": "PENDING",
        "paradox_verification": "PENDING",
        "strategist": "PENDING",
    }

    # --------------------------------------------------------
    # 1. ANALYST
    # --------------------------------------------------------
    try:
        plan = plan_primary_analyses(
            question,
            history,
        )
    except Exception as e:
        stage_trace["analyst"] = (
            f"FAILED — {type(e).__name__}: {e}"
        )

        return error_result(
            (
                "Analyst failed while planning SQL. "
                "The model response could not be converted into executable analysis blocks."
            ),
            stage_trace,
        )

    primary_results = run_sql_plan(
        plan,
        max_queries=4,
    )

    usable_primary = [
        item
        for item in primary_results
        if (
            item.get("status")
            in {"SAFE", "WARNING"}
            and item.get("records")
        )
    ]

    if not usable_primary:
        stage_trace["analyst"] = "FAILED"

        return error_result(
            (
                "Analyst could not obtain usable SQL evidence. "
                "Inspect the SQL tab for generated queries."
            ),
            stage_trace,
        )

    stage_trace["analyst"] = (
        f"COMPLETED — {len(usable_primary)} usable analysis table(s)"
    )

    # --------------------------------------------------------
    # 2. PARADOX HUNTER
    # --------------------------------------------------------
    stage_trace["paradox_hunter"] = "RUNNING"

    try:
        paradox_tests = plan_fast_paradox_tests(
            question=question,
            primary_results=usable_primary,
        )

        stage_trace["paradox_hunter"] = (
            f"COMPLETED — {len(paradox_tests)} broad paradox test(s)"
        )

    except Exception as e:
        paradox_tests = []
        stage_trace["paradox_hunter"] = (
            f"FAILED — {type(e).__name__}: {e}"
        )

    # --------------------------------------------------------
    # 3. SQL VERIFICATION
    # --------------------------------------------------------
    stage_trace["paradox_verification"] = "RUNNING"

    paradox_results = execute_fast_paradox_tests(
        paradox_tests
    )

    usable_tests = [
        item
        for item in paradox_results
        if (
            item.get("status")
            in {"SAFE", "WARNING"}
            and item.get("records")
        )
    ]

    stage_trace["paradox_verification"] = (
        f"COMPLETED — {len(usable_tests)} test table(s) returned evidence"
    )

    # --------------------------------------------------------
    # 4. FINAL JUDGE + STRATEGIST
    # --------------------------------------------------------
    stage_trace["strategist"] = "RUNNING"

    chart_requested = detect_chart_requested(
        question
    )

    try:
        report = fast_final_report(
            question=question,
            primary_results=usable_primary,
            paradox_results=usable_tests,
            chart_requested=chart_requested,
        )

        judgments = report.get(
            "judgments",
            [],
        )

        supported_count = sum(
            1
            for item in judgments
            if item.get("supported") is True
        )

        stage_trace["paradox_verification"] += (
            f" | {supported_count} paradox(es) verified"
        )

        stage_trace["strategist"] = "COMPLETED"

    except Exception as e:
        # Important: do not discard all collected evidence.
        judgments = []

        stage_trace["strategist"] = (
            f"FAILED — {type(e).__name__}: {e}"
        )

        report = {
            "answer": (
                "SQL evidence đã được thu thập thành công, nhưng bước tổng hợp "
                "ngôn ngữ cuối cùng gặp lỗi. Evidence vẫn có thể kiểm tra ở các tab bên dưới."
            ),
            "basic_insights": [],
            "paradoxical_insights": [
                "Chưa thể kết luận insight nghịch lý vì Final Judge không hoàn tất."
            ],
            "strategy": empty_strategy(),
            "limitations": [],
            "chart": no_chart(),
        }

    chart = deterministic_chart_fallback(
        chart=report.get(
            "chart",
            no_chart(),
        ),
        primary_results=usable_primary,
        chart_requested=chart_requested,
    )

    return {
        "answer": report.get(
            "answer",
            "",
        ),
        "primary_analyses": primary_results,
        "paradox_candidates": paradox_results,
        "paradox_judgments": judgments,
        "basic_insights": report.get(
            "basic_insights",
            [],
        ),
        "paradoxical_insights": report.get(
            "paradoxical_insights",
            [],
        ),
        "strategy": report.get(
            "strategy",
            empty_strategy(),
        ),
        "limitations": report.get(
            "limitations",
            [],
        ),
        "chart": chart,
        "stage_trace": stage_trace,
        "error": False,
    }


# ============================================================
# 13. MULTI-CHAT SESSION STATE
# ============================================================

def create_new_chat():
    chat_id = str(
        uuid.uuid4()
    )

    st.session_state.chats[
        chat_id
    ] = {
        "title": "New chat",
        "messages": [],
    }

    st.session_state.current_chat_id = (
        chat_id
    )


def delete_chat(chat_id):
    if (
        chat_id
        not in st.session_state.chats
    ):
        return

    del st.session_state.chats[
        chat_id
    ]

    if not st.session_state.chats:
        create_new_chat()
        return

    if (
        st.session_state.current_chat_id
        == chat_id
    ):
        st.session_state.current_chat_id = (
            next(
                reversed(
                    st.session_state.chats
                )
            )
        )


if "chats" not in st.session_state:
    st.session_state.chats = {}

if (
    "current_chat_id"
    not in st.session_state
    or st.session_state.current_chat_id
    not in st.session_state.chats
):
    create_new_chat()


# ============================================================
# 14. SIDEBAR — NEW CHAT + HISTORY + DELETE
# ============================================================

st.sidebar.title("💬 Chats")

if st.sidebar.button(
    "＋ New chat",
    use_container_width=True,
    type="primary",
):
    create_new_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Conversation history"
)

for chat_id, chat in reversed(
    list(
        st.session_state.chats.items()
    )
):
    cols = st.sidebar.columns(
        [0.82, 0.18]
    )

    is_current = (
        chat_id
        == st.session_state.current_chat_id
    )

    title = chat.get(
        "title",
        "New chat",
    )

    label = (
        f"● {title}"
        if is_current
        else title
    )

    with cols[0]:
        if st.button(
            label,
            key=f"open_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.current_chat_id = (
                chat_id
            )
            st.rerun()

    with cols[1]:
        if st.button(
            "🗑️",
            key=f"delete_{chat_id}",
            help="Delete chat",
            use_container_width=True,
        ):
            delete_chat(chat_id)
            st.rerun()


current_chat = (
    st.session_state.chats[
        st.session_state.current_chat_id
    ]
)

history = current_chat[
    "messages"
]


# ============================================================
# 15. UI HELPERS
# ============================================================

def render_bullets(
    items,
    empty_text="No evidence-based item available.",
):
    if not items:
        st.info(empty_text)
        return

    for item in items:
        if item:
            st.markdown(
                f"- {item}"
            )


def analysis_to_df(item):
    return pd.DataFrame(
        item.get(
            "records",
            [],
        )
    )


def get_chart_df(
    result,
):
    chart = result.get(
        "chart",
        {},
    )

    source_id = chart.get(
        "source_analysis_id"
    )

    if not source_id:
        return None

    for item in result.get(
        "primary_analyses",
        [],
    ):
        if item.get("id") == source_id:
            return analysis_to_df(
                item
            )

    return None


def render_chart(result):
    chart = result.get(
        "chart",
        no_chart(),
    )

    chart_type = chart.get(
        "type",
        "none",
    )

    if chart_type == "none":
        return

    df = get_chart_df(result)

    if (
        df is None
        or df.empty
    ):
        return

    x = chart.get("x")
    y = chart.get("y")
    title = chart.get(
        "title"
    )

    if (
        x not in df.columns
        or y not in df.columns
    ):
        return

    fig = None

    if chart_type == "bar":
        fig = px.bar(
            df,
            x=x,
            y=y,
            title=title,
        )

    elif chart_type == "line":
        fig = px.line(
            df,
            x=x,
            y=y,
            title=title,
            markers=True,
        )

    elif chart_type == "scatter":
        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=title,
        )

    elif chart_type == "pie":
        fig = px.pie(
            df,
            names=x,
            values=y,
            title=title,
        )

    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
        )


def render_stage_audit(result):
    trace = result.get(
        "stage_trace",
        {},
    )

    with st.expander(
        "🔍 Auto-Audit & Agent Workflow",
        expanded=False,
    ):
        st.markdown(
            "#### Agent stages"
        )

        st.markdown(
            f"- **1. Analyst:** {trace.get('analyst', 'UNKNOWN')}"
        )
        st.markdown(
            f"- **2. Paradox Hunter:** {trace.get('paradox_hunter', 'UNKNOWN')}"
        )
        st.markdown(
            f"- **3. Paradox Verification/Judge:** {trace.get('paradox_verification', 'UNKNOWN')}"
        )
        st.markdown(
            f"- **4. Business Strategist:** {trace.get('strategist', 'UNKNOWN')}"
        )

        st.markdown("---")
        st.markdown(
            "#### SQL validation summary"
        )

        all_items = (
            result.get(
                "primary_analyses",
                [],
            )
            + result.get(
                "paradox_candidates",
                [],
            )
        )

        if not all_items:
            st.info(
                "No SQL was executed."
            )
            return

        safe = sum(
            1
            for item in all_items
            if item.get("status") == "SAFE"
        )
        warning = sum(
            1
            for item in all_items
            if item.get("status") == "WARNING"
        )
        blocked = sum(
            1
            for item in all_items
            if item.get("status") == "BLOCKED"
        )

        st.write(
            f"SAFE: {safe} | WARNING: {warning} | BLOCKED: {blocked}"
        )


def render_evidence_tab(
    result,
):
    st.markdown(
        "### Primary evidence"
    )

    primary = result.get(
        "primary_analyses",
        [],
    )

    if not primary:
        st.info(
            "No primary SQL evidence."
        )

    for item in primary:
        st.markdown(
            f"#### {item.get('id')} — {item.get('title')}"
        )

        if item.get("reason"):
            st.caption(
                item["reason"]
            )

        st.caption(
            f"SQL validation: {item.get('status')}"
        )

        df = analysis_to_df(
            item
        )

        if not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.markdown(
        "### Paradox discovery & verification"
    )

    candidates = result.get(
        "paradox_candidates",
        [],
    )

    judgments = {
        str(item.get("id")): item
        for item in result.get(
            "paradox_judgments",
            [],
        )
    }

    if not candidates:
        st.info(
            "Paradox Hunter did not produce a testable candidate in this run."
        )

    for item in candidates:
        candidate_id = str(
            item.get("id")
        )

        judgment = judgments.get(
            candidate_id,
            {},
        )

        st.markdown(
            f"#### {candidate_id}"
        )
        st.write(
            item.get(
                "hypothesis",
                "",
            )
        )

        if item.get(
            "why_surprising"
        ):
            st.caption(
                "Why it may be surprising: "
                + item["why_surprising"]
            )

        df = pd.DataFrame(
            item.get(
                "records",
                [],
            )
        )

        if not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

        supported = judgment.get(
            "supported"
        )

        if supported is True:
            st.success(
                "VERIFIED — "
                + str(
                    judgment.get(
                        "reason",
                        "",
                    )
                )
            )

        elif supported is False:
            st.warning(
                "NOT VERIFIED — "
                + str(
                    judgment.get(
                        "reason",
                        "",
                    )
                )
            )

        else:
            st.info(
                "No judge decision available."
            )


def render_sql_tab(result):
    st.markdown(
        "### Analyst SQL"
    )

    for item in result.get(
        "primary_analyses",
        [],
    ):
        st.markdown(
            f"#### {item.get('id')} — {item.get('status')}"
        )

        st.code(
            item.get(
                "sql",
                "",
            ),
            language="sql",
        )

        render_bullets(
            item.get(
                "validation_messages",
                [],
            ),
            "",
        )

    st.markdown("---")
    st.markdown(
        "### Paradox verification SQL"
    )

    for item in result.get(
        "paradox_candidates",
        [],
    ):
        st.markdown(
            f"#### {item.get('id')} — {item.get('status')}"
        )

        st.code(
            item.get(
                "verification_sql",
                "",
            ),
            language="sql",
        )

        render_bullets(
            item.get(
                "validation_messages",
                [],
            ),
            "",
        )


def render_report(result):
    if result.get("error"):
        st.error(
            result.get(
                "answer",
                "Agent workflow failed.",
            )
        )

        render_stage_audit(
            result
        )
        return

    st.success(
        "💡 AI Agent đã hoàn tất: phân tích → tìm nghịch lý → "
        "kiểm chứng bằng SQL → đề xuất chiến lược."
    )

    render_stage_audit(
        result
    )

    insight_tab, strategy_tab, evidence_tab, sql_tab = st.tabs([
        "📊 Báo cáo Insight",
        "💡 Chiến lược",
        "🧪 Evidence & Paradox Test",
        "⚙️ SQL",
    ])

    with insight_tab:
        st.markdown(
            "### Kết luận"
        )
        st.markdown(
            result.get(
                "answer",
                "",
            )
        )

        render_chart(
            result
        )

        st.markdown(
            "### 1. Insight cơ bản"
        )
        render_bullets(
            result.get(
                "basic_insights",
                [],
            ),
            "No basic insight was produced.",
        )

        st.markdown(
            "### 2. Insight nghịch lý"
        )
        render_bullets(
            result.get(
                "paradoxical_insights",
                [],
            ),
            "No verified paradox was produced.",
        )

        limitations = result.get(
            "limitations",
            [],
        )

        if limitations:
            with st.expander(
                "⚠️ Giới hạn diễn giải"
            ):
                render_bullets(
                    limitations
                )

    with strategy_tab:
        strategy = result.get(
            "strategy",
            empty_strategy(),
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "### ⚡ Ngắn hạn"
            )
            render_bullets(
                strategy.get(
                    "short_term",
                    [],
                ),
                "No short-term recommendation.",
            )

        with col2:
            st.markdown(
                "### 🧭 Trung hạn"
            )
            render_bullets(
                strategy.get(
                    "medium_term",
                    [],
                ),
                "No medium-term recommendation.",
            )

        with col3:
            st.markdown(
                "### 🏗️ Dài hạn"
            )
            render_bullets(
                strategy.get(
                    "long_term",
                    [],
                ),
                "No long-term recommendation.",
            )

    with evidence_tab:
        render_evidence_tab(
            result
        )

    with sql_tab:
        render_sql_tab(
            result
        )


def render_message(message):
    role = message.get(
        "role",
        "assistant",
    )

    with st.chat_message(role):
        if role == "user":
            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )
            return

        result = message.get(
            "result"
        )

        if result:
            render_report(
                result
            )
        else:
            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )


def make_history_for_agent(
    messages,
):
    converted = []

    for message in messages[-10:]:
        converted.append({
            "role": message.get(
                "role",
                "assistant",
            ),
            "content": message.get(
                "content",
                "",
            ),
        })

    return converted


# ============================================================
# 16. MAIN PAGE
# ============================================================

st.title(
    "E-commerce AI Data Analyst"
)

st.caption(
    "Analyst → Paradox Hunter → SQL Verification → Business Strategist "
    "• SQLite Read-only"
)

for message in history:
    render_message(
        message
    )


question = st.chat_input(
    "Ask a question about the e-commerce data..."
)


if question:
    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    prior_history = (
        make_history_for_agent(
            history
        )
    )

    with st.spinner(
        "Agent đang phân tích và kiểm chứng dữ liệu..."
    ):
        result = ask_data_agent(
            question=question,
            history=prior_history,
        )

    history.append({
        "role": "user",
        "content": question,
    })

    history.append({
        "role": "assistant",
        "content": result.get(
            "answer",
            "",
        ),
        "result": result,
    })

    if (
        current_chat.get(
            "title"
        )
        == "New chat"
    ):
        clean_title = " ".join(
            question.strip().split()
        )

        if len(
            clean_title
        ) > 34:
            clean_title = (
                clean_title[:34]
                + "..."
            )

        current_chat[
            "title"
        ] = clean_title

    st.rerun()
