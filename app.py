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

    # Semantic label guard only: this does not create any insight.
    # It prevents presentation text from calling SUM(price) revenue/sales.
    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)",
        normalized["sql"],
        flags=re.IGNORECASE,
    ):
        for field in ("title", "reason"):
            text = normalized[field]
            text = re.sub(
                r"\brevenue\b",
                "retained item-price total",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bsales\b",
                "retained item-price total",
                text,
                flags=re.IGNORECASE,
            )
            normalized[field] = text

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

def plan_primary_analyses(
    question: str,
    history,
):
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
- Every numerical claim later must come from SQL you request here.
- Respect all transformed-data limitations in the codebook.
- Do not use unsupported basket-size, repeat-customer, original item-count,
  or original payment-transaction-count metrics.
- Never assume a currency.
- If a business metric is ambiguous, use a clearly named metric rather than
  silently redefining it.
- If you use SUM(price), label it explicitly as a retained item-price total.
  NEVER call SUM(price) revenue or sales.
- If you use SUM(payment_value), label it explicitly as a retained payment-value total.
  NEVER call it guaranteed complete revenue.
- Do not describe payment_type counts as customer preference.
- For a narrow factual question, generate 1-3 focused analyses.
- For an open-ended business/strategy/insight question, generate 3-6
  complementary analyses chosen by YOU from the available schema.
- Do not hard-code a predetermined business story. Choose analyses because
  they help answer THIS question.

Return ONLY this JSON shape:

{{
  "analyses": [
    {{
      "id": "A1",
      "title": "short analysis title",
      "reason": "why this evidence is needed",
      "sql": "SELECT ..."
    }}
  ]
}}

No markdown.
No prose outside JSON.
"""

    data = llm_json(
        prompt,
        required_key="analyses",
    )

    analyses = data.get(
        "analyses",
        [],
    )

    if not isinstance(
        analyses,
        list,
    ):
        raise ValueError(
            "analyses must be a list."
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
    q = question.lower()

    # Semantic hard guard — not a paradox rule.
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
                "Revenue/doanh thu là một metric mơ hồ trong transformed dataset này. "
                "Bạn muốn dùng định nghĩa nào?\n\n"
                "1. **SUM(price)** — tổng `price` trên retained order-item rows.\n"
                "2. **SUM(payment_value)** — tổng retained payment values.\n\n"
                "Hai metric này không nên tự động được diễn giải là complete original-order revenue."
            ),
            limitation=(
                "Dataset đã transformation nên không có một metric complete "
                "original-order revenue được xác định chắc chắn."
            ),
        )

    stage_trace = {
        "analyst": "RUNNING",
        "paradox_hunter": "PENDING",
        "paradox_verification": "PENDING",
        "strategist": "PENDING",
    }

    # -----------------------
    # Stage 1 — Analyst
    # -----------------------
    try:
        plan = plan_primary_analyses(
            question,
            history,
        )
    except Exception as e:
        stage_trace["analyst"] = "FAILED"

        return error_result(
            (
                "Analyst stage failed while planning SQL: "
                f"{type(e).__name__}: {e}"
            ),
            stage_trace,
        )

    primary_results = run_sql_plan(
        plan,
        max_queries=6,
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
                "Open the SQL/Audit tab to inspect the generated queries."
            ),
            stage_trace,
        )

    stage_trace["analyst"] = (
        f"COMPLETED — {len(usable_primary)} usable SQL analyses"
    )

    # -----------------------
    # Stage 2 — Paradox Hunter
    # -----------------------
    stage_trace["paradox_hunter"] = "RUNNING"

    all_candidates = []
    all_paradox_results = []
    all_judgments = []
    supported_count = 0
    prior_attempts = []

    # Up to two discovery rounds.
    # Round 2 is only used if Round 1 fails to verify a paradox.
    for round_number in (1, 2):
        try:
            candidates = discover_paradox_candidates(
                question=question,
                primary_results=usable_primary,
                prior_attempts=prior_attempts,
                round_number=round_number,
            )
        except Exception as e:
            stage_trace["paradox_hunter"] = (
                f"ROUND {round_number} FAILED — {type(e).__name__}: {e}"
            )
            candidates = []

        if not candidates:
            # Give the Hunter a second independent discovery round instead
            # of stopping immediately after one empty generation.
            if round_number == 1:
                prior_attempts.append({
                    "round": 1,
                    "result": "No candidate hypotheses were generated. Try different relationships/segments in round 2.",
                })
                continue
            break

        paradox_results = verify_paradox_candidates(
            question=question,
            candidates=candidates,
            primary_results=usable_primary,
        )

        executable = [
            item
            for item in paradox_results
            if (
                item.get("status")
                in {"SAFE", "WARNING"}
                and item.get("records")
            )
        ]

        try:
            judgments = judge_paradoxes(
                question=question,
                primary_results=usable_primary,
                paradox_results=executable,
            )
        except Exception as e:
            judgments = []
            stage_trace["paradox_verification"] = (
                f"Judge round {round_number} failed: {type(e).__name__}: {e}"
            )

        all_candidates.extend(candidates)
        all_paradox_results.extend(paradox_results)
        all_judgments.extend(judgments)

        round_supported = sum(
            1
            for item in judgments
            if item.get("supported") is True
        )
        supported_count += round_supported

        # Feed failures/results back to round 2 so the Hunter does not
        # repeat the same weak hypothesis.
        judgment_map_round = {
            str(j.get("id")): j
            for j in judgments
        }

        prior_attempts.extend([
            {
                "id": item.get("id"),
                "hypothesis": item.get("hypothesis"),
                "verification_logic": item.get("verification_logic"),
                "verification_rows": item.get("records", [])[:15],
                "judge": judgment_map_round.get(
                    str(item.get("id")),
                    {},
                ),
            }
            for item in paradox_results
        ])

        if round_supported > 0:
            break

    stage_trace["paradox_hunter"] = (
        f"COMPLETED — {len(all_candidates)} candidate hypothesis(es) across up to 2 rounds"
    )

    stage_trace["paradox_verification"] = (
        f"COMPLETED — {len(all_paradox_results)} SQL test(s); "
        f"{supported_count} paradox(es) verified"
    )

    paradox_results = all_paradox_results
    judgments = all_judgments

    # -----------------------
    # Stage 3B — Strategist
    # -----------------------
    stage_trace["strategist"] = "RUNNING"

    chart_requested = (
        detect_chart_requested(
            question
        )
    )

    try:
        report = synthesize_report(
            question=question,
            primary_results=usable_primary,
            paradox_results=[
                item
                for item in paradox_results
                if (
                    item.get("status") in {"SAFE", "WARNING"}
                    and item.get("records")
                )
            ],
            paradox_judgments=judgments,
            chart_requested=chart_requested,
        )

        stage_trace["strategist"] = "COMPLETED"

    except Exception as e:
        stage_trace["strategist"] = (
            "FAILED — "
            f"{type(e).__name__}: {e}"
        )

        # Preserve verified evidence even if final prose synthesis fails.
        report = {
            "answer": (
                "SQL evidence was collected, but the final strategy synthesis failed."
            ),
            "basic_insights": [
                "Primary SQL evidence is available in the Evidence tab."
            ],
            "paradoxical_insights": [
                (
                    item.get("insight")
                    for item in judgments
                    if item.get("supported") is True
                )
            ],
            "strategy": empty_strategy(),
            "limitations": [
                "Final LLM synthesis failed; SQL evidence remains available."
            ],
            "chart": no_chart(),
            "supported_paradoxes": [],
        }

        # Flatten accidental generator/list shape.
        flattened = []
        for item in judgments:
            if (
                item.get("supported") is True
                and item.get("insight")
            ):
                flattened.append(
                    item["insight"]
                )

        report[
            "paradoxical_insights"
        ] = (
            flattened
            or [
                "No verified paradoxical insight was available after the synthesis failure."
            ]
        )

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
