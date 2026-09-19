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
    Ask the model for JSON using ordinary text generation,
    not function calling.

    A retry asks the model to repair format only.
    """
    last_error = None

    for attempt in range(retries + 1):
        current_prompt = prompt

        if attempt > 0:
            current_prompt += """

IMPORTANT FORMAT REPAIR:
Your previous response could not be parsed.
Return ONLY valid JSON.
Do not use markdown fences.
Do not include commentary before or after the JSON.
"""

        try:
            raw = llm_text(current_prompt)
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
        f"LLM JSON generation failed: {last_error}"
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
    return {
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


def empty_strategy():
    return {
        "short_term": [],
        "medium_term": [],
        "long_term": [],
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
):
    evidence = compact_evidence(
        primary_results,
        max_rows_per_query=18,
    )

    prompt = f"""
You are Stage 2: PARADOX HUNTER.

The user asked:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK:
{codebook_text}

Your task:
Discover potentially counter-intuitive, surprising, or tension-filled
patterns that are RELEVANT to the user's question.

A paradox candidate can be:
- a reversal across segments,
- high overall performance but weak performance in a subgroup,
- volume concentration without corresponding performance,
- rank reversal under another valid metric,
- a surprising time pattern,
- an exception to an apparent aggregate pattern,
- two related business indicators moving in opposite directions.

IMPORTANT:
- YOU must discover the candidate from evidence/schema.
- No paradox content is pre-programmed.
- Do not call something paradoxical merely because it is large or small.
- Do not claim a paradox is true yet.
- Each candidate MUST include a NEW verification SQL query that could
  falsify or support it.
- SQL must be SELECT/WITH only.
- Respect transformed-data limitations.
- Do not invent currency.
- Generate 1-4 candidates.
- If primary evidence genuinely offers no plausible candidate, return an
  empty list.

Return ONLY valid JSON:

{{
  "candidates": [
    {{
      "id": "P1",
      "hypothesis": "candidate paradox stated as a testable hypothesis",
      "why_surprising": "why this would be counter-intuitive if supported",
      "verification_sql": "SELECT ..."
    }}
  ]
}}

No markdown.
No prose outside JSON.
"""

    data = llm_json(
        prompt,
        required_key="candidates",
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
        candidates[:4],
        start=1,
    ):
        normalized.append({
            "id": str(
                candidate.get(
                    "id",
                    f"P{index}",
                )
            ),
            "hypothesis": str(
                candidate.get(
                    "hypothesis",
                    "",
                )
            ),
            "why_surprising": str(
                candidate.get(
                    "why_surprising",
                    "",
                )
            ),
            "verification_sql": str(
                candidate.get(
                    "verification_sql",
                    "",
                )
            ).strip(),
        })

    return normalized


def verify_paradox_candidates(
    candidates,
):
    verified = []

    for candidate in candidates[:4]:
        sql = candidate.get(
            "verification_sql",
            "",
        )

        validation = validate_sql_query(sql)

        item = {
            **candidate,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(
                    sql,
                    max_rows=100,
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

PARADOX CANDIDATES + THEIR VERIFICATION SQL RESULTS:
{json.dumps(paradox_results, ensure_ascii=False, default=str)}

Your job:
Judge whether each candidate paradox is SUPPORTED by its executed SQL.

Rules:
- Use only the SQL rows shown above.
- A candidate is supported only when its verification result actually
  demonstrates the claimed contrast/reversal/tension.
- A plausible story is NOT enough.
- If evidence is mixed or insufficient, mark supported=false.
- Do not invent numbers.
- Respect dataset limitations.
- Rewrite supported insight concisely in the user's language.
- Include the concrete evidence that makes it surprising.

Return ONLY valid JSON:

{{
  "judgments": [
    {{
      "id": "P1",
      "supported": true,
      "insight": "verified paradoxical insight, or empty string if unsupported",
      "reason": "brief evidence-based judgment"
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
    primary_evidence = compact_evidence(
        primary_results,
        max_rows_per_query=20,
    )

    supported_paradoxes = []

    judgment_map = {
        str(item.get("id")): item
        for item in paradox_judgments
    }

    for item in paradox_results:
        candidate_id = str(
            item.get("id")
        )

        judgment = judgment_map.get(
            candidate_id,
            {},
        )

        if judgment.get(
            "supported"
        ) is True:
            supported_paradoxes.append({
                "id": candidate_id,
                "hypothesis": item.get(
                    "hypothesis"
                ),
                "verification_sql": item.get(
                    "verification_sql"
                ),
                "verification_rows": item.get(
                    "records",
                    [],
                )[:20],
                "verified_insight": judgment.get(
                    "insight",
                    "",
                ),
                "judgment_reason": judgment.get(
                    "reason",
                    "",
                ),
            })

    prompt = f"""
You are Stage 3B: BUSINESS STRATEGIST.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

VERIFIED PARADOXICAL INSIGHTS:
{json.dumps(supported_paradoxes, ensure_ascii=False, default=str)}

DATABASE CODEBOOK:
{codebook_text}

Create the final presentation-ready report.

CRITICAL GROUNDING RULES:
- Use only executed SQL evidence shown above.
- Do not invent numbers.
- Never invent a currency.
- Do not claim causality.
- Do not infer profitability, customer preference, or demand unless the
  executed evidence directly supports that claim.
- Respect transformed-data limitations.

BASIC INSIGHTS:
- Provide 1-4 direct insights relevant to the user's question.
- Prefer concrete evidence.

PARADOXICAL INSIGHTS:
- Copy/summarize ONLY verified paradoxes listed above.
- Never create a new paradox at this stage.
- If no paradox was verified, return one item explaining that this run
  did not verify a defensible paradox and that the Paradox Hunter's
  candidates failed or lacked evidence.

STRATEGY:
Provide:
- short_term: 1-3 evidence-linked operational actions
- medium_term: 1-3 process/analysis/resource-allocation actions
- long_term: 1-3 structural strategic actions

If evidence is insufficient for action, recommend what to measure next.

CHART:
User explicitly requested chart: {chart_requested}

If chart_requested is false:
- chart.type = "none"

If chart_requested is true:
- choose one useful primary analysis table
- source_analysis_id must match an analysis id such as A1
- x and y must exactly match columns returned by that analysis
- allowed type: bar, line, scatter, pie
- otherwise type = "none"

Return ONLY valid JSON:

{{
  "answer": "concise executive answer",
  "basic_insights": ["..."],
  "paradoxical_insights": ["..."],
  "strategy": {{
    "short_term": ["..."],
    "medium_term": ["..."],
    "long_term": ["..."]
  }},
  "limitations": ["..."],
  "chart": {{
    "type": "none|bar|line|scatter|pie",
    "x": null,
    "y": null,
    "title": null,
    "source_analysis_id": null
  }}
}}

{same_language_instruction(question)}

No markdown outside JSON.
"""

    data = llm_json(prompt)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Final report must be a JSON object."
        )

    # Defensive normalization.
    return {
        "answer": str(
            data.get(
                "answer",
                "Analysis completed.",
            )
        ),
        "basic_insights": (
            data.get(
                "basic_insights",
                [],
            )
            if isinstance(
                data.get(
                    "basic_insights",
                    [],
                ),
                list,
            )
            else []
        ),
        "paradoxical_insights": (
            data.get(
                "paradoxical_insights",
                [],
            )
            if isinstance(
                data.get(
                    "paradoxical_insights",
                    [],
                ),
                list,
            )
            else []
        ),
        "strategy": (
            data.get(
                "strategy",
                empty_strategy(),
            )
            if isinstance(
                data.get(
                    "strategy",
                    {},
                ),
                dict,
            )
            else empty_strategy()
        ),
        "limitations": (
            data.get(
                "limitations",
                [],
            )
            if isinstance(
                data.get(
                    "limitations",
                    [],
                ),
                list,
            )
            else []
        ),
        "chart": (
            data.get(
                "chart",
                no_chart(),
            )
            if isinstance(
                data.get(
                    "chart",
                    {},
                ),
                dict,
            )
            else no_chart()
        ),
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

    try:
        candidates = discover_paradox_candidates(
            question,
            usable_primary,
        )
        stage_trace[
            "paradox_hunter"
        ] = (
            f"COMPLETED — {len(candidates)} candidate hypotheses"
        )

    except Exception as e:
        candidates = []
        stage_trace[
            "paradox_hunter"
        ] = (
            "FAILED — "
            f"{type(e).__name__}: {e}"
        )

    # -----------------------
    # Stage 2B — Verification
    # -----------------------
    stage_trace[
        "paradox_verification"
    ] = "RUNNING"

    paradox_results = (
        verify_paradox_candidates(
            candidates
        )
        if candidates
        else []
    )

    executable_paradox_results = [
        item
        for item in paradox_results
        if (
            item.get("status")
            in {"SAFE", "WARNING"}
            and item.get("records")
        )
    ]

    stage_trace[
        "paradox_verification"
    ] = (
        "COMPLETED — "
        f"{len(executable_paradox_results)} candidate SQL tests returned evidence"
    )

    # -----------------------
    # Stage 3A — Judge
    # -----------------------
    try:
        judgments = judge_paradoxes(
            question,
            usable_primary,
            executable_paradox_results,
        )
    except Exception as e:
        judgments = []
        stage_trace[
            "paradox_verification"
        ] += (
            " | Judge failed: "
            f"{type(e).__name__}: {e}"
        )

    supported_count = sum(
        1
        for item in judgments
        if item.get("supported") is True
    )

    stage_trace[
        "paradox_verification"
    ] += (
        f" | {supported_count} paradox(es) verified"
    )

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
            paradox_results=executable_paradox_results,
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
