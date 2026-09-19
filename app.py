import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd
import plotly.express as px
import streamlit as st

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq


# ============================================================
# 1. APP CONFIG + PATHS
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
# 2. SQL SAFETY + VALIDATION
# ============================================================

FORBIDDEN_SQL = {
    "INSERT", "UPDATE", "DELETE", "DROP",
    "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "VACUUM",
    "REINDEX", "PRAGMA",
}


def validate_readonly_sql(sql: str):
    """Basic deterministic read-only guard."""
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL query is empty."

    cleaned = sql.strip()
    first_word = cleaned.split()[0].upper()

    if first_word not in {"SELECT", "WITH"}:
        return False, "Only SELECT or WITH queries are allowed."

    for keyword in FORBIDDEN_SQL:
        if re.search(rf"\b{keyword}\b", cleaned, flags=re.IGNORECASE):
            return False, f"Forbidden SQL keyword: {keyword}"

    without_last_semicolon = cleaned.rstrip(";")
    if ";" in without_last_semicolon:
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"


def validate_sql_query(sql: str) -> dict:
    """
    Validate SQL syntax/schema + flag semantic risks.
    SAFE     = okay to execute.
    WARNING  = okay to execute, but interpretation needs care.
    BLOCKED  = do not execute.
    """
    safe, reason = validate_readonly_sql(sql)

    if not safe:
        return {
            "status": "BLOCKED",
            "messages": [reason],
        }

    normalized = " ".join(sql.lower().split())

    # Compile check only; does not run the data query.
    try:
        db_uri = f"file:{DB_PATH}?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
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
            "SUM(price) is the sum of prices on retained order-item rows, "
            "not guaranteed complete original-order revenue."
        )

    if re.search(
        r"\b(sum|avg)\s*\(\s*(?:\w+\.)?payment_value\s*\)",
        normalized,
    ):
        warnings.append(
            "payment_value comes from one retained payment row per order; "
            "do not automatically interpret it as guaranteed complete revenue."
        )

    if (
        "product_category_name" in normalized
        and "payment_value" in normalized
    ):
        warnings.append(
            "Category × payment_value is high-risk: the category belongs "
            "to the retained product row while payment_value belongs to "
            "the retained payment row."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "order_items" in normalized
    ):
        warnings.append(
            "Counting order_items rows does not measure the original number "
            "of items because this transformed table has one retained row per order."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "payments" in normalized
    ):
        warnings.append(
            "Counting payments rows does not measure the original number "
            "of payment transactions because only one retained row per order remains."
        )

    if (
        "order_delivered_timestamp" in normalized
        and "is not null" not in normalized
    ):
        warnings.append(
            "Delivery analysis references order_delivered_timestamp without "
            "explicitly requiring it to be non-NULL."
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


# ============================================================
# 3. LANGCHAIN SQL TOOLS
# ============================================================

@tool
def list_tables() -> str:
    """List all available tables and views in the e-commerce SQLite database."""
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY type, name;
            """
        ).fetchall()

    return "\n".join(
        f"{name} ({kind})"
        for name, kind in rows
    )


@tool
def get_schema(table_name: str) -> str:
    """Return the SQL schema definition for a table or view."""
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        row = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE name = ?
              AND type IN ('table', 'view');
            """,
            (table_name,),
        ).fetchone()

    if row is None:
        return f"Table or view '{table_name}' does not exist."

    return row[0]


@tool
def validate_sql(sql: str) -> str:
    """Validate SQL before execution. Always use this before execute_sql."""
    result = validate_sql_query(sql)

    lines = [f"STATUS: {result['status']}"]
    for message in result["messages"]:
        lines.append(f"- {message}")

    return "\n".join(lines)


@tool
def execute_sql(sql: str) -> str:
    """
    Execute read-only SQL against the e-commerce database.
    Returns at most 50 rows as JSON.
    """
    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        return "SQL BLOCKED: " + " ".join(validation["messages"])

    db_uri = f"file:{DB_PATH}?mode=ro"

    try:
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(51)
            columns = [item[0] for item in cursor.description]

        truncated = len(rows) > 50
        rows = rows[:50]

        result = [
            {col: row[col] for col in columns}
            for row in rows
        ]

        payload = {
            "rows": result,
            "truncated": truncated,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

    except Exception as e:
        return f"SQL ERROR: {type(e).__name__}: {e}"


TOOLS = [
    list_tables,
    get_schema,
    validate_sql,
    execute_sql,
]


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an AI Data Analyst for one specific e-commerce SQLite database.

LANGUAGE:
- Reply in the same language as the user whenever practical.
- Keep business explanations concise and presentation-ready.

AVAILABLE TOOLS:
- list_tables
- get_schema
- validate_sql
- execute_sql

CORE RULES:
1. Never invent numerical values.
2. Every dataset number must come from executed SQL.
3. Always call validate_sql before execute_sql for every new SQL query.
4. If validation returns BLOCKED, fix the SQL and validate again.
5. Only use read-only SQL.
6. If SQL fails, inspect the error and retry when possible.
7. Never turn correlation into causation.
8. Do not invent currency symbols or units.

SEMANTIC RULES:
- orders has one row per order.
- order_items contains one retained row per order in this transformed dataset.
- payments contains one retained payment row per order.
- products contains one metadata row per product_id.
- customers contains one transformed customer row.

UNSUPPORTED METRICS:
- original basket size
- original number of items per order
- original number of payment transactions per order
- repeat-customer rate / retention from customer_id

MONEY RULES:
If the user says "revenue" without defining it, do NOT choose automatically.
Ask whether they mean:
A. SUM(price)
B. SUM(payment_value)

SUM(price) means:
sum of price on retained order-item rows.

SUM(payment_value) means:
sum of retained payment values.

Do not automatically describe either one as true complete original-order revenue.

PRODUCT RULE:
product_category_name belongs to the retained product row.
Do not imply it represents every item originally contained in an order.

DELIVERY RULE:
For delivery duration/performance:
- normally use delivered orders
- require order_delivered_timestamp IS NOT NULL

OPEN-ENDED BUSINESS ANALYSIS RULE:
If the user broadly asks for business insights, business advice,
recommendations, overall performance, or strategy without specifying
one exact metric:

- DO NOT immediately ask the user to choose an analysis area.
- Proactively inspect the database.
- Perform a concise multi-dimensional diagnostic using 3-5 targeted SQL queries.
- Prefer supported evidence such as:
  * order volume / order-status distribution
  * geographic concentration
  * delivery performance
  * payment-type patterns
  * retained product/category performance using explicitly defined metrics
- Then synthesize the evidence into an answer.

INSIGHT RULE:
For every completed SQL-based analysis, support:
1. Basic Insight:
   the most important direct pattern in the SQL evidence.
2. Paradoxical Insight:
   a surprising contrast, tension, reversal, exception, or counter-intuitive
   pattern that is actually supported by the SQL evidence.

Never manufacture a paradox.
If the evidence is insufficient, explicitly say that no defensible
paradoxical pattern can be established from the current query alone.

STRATEGY RULE:
For every completed SQL-based analysis, support:
- Short-term strategy: immediate operational action.
- Medium-term strategy: analysis/process/resource action.
- Long-term strategy: structural or strategic action.

Strategies must be tied to evidence.
If evidence is too weak for a business action, say what additional analysis
is required rather than inventing a recommendation.

Do not infer demand, preference, profitability, or causality merely from
an aggregate ranking.

CHART RULE:
If the user requests a chart, graph, plot, visualization, "vẽ", or "biểu đồ":
- query the data needed for the chart
- DO NOT write chart JSON in the textual answer
- DO NOT write chart specifications in the textual answer
- the Streamlit layer renders the chart

DATABASE CODEBOOK:

{codebook_text}
"""


# ============================================================
# 5. STRUCTURED PRESENTATION MODELS
# ============================================================

class ChartSpec(BaseModel):
    type: Literal[
        "none",
        "bar",
        "line",
        "scatter",
        "pie",
    ] = "none"

    x: Optional[str] = None
    y: Optional[str] = None
    title: Optional[str] = None


class StrategyPlan(BaseModel):
    short_term: List[str] = Field(default_factory=list)
    medium_term: List[str] = Field(default_factory=list)
    long_term: List[str] = Field(default_factory=list)


class Presentation(BaseModel):
    answer: str
    basic_insights: List[str] = Field(default_factory=list)
    paradoxical_insights: List[str] = Field(default_factory=list)
    strategy: StrategyPlan = Field(default_factory=StrategyPlan)
    limitations: List[str] = Field(default_factory=list)
    chart: ChartSpec = Field(default_factory=ChartSpec)


# ============================================================
# 6. BUILD AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    structured_llm = llm.with_structured_output(
        Presentation,
        method="function_calling",
    )

    return agent, structured_llm


# ============================================================
# 7. TRACE / SQL HELPERS
# ============================================================

def extract_sql_queries(messages):
    queries = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            continue

        for call in tool_calls:
            if call.get("name") == "execute_sql":
                sql = call.get("args", {}).get("sql")
                if sql:
                    queries.append(sql)

    return queries


def sql_to_dataframe(sql: str) -> pd.DataFrame:
    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        return pd.DataFrame()

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        return pd.read_sql_query(sql, conn).head(200)


def build_sql_evidence(sql_queries):
    evidence = []

    # Keep the formatter prompt bounded.
    for idx, sql in enumerate(sql_queries[-6:], start=1):
        validation = validate_sql_query(sql)

        item = {
            "query_number": idx,
            "sql": sql,
            "validation": validation,
            "rows": [],
        }

        if validation["status"] != "BLOCKED":
            try:
                df = sql_to_dataframe(sql)
                item["rows"] = df.head(25).to_dict(orient="records")
            except Exception as e:
                item["rows"] = []
                item["read_error"] = f"{type(e).__name__}: {e}"

        evidence.append(item)

    return evidence


def build_audit_report(sql_queries):
    audit = []

    for idx, sql in enumerate(sql_queries, start=1):
        validation = validate_sql_query(sql)

        audit.append({
            "query_number": idx,
            "status": validation["status"],
            "messages": validation["messages"],
            "sql": sql,
        })

    return audit


def detect_chart_requested(question: str) -> bool:
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

    return any(word in q for word in keywords)


def deterministic_chart_fallback(question, chart, data_tables):
    """
    If the LLM chart spec is missing/invalid, select a simple safe chart
    from the SQL result columns.
    """
    if not data_tables:
        return ChartSpec(type="none")

    # Try the latest non-empty table first.
    for table in reversed(data_tables):
        df = table["df"]

        if df.empty:
            continue

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = [
            col for col in df.columns
            if col not in numeric_cols
        ]

        if (
            chart.type != "none"
            and chart.x in df.columns
            and chart.y in df.columns
        ):
            return chart

        q_lower = question.lower()

        if (
            ("relationship" in q_lower or "scatter" in q_lower)
            and len(numeric_cols) >= 2
        ):
            return ChartSpec(
                type="scatter",
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Data Relationship",
            )

        if categorical_cols and numeric_cols:
            return ChartSpec(
                type="bar",
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}",
            )

        if len(numeric_cols) >= 2:
            return ChartSpec(
                type="scatter",
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Data Relationship",
            )

    return ChartSpec(type="none")


def empty_strategy():
    return {
        "short_term": [],
        "medium_term": [],
        "long_term": [],
    }


def clarification_result(answer, limitation=None):
    return {
        "answer": answer,
        "sql": [],
        "data_tables": [],
        "audit": [],
        "basic_insights": [
            "Chưa thể tạo insight định lượng vì câu hỏi hiện tại cần được làm rõ trước khi chạy SQL."
        ],
        "paradoxical_insights": [
            "Chưa có đủ bằng chứng SQL để xác lập một insight nghịch lý đáng tin cậy."
        ],
        "strategy": {
            "short_term": [
                "Làm rõ metric hoặc phạm vi phân tích trước khi ra quyết định."
            ],
            "medium_term": [
                "Chuẩn hóa định nghĩa metric để các phân tích sau nhất quán."
            ],
            "long_term": [
                "Xây dựng semantic layer ổn định cho các KPI kinh doanh quan trọng."
            ],
        },
        "limitations": [limitation] if limitation else [],
        "chart": {
            "type": "none",
            "x": None,
            "y": None,
            "title": None,
        },
    }



# ============================================================
# 7B. DETERMINISTIC OPEN-ENDED BUSINESS DIAGNOSTIC
# ============================================================

def is_open_ended_business_request(question: str) -> bool:
    """
    Detect broad requests such as:
    - cho tôi insight doanh nghiệp
    - cho tôi lời khuyên kinh doanh
    - give me business advice
    - overall business insights / recommendations

    These requests bypass the agent tool loop and use a fixed,
    auditable diagnostic bundle. This avoids malformed multi-tool calls
    and guarantees that broad business advice is grounded in SQL.
    """
    q = " ".join(question.lower().split())

    markers = [
        "lời khuyên kinh doanh",
        "loi khuyen kinh doanh",
        "insight doanh nghiệp",
        "insight doanh nghiep",
        "insight kinh doanh",
        "tư vấn kinh doanh",
        "tu van kinh doanh",
        "đề xuất kinh doanh",
        "de xuat kinh doanh",
        "business advice",
        "business insights",
        "business insight",
        "overall business",
        "overall performance",
        "business recommendation",
        "business recommendations",
        "recommendations from the data",
        "strategy from the data",
        "chiến lược từ data",
        "chien luoc tu data",
    ]

    return any(marker in q for marker in markers)


def run_open_ended_business_analysis(question: str) -> dict:
    """
    Reliable open-ended business diagnostic.

    This path deliberately does NOT ask the LLM to generate tool calls
    or structured-output function calls. Python runs five validated SQL
    diagnostics, then derives a conservative executive summary directly
    from those query results.

    This makes broad prompts such as "cho tôi insight doanh nghiệp"
    stable even when the provider has intermittent tool-call JSON errors.
    """

    diagnostic_queries = [
        # 1. Order status / fulfillment mix
        """
        SELECT
            order_status,
            COUNT(*) AS order_count,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders),
                2
            ) AS share_pct
        FROM orders
        GROUP BY order_status
        ORDER BY order_count DESC;
        """,

        # 2. Geographic concentration
        """
        SELECT
            c.customer_state,
            COUNT(*) AS order_count,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders),
                2
            ) AS share_pct
        FROM orders AS o
        JOIN customers AS c
            ON o.customer_id = c.customer_id
        GROUP BY c.customer_state
        ORDER BY order_count DESC
        LIMIT 10;
        """,

        # 3. Delivery performance
        """
        SELECT
            COUNT(*) AS delivered_orders_with_timestamp,
            ROUND(
                AVG(
                    julianday(order_delivered_timestamp)
                    - julianday(order_purchase_timestamp)
                ),
                2
            ) AS avg_delivery_days,
            ROUND(
                100.0 * AVG(
                    CASE
                        WHEN date(order_delivered_timestamp)
                             <= date(order_estimated_delivery_date)
                        THEN 1.0
                        ELSE 0.0
                    END
                ),
                2
            ) AS on_time_or_early_pct
        FROM orders
        WHERE order_status = 'delivered'
          AND order_delivered_timestamp IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL;
        """,

        # 4. Payment-type mix.
        # This is an order-associated retained payment row distribution,
        # not the original number of payment transactions.
        """
        SELECT
            payment_type,
            COUNT(*) AS associated_order_count,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM payments),
                2
            ) AS share_pct
        FROM payments
        GROUP BY payment_type
        ORDER BY associated_order_count DESC;
        """,

        # 5. Retained product-category concentration
        """
        SELECT
            pr.product_category_name,
            COUNT(*) AS associated_order_count,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM order_items),
                2
            ) AS share_pct
        FROM order_items AS oi
        JOIN products AS pr
            ON oi.product_id = pr.product_id
        WHERE pr.product_category_name IS NOT NULL
        GROUP BY pr.product_category_name
        ORDER BY associated_order_count DESC
        LIMIT 10;
        """,
    ]

    sql_queries = []
    data_tables = []

    for i, sql in enumerate(diagnostic_queries, start=1):
        sql = sql.strip()
        validation = validate_sql_query(sql)

        if validation["status"] == "BLOCKED":
            continue

        try:
            df = sql_to_dataframe(sql)
        except Exception:
            df = pd.DataFrame()

        sql_queries.append(sql)
        data_tables.append({
            "query_number": i,
            "sql": sql,
            "df": df,
        })

    if len(data_tables) < 5 or any(item["df"].empty for item in data_tables):
        return {
            "answer": (
                "Không thể hoàn tất business diagnostic vì một hoặc nhiều "
                "SQL diagnostic không trả về dữ liệu."
            ),
            "sql": sql_queries,
            "data_tables": [
                {
                    "query_number": item["query_number"],
                    "sql": item["sql"],
                    "records": item["df"].to_dict(orient="records"),
                    "columns": item["df"].columns.tolist(),
                }
                for item in data_tables
            ],
            "audit": build_audit_report(sql_queries),
            "basic_insights": [],
            "paradoxical_insights": [],
            "strategy": empty_strategy(),
            "limitations": [
                "Một hoặc nhiều diagnostic queries không trả về dữ liệu."
            ],
            "chart": ChartSpec(type="none").model_dump(),
            "error": True,
        }

    status_df = data_tables[0]["df"]
    geo_df = data_tables[1]["df"]
    delivery_df = data_tables[2]["df"]
    payment_df = data_tables[3]["df"]
    category_df = data_tables[4]["df"]

    # -----------------------------
    # Extract auditable facts
    # -----------------------------
    total_orders = int(status_df["order_count"].sum())

    delivered_match = status_df[
        status_df["order_status"].astype(str).str.lower() == "delivered"
    ]

    if not delivered_match.empty:
        delivered_orders = int(delivered_match.iloc[0]["order_count"])
        delivered_share = float(delivered_match.iloc[0]["share_pct"])
    else:
        delivered_orders = 0
        delivered_share = 0.0

    non_delivered_orders = total_orders - delivered_orders
    non_delivered_share = round(100.0 - delivered_share, 2)

    top_state = geo_df.iloc[0]
    top_state_name = str(top_state["customer_state"])
    top_state_orders = int(top_state["order_count"])
    top_state_share = float(top_state["share_pct"])

    delivery_row = delivery_df.iloc[0]
    avg_delivery_days = float(delivery_row["avg_delivery_days"])
    on_time_pct = float(delivery_row["on_time_or_early_pct"])
    late_pct = round(100.0 - on_time_pct, 2)

    top_payment = payment_df.iloc[0]
    top_payment_name = str(top_payment["payment_type"])
    top_payment_orders = int(top_payment["associated_order_count"])
    top_payment_share = float(top_payment["share_pct"])

    top_category = category_df.iloc[0]
    top_category_name = str(top_category["product_category_name"])
    top_category_orders = int(top_category["associated_order_count"])
    top_category_share = float(top_category["share_pct"])

    # Detect language without another model call.
    q_lower = question.lower()
    vi_markers = [
        "cho tôi", "doanh nghiệp", "kinh doanh", "lời khuyên",
        "chiến lược", "dữ liệu", "của bạn", "insight doanh",
    ]
    is_vi = any(marker in q_lower for marker in vi_markers)

    if is_vi:
        answer = (
            f"Business diagnostic cho thấy hệ thống có **{total_orders:,} orders**; "
            f"**{delivered_share:.2f}%** mang trạng thái delivered. "
            f"Bang **{top_state_name}** chiếm **{top_state_share:.2f}%** số orders. "
            f"Đối với delivered orders có timestamp hợp lệ, thời gian giao trung bình là "
            f"**{avg_delivery_days:.2f} ngày** và **{on_time_pct:.2f}%** được giao "
            f"đúng hoặc sớm hơn estimated date. "
            f"Payment type gắn với nhiều orders nhất là **{top_payment_name} "
            f"({top_payment_share:.2f}%)**, còn retained product category đứng đầu là "
            f"**{top_category_name} ({top_category_share:.2f}% associated orders)**."
        )

        basic_insights = [
            (
                f"Fulfillment nhìn chung cao: {delivered_orders:,}/{total_orders:,} orders "
                f"({delivered_share:.2f}%) có status delivered; phần còn lại là "
                f"{non_delivered_orders:,} orders ({non_delivered_share:.2f}%)."
            ),
            (
                f"Đơn hàng tập trung mạnh về địa lý: {top_state_name} có "
                f"{top_state_orders:,} orders, chiếm {top_state_share:.2f}% toàn bộ orders."
            ),
            (
                f"Delivery performance của delivered orders có timestamp hợp lệ: "
                f"trung bình {avg_delivery_days:.2f} ngày; {on_time_pct:.2f}% "
                f"đúng/sớm hạn và khoảng {late_pct:.2f}% muộn hơn estimated date."
            ),
            (
                f"Payment mix tập trung vào {top_payment_name}: "
                f"{top_payment_orders:,} retained payment rows, tương ứng "
                f"{top_payment_share:.2f}% orders trong transformed payments table."
            ),
            (
                f"Retained category lớn nhất là {top_category_name}: "
                f"{top_category_orders:,} associated orders "
                f"({top_category_share:.2f}% trên retained order-item rows)."
            ),
        ]

        paradoxical_insights = []

        if delivered_share >= 90 and on_time_pct < 90:
            paradoxical_insights.append(
                f"Tỷ lệ delivered rất cao ({delivered_share:.2f}%), nhưng tỷ lệ giao "
                f"đúng/sớm hạn chỉ {on_time_pct:.2f}%. Nghĩa là hoàn tất đơn tốt không "
                f"đồng nghĩa với punctuality tốt."
            )

        if top_state_share >= max(15.0, top_category_share * 2):
            paradoxical_insights.append(
                f"Orders tập trung địa lý ở {top_state_name} ({top_state_share:.2f}%), "
                f"trong khi category đứng đầu chỉ chiếm {top_category_share:.2f}% retained "
                f"order-item rows. Dữ liệu cho thấy concentration theo geography mạnh hơn "
                f"concentration theo một category đơn lẻ."
            )

        if top_payment_share >= 50 and top_category_share < 20:
            paradoxical_insights.append(
                f"Payment behavior khá tập trung vào {top_payment_name} "
                f"({top_payment_share:.2f}%), nhưng retained product mix phân tán hơn: "
                f"category lớn nhất chỉ {top_category_share:.2f}%."
            )

        if not paradoxical_insights:
            paradoxical_insights = [
                "Không có pattern nghịch lý đủ mạnh để kết luận từ 5 diagnostic SQL hiện tại."
            ]

        short_term = [
            (
                f"Theo dõi service capacity tại {top_state_name}, vì bang này đang chiếm "
                f"{top_state_share:.2f}% order volume; ưu tiên kiểm tra backlog và SLA tại đây."
            ),
            (
                f"Điều tra nhóm khoảng {late_pct:.2f}% delivered orders giao sau estimated date "
                f"để xác định bottleneck theo state, seller hoặc category."
            ),
            (
                f"Kiểm tra tính liên tục của luồng thanh toán {top_payment_name}, vì nó gắn với "
                f"{top_payment_share:.2f}% retained payment rows."
            ),
        ]

        medium_term = [
            (
                "Phân rã delivery performance theo state × category × tháng để xác định "
                "late-delivery concentration thay vì tối ưu logistics dựa trên average toàn hệ thống."
            ),
            (
                "Theo dõi geographic concentration theo thời gian và so sánh order volume "
                "với service level trước khi tái phân bổ inventory/fulfillment resources."
            ),
            (
                "Xây dashboard kết hợp order status, delivery SLA, payment mix và retained "
                "category mix với cùng định nghĩa KPI."
            ),
        ]

        long_term = [
            (
                "Thiết kế network/fulfillment strategy có tính đến geographic concentration, "
                "nhưng chỉ mở rộng hoặc tái cấu trúc sau khi bổ sung profitability và cost-to-serve."
            ),
            (
                "Giảm phụ thuộc vận hành vào một payment path nếu phân tích failure/availability "
                "sau này xác nhận concentration hiện tại tạo rủi ro."
            ),
            (
                "Nếu có thể, quay lại dữ liệu normalized/raw để phục hồi basket-level, "
                "multi-item, multi-payment và repeat-customer metrics cho quyết định dài hạn."
            ),
        ]

        limitations = [
            (
                "Product/category evidence dựa trên retained product rows của transformed dataset; "
                "không đại diện đầy đủ cho toàn bộ basket gốc."
            ),
            (
                "Payment mix phản ánh một retained payment row/order và không đo số payment "
                "transactions gốc."
            ),
            (
                "Không thể suy ra profitability, causal demand, repeat-customer behavior hoặc "
                "basket size từ database hiện tại."
            ),
        ]

    else:
        answer = (
            f"The database contains {total_orders:,} orders, with {delivered_share:.2f}% "
            f"marked delivered. {top_state_name} accounts for {top_state_share:.2f}% "
            f"of orders. Among delivered orders with valid timestamps, average delivery "
            f"time is {avg_delivery_days:.2f} days and {on_time_pct:.2f}% are on or before "
            f"the estimated date. The leading retained payment type is {top_payment_name} "
            f"({top_payment_share:.2f}%), while the leading retained product category is "
            f"{top_category_name} ({top_category_share:.2f}% associated orders)."
        )

        basic_insights = [
            f"{delivered_share:.2f}% of orders are marked delivered.",
            f"{top_state_name} represents {top_state_share:.2f}% of all orders.",
            (
                f"Average delivery time is {avg_delivery_days:.2f} days and "
                f"{on_time_pct:.2f}% are delivered on/before the estimated date."
            ),
            f"{top_payment_name} is associated with {top_payment_share:.2f}% of retained payment rows.",
            (
                f"{top_category_name} is the largest retained category at "
                f"{top_category_share:.2f}% of associated orders."
            ),
        ]

        paradoxical_insights = []

        if delivered_share >= 90 and on_time_pct < 90:
            paradoxical_insights.append(
                f"Completion is high ({delivered_share:.2f}% delivered), yet punctuality "
                f"is lower ({on_time_pct:.2f}% on/before estimate)."
            )

        if top_state_share >= max(15.0, top_category_share * 2):
            paradoxical_insights.append(
                f"Geographic concentration ({top_state_share:.2f}% in {top_state_name}) "
                f"is much stronger than concentration in the leading retained category "
                f"({top_category_share:.2f}%)."
            )

        if not paradoxical_insights:
            paradoxical_insights = [
                "No defensible paradoxical pattern can be established from the current diagnostic evidence."
            ]

        short_term = [
            f"Monitor capacity and SLA in {top_state_name}, the largest order geography.",
            f"Investigate the approximately {late_pct:.2f}% of delivered orders that miss the estimated date.",
            f"Protect continuity for {top_payment_name}, the dominant retained payment type.",
        ]

        medium_term = [
            "Segment delivery performance by state × category × month.",
            "Track geographic concentration together with service level before reallocating resources.",
            "Create a unified KPI dashboard for status, delivery, payment mix and retained category mix.",
        ]

        long_term = [
            "Base fulfillment-network changes on geography plus profitability and cost-to-serve evidence.",
            "Assess payment-path concentration risk using failure and availability data.",
            "Restore normalized/raw data when possible to recover basket, multi-payment and repeat-customer metrics.",
        ]

        limitations = [
            "Product/category evidence refers to retained product rows in the transformed dataset.",
            "Payment mix reflects one retained payment row per order.",
            "The current database cannot establish profitability, causal demand, repeat-customer behavior or basket size.",
        ]

    return {
        "answer": answer,
        "sql": sql_queries,
        "data_tables": [
            {
                "query_number": item["query_number"],
                "sql": item["sql"],
                "records": item["df"].to_dict(orient="records"),
                "columns": item["df"].columns.tolist(),
            }
            for item in data_tables
        ],
        "audit": build_audit_report(sql_queries),
        "basic_insights": basic_insights,
        "paradoxical_insights": paradoxical_insights,
        "strategy": {
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
        },
        "limitations": limitations,
        "chart": ChartSpec(type="none").model_dump(),
        "error": False,
    }


# ============================================================
# 8. MAIN AGENT FUNCTION
# ============================================================

def ask_data_agent(question, history):
    q = question.lower()

    # Hard guard: revenue is ambiguous in this transformed database.
    mentions_revenue = "revenue" in q or "doanh thu" in q

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
                "1. **SUM(price)** — tổng `price` trên các retained order-item rows.\n"
                "2. **SUM(payment_value)** — tổng các retained payment values.\n\n"
                "Hai metric này không nên tự động được diễn giải là complete original-order revenue."
            ),
            limitation=(
                "Dataset đã transformation nên không có một metric complete original-order "
                "revenue được xác định chắc chắn."
            ),
        )

    # Broad business-advice requests use a deterministic SQL diagnostic
    # instead of the model's multi-tool loop. This is more reliable and
    # avoids malformed Groq tool-call JSON on complex open-ended prompts.
    if is_open_ended_business_request(question):
        return run_open_ended_business_analysis(question)

    agent, structured_llm = build_agent()

    agent_history = history[-10:] + [
        {
            "role": "user",
            "content": question,
        }
    ]

    try:
        result = agent.invoke({
            "messages": agent_history
        })
    except Exception as e:
        return {
            "answer": f"Agent error: {type(e).__name__}: {e}",
            "sql": [],
            "data_tables": [],
            "audit": [],
            "basic_insights": [],
            "paradoxical_insights": [],
            "strategy": empty_strategy(),
            "limitations": [
                "The agent could not complete this request."
            ],
            "chart": ChartSpec(type="none").model_dump(),
            "error": True,
        }

    final_answer = result["messages"][-1].content
    sql_queries = extract_sql_queries(result["messages"])

    # No SQL = clarification / unsupported / conversational response.
    if not sql_queries:
        return clarification_result(
            answer=final_answer,
            limitation="No SQL query was executed for this response.",
        )

    # Load all executed query results for UI + synthesis.
    data_tables = []

    for i, sql in enumerate(sql_queries, start=1):
        try:
            df = sql_to_dataframe(sql)
        except Exception:
            df = pd.DataFrame()

        data_tables.append({
            "query_number": i,
            "sql": sql,
            "df": df,
        })

    evidence = build_sql_evidence(sql_queries)
    chart_requested = detect_chart_requested(question)

    format_prompt = f"""
Convert the completed data analysis into a presentation-ready structured report.

USER QUESTION:
{question}

AGENT ANSWER:
{final_answer}

SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

REQUIREMENTS:

ANSWER
- Preserve database facts.
- Be concise.
- Never invent numbers, currency, causality, profitability, or customer preference.

BASIC INSIGHTS
- Always provide 1-3 concise basic insights.
- Each insight must be directly supported by SQL evidence.
- Include concrete numbers only when present in SQL evidence.

PARADOXICAL INSIGHTS
- Always provide 1-2 paradoxical insight items.
- A paradoxical insight means a surprising contrast, tension, exception,
  reversal, or counter-intuitive pattern that is supported by the SQL evidence.
- Never manufacture a paradox.
- If the current evidence cannot support one, return exactly one item saying:
  "No defensible paradoxical pattern can be established from the current SQL evidence."

STRATEGY
Always provide:
- short_term: 1-3 immediate operational actions
- medium_term: 1-3 actions involving analysis, process, or resource allocation
- long_term: 1-3 structural or strategic actions

Every strategy must connect to SQL evidence.
If evidence is too weak, say what should be analyzed before acting.

LIMITATIONS
- Include only material limitations.
- Respect transformed-data limitations in the codebook.

CHART
Chart requested: {chart_requested}

If chart_requested is False:
- chart.type MUST be "none".

If chart_requested is True:
- choose a chart only if SQL evidence supports it.
- x and y must exactly match SQL-result column names.
"""

    try:
        presentation = structured_llm.invoke(format_prompt)
    except Exception:
        presentation = Presentation(
            answer=final_answer,
            basic_insights=[
                "The SQL result provides the direct evidence summarized in the answer."
            ],
            paradoxical_insights=[
                "No defensible paradoxical pattern can be established from the current SQL evidence."
            ],
            strategy=StrategyPlan(
                short_term=[
                    "Use the current result as a diagnostic signal, not as a standalone decision rule."
                ],
                medium_term=[
                    "Run additional segmented analyses before reallocating resources."
                ],
                long_term=[
                    "Build decisions around repeated evidence across several KPIs and time periods."
                ],
            ),
            limitations=[],
            chart=ChartSpec(type="none"),
        )

    # Deterministic chart safety/fallback.
    if not chart_requested:
        presentation.chart = ChartSpec(type="none")
    else:
        presentation.chart = deterministic_chart_fallback(
            question=question,
            chart=presentation.chart,
            data_tables=data_tables,
        )

    return {
        "answer": presentation.answer,
        "sql": sql_queries,
        "data_tables": [
            {
                "query_number": item["query_number"],
                "sql": item["sql"],
                "records": item["df"].to_dict(orient="records"),
                "columns": item["df"].columns.tolist(),
            }
            for item in data_tables
        ],
        "audit": build_audit_report(sql_queries),
        "basic_insights": presentation.basic_insights,
        "paradoxical_insights": presentation.paradoxical_insights,
        "strategy": presentation.strategy.model_dump(),
        "limitations": presentation.limitations,
        "chart": presentation.chart.model_dump(),
        "error": False,
    }


# ============================================================
# 9. MULTI-CHAT SESSION STATE
# ============================================================

def create_new_chat():
    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
    }

    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    if chat_id not in st.session_state.chats:
        return

    del st.session_state.chats[chat_id]

    if not st.session_state.chats:
        create_new_chat()
        return

    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = next(
            reversed(st.session_state.chats)
        )


if "chats" not in st.session_state:
    st.session_state.chats = {}

if (
    "current_chat_id" not in st.session_state
    or st.session_state.current_chat_id not in st.session_state.chats
):
    create_new_chat()


# ============================================================
# 10. SIDEBAR
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
st.sidebar.caption("Conversation history")

# List newest chats first.
for chat_id, chat in reversed(
    list(st.session_state.chats.items())
):
    title = chat["title"]

    cols = st.sidebar.columns([0.82, 0.18])

    with cols[0]:
        is_current = (
            chat_id == st.session_state.current_chat_id
        )

        label = f"● {title}" if is_current else title

        if st.button(
            label,
            key=f"open_chat_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    with cols[1]:
        if st.button(
            "🗑️",
            key=f"delete_chat_{chat_id}",
            help="Delete chat",
            use_container_width=True,
        ):
            delete_chat(chat_id)
            st.rerun()


current_chat = st.session_state.chats[
    st.session_state.current_chat_id
]

history = current_chat["messages"]


# ============================================================
# 11. UI RENDER HELPERS
# ============================================================

def records_to_df(table):
    return pd.DataFrame(table.get("records", []))


def find_chart_dataframe(result):
    chart = result.get("chart", {})
    x = chart.get("x")
    y = chart.get("y")

    if not x or not y:
        return None

    for table in reversed(result.get("data_tables", [])):
        df = records_to_df(table)

        if x in df.columns and y in df.columns:
            return df

    return None


def render_chart(result):
    chart = result.get("chart", {})
    chart_type = chart.get("type", "none")

    if chart_type == "none":
        return

    df = find_chart_dataframe(result)

    if df is None or df.empty:
        return

    x = chart.get("x")
    y = chart.get("y")
    title = chart.get("title")

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


def render_bullets(items, empty_text="No evidence-based item available."):
    if not items:
        st.info(empty_text)
        return

    for item in items:
        st.markdown(f"- {item}")


def render_audit(result):
    audit = result.get("audit", [])

    with st.expander(
        "🔍 Biên bản Kiểm định Dữ liệu (Auto-Audit Workflow)",
        expanded=False,
    ):
        if not audit:
            st.info(
                "Không có SQL query nào được chạy trong lượt này."
            )
            return

        safe_count = sum(
            1 for item in audit
            if item["status"] == "SAFE"
        )
        warning_count = sum(
            1 for item in audit
            if item["status"] == "WARNING"
        )
        blocked_count = sum(
            1 for item in audit
            if item["status"] == "BLOCKED"
        )

        st.markdown(
            f"**Queries:** {len(audit)}  |  "
            f"**SAFE:** {safe_count}  |  "
            f"**WARNING:** {warning_count}  |  "
            f"**BLOCKED:** {blocked_count}"
        )

        for item in audit:
            st.markdown(
                f"**Query {item['query_number']} — {item['status']}**"
            )

            for message in item["messages"]:
                st.markdown(f"- {message}")


def render_report(result):
    if result.get("error"):
        st.error(result["answer"])
        return

    st.success(
        "💡 Hệ thống AI đã bóc tách dữ liệu và hoàn tất báo cáo. "
        "Xem chi tiết tại các tab bên dưới."
    )

    render_audit(result)

    insight_tab, strategy_tab, sql_tab = st.tabs([
        "📊 Báo cáo Phân tích (Insight)",
        "💡 Đề xuất Chiến lược",
        "⚙️ Tiến trình SQL",
    ])

    with insight_tab:
        st.markdown("### Kết luận")
        st.markdown(result.get("answer", ""))

        data_tables = result.get("data_tables", [])

        if data_tables:
            st.markdown("### Dữ liệu truy xuất")

            for table in data_tables:
                df = records_to_df(table)

                if df.empty:
                    continue

                if len(data_tables) > 1:
                    st.markdown(
                        f"**Query {table['query_number']}**"
                    )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

        render_chart(result)

        st.markdown("### 1. Insight cơ bản")
        render_bullets(
            result.get("basic_insights", []),
            "Chưa có insight cơ bản dựa trên SQL.",
        )

        st.markdown("### 2. Insight nghịch lý")
        render_bullets(
            result.get("paradoxical_insights", []),
            "Chưa có insight nghịch lý đáng tin cậy.",
        )

        limitations = result.get("limitations", [])

        if limitations:
            with st.expander("⚠️ Giới hạn diễn giải"):
                render_bullets(limitations)

    with strategy_tab:
        strategy = result.get(
            "strategy",
            empty_strategy(),
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### ⚡ Ngắn hạn")
            render_bullets(
                strategy.get("short_term", []),
                "Chưa có đề xuất ngắn hạn.",
            )

        with col2:
            st.markdown("### 🧭 Trung hạn")
            render_bullets(
                strategy.get("medium_term", []),
                "Chưa có đề xuất trung hạn.",
            )

        with col3:
            st.markdown("### 🏗️ Dài hạn")
            render_bullets(
                strategy.get("long_term", []),
                "Chưa có đề xuất dài hạn.",
            )

    with sql_tab:
        sql_queries = result.get("sql", [])

        if not sql_queries:
            st.info("Không có SQL query trong lượt này.")

        for i, sql in enumerate(
            sql_queries,
            start=1,
        ):
            validation = validate_sql_query(sql)

            st.markdown(
                f"### Query {i} — {validation['status']}"
            )

            st.code(
                sql,
                language="sql",
            )

            for message in validation["messages"]:
                st.caption(message)


def render_message(message):
    role = message["role"]

    with st.chat_message(role):
        if role == "user":
            st.markdown(message["content"])
            return

        result = message.get("result")

        if result:
            render_report(result)
        else:
            st.markdown(message.get("content", ""))


def make_agent_history(messages):
    """
    Convert stored UI messages to the minimal role/content messages
    the LangChain agent needs for conversational follow-ups.
    """
    converted = []

    for message in messages[-10:]:
        converted.append({
            "role": message["role"],
            "content": message.get("content", ""),
        })

    return converted


# ============================================================
# 12. MAIN PAGE
# ============================================================

st.title("E-commerce AI Data Analyst")

st.caption(
    "Groq + LangChain + SQLite • Read-only SQL • Auto-Audit"
)

for message in history:
    render_message(message)


question = st.chat_input(
    "Ask a question about the e-commerce data..."
)

if question:
    # Display immediately.
    with st.chat_message("user"):
        st.markdown(question)

    prior_history = make_agent_history(history)

    with st.spinner("Analyzing database..."):
        result = ask_data_agent(
            question=question,
            history=prior_history,
        )

    # Save complete message state so switching chats reproduces
    # the report/tabs, not only the final prose answer.
    history.append({
        "role": "user",
        "content": question,
    })

    history.append({
        "role": "assistant",
        "content": result["answer"],
        "result": result,
    })

    # Auto-title new conversation.
    if current_chat["title"] == "New chat":
        clean_title = " ".join(
            question.strip().split()
        )

        if len(clean_title) > 34:
            clean_title = clean_title[:34] + "..."

        current_chat["title"] = clean_title

    # Rerun so the newly stored structured report is rendered
    # through the same history rendering path.
    st.rerun()
