
import re
import json
import sqlite3
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
# 1. PATHS
# ============================================================

DB_PATH = Path("data/processed/ecommerce_clean.db").resolve()
CODEBOOK_PATH = Path("ecommerce_agent_codebook.md")

codebook_text = CODEBOOK_PATH.read_text(encoding="utf-8")


# ============================================================
# 2. SQL SAFETY
# ============================================================

FORBIDDEN_SQL = {
    "INSERT", "UPDATE", "DELETE", "DROP",
    "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "VACUUM",
    "REINDEX", "PRAGMA"
}


def validate_readonly_sql(sql: str):
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


def validate_sql_query(sql: str):
    safe, reason = validate_readonly_sql(sql)

    if not safe:
        return {
            "status": "BLOCKED",
            "messages": [reason]
        }

    normalized = " ".join(sql.lower().split())

    # Does SQL actually compile?
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
            ]
        }

    warnings = []

    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)",
        normalized
    ):
        warnings.append(
            "SUM(price) is the sum of prices on retained "
            "order-item rows, not guaranteed complete order revenue."
        )

    if re.search(
        r"\b(sum|avg)\s*\(\s*(?:\w+\.)?payment_value\s*\)",
        normalized
    ):
        warnings.append(
            "payment_value comes from one retained payment row "
            "per order and is not guaranteed complete order revenue."
        )

    if (
        "product_category_name" in normalized
        and "payment_value" in normalized
    ):
        warnings.append(
            "Category × payment_value is high-risk because category "
            "belongs to the retained product row."
        )

    if (
        "order_delivered_timestamp" in normalized
        and "is not null" not in normalized
    ):
        warnings.append(
            "Delivery analysis should normally require "
            "order_delivered_timestamp IS NOT NULL."
        )

    if warnings:
        return {
            "status": "WARNING",
            "messages": warnings
        }

    return {
        "status": "SAFE",
        "messages": ["SQL passed validation."]
    }


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def list_tables() -> str:
    """List all tables and views in the e-commerce database."""

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        rows = conn.execute("""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY type, name;
        """).fetchall()

    return "\n".join(
        f"{name} ({kind})"
        for name, kind in rows
    )


@tool
def get_schema(table_name: str) -> str:
    """Return schema for a table or view."""

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        row = conn.execute("""
            SELECT sql
            FROM sqlite_master
            WHERE name = ?
            AND type IN ('table', 'view');
        """, (table_name,)).fetchone()

    if row is None:
        return f"{table_name} does not exist."

    return row[0]


@tool
def validate_sql(sql: str) -> str:
    """Validate SQL before execution. Always call this before execute_sql."""

    result = validate_sql_query(sql)

    text = [f"STATUS: {result['status']}"]

    for message in result["messages"]:
        text.append(f"- {message}")

    return "\n".join(text)


@tool
def execute_sql(sql: str) -> str:
    """Execute validated read-only SQL against the database."""

    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        return (
            "SQL BLOCKED: "
            + " ".join(validation["messages"])
        )

    db_uri = f"file:{DB_PATH}?mode=ro"

    try:
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(sql)

            rows = cursor.fetchmany(51)

            columns = [
                x[0] for x in cursor.description
            ]

        rows = rows[:50]

        result = [
            {col: row[col] for col in columns}
            for row in rows
        ]

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

    except Exception as e:
        return f"SQL ERROR: {type(e).__name__}: {e}"


TOOLS = [
    list_tables,
    get_schema,
    validate_sql,
    execute_sql
]


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an AI Data Analyst for one specific e-commerce SQLite database.

You have tools for:
- listing tables,
- reading schemas,
- validating SQL,
- executing SQL.

CORE RULES:

1. Never invent numerical values.
2. Every dataset number must come from executed SQL.
3. Always validate new SQL before executing it.
4. If SQL is BLOCKED, fix it before execution.
5. Only use read-only SQL.
6. If SQL returns an error, diagnose and retry.
7. Do not claim correlation is causation.

SEMANTIC RULES:

- orders has one row per order.
- order_items contains one retained row per order.
- payments contains one retained payment row per order.
- products contains one metadata row per product_id.
- customers contains one transformed customer row.

UNSUPPORTED:
- basket size
- original number of items per order
- original number of payments per order
- repeat-customer rate / retention

MONEY RULES:

If user says "revenue" without defining it,
DO NOT choose automatically.

Ask whether they mean:

A. SUM(price)
B. SUM(payment_value)

Never invent a currency.
Do not add $, €, R$, £, etc.

SUM(price) means:
sum of price on retained order-item rows.

SUM(payment_value) means:
sum of retained payment values.

Do not automatically describe either one as true complete order revenue.

PRODUCT RULE:

product_category_name belongs to the retained product row.
Do not imply it represents every item originally contained in the order.

DELIVERY RULE:

For delivery duration/performance:
- normally use delivered orders
- require order_delivered_timestamp IS NOT NULL

DATA vs INSIGHT vs STRATEGY:

DATA = SQL evidence.
INSIGHT = interpretation of evidence.
STRATEGY = possible business action supported by evidence.

Do not propose a strong strategy if the available analysis does not support it.

CHART RULE:

If the user requests a chart, graph, plot, or visualization:
- query the data needed for the chart,
- DO NOT write chart JSON,
- DO NOT write a chart specification in the textual answer,
- the Streamlit application will render the chart from the SQL result.

DATABASE CODEBOOK:

{codebook_text}
"""


# ============================================================
# 5. STRUCTURED OUTPUT
# ============================================================

class ChartSpec(BaseModel):
    type: Literal[
        "none",
        "bar",
        "line",
        "scatter",
        "pie"
    ] = "none"

    x: Optional[str] = None
    y: Optional[str] = None
    title: Optional[str] = None


class StrategyPlan(BaseModel):
    short_term: str = ""
    medium_term: str = ""
    long_term: str = ""


class Presentation(BaseModel):
    answer: str

    basic_insight: str = ""

    paradoxical_insight: str = ""

    strategy: StrategyPlan = Field(
        default_factory=StrategyPlan
    )

    limitations: List[str] = Field(
        default_factory=list
    )

    chart: ChartSpec = Field(
        default_factory=ChartSpec
    )


# ============================================================
# 6. BUILD AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT
    )

    structured_llm = llm.with_structured_output(
        Presentation,
        method="function_calling"
    )

    return agent, structured_llm


# ============================================================
# 7. EXTRACT SQL FROM AGENT TRACE
# ============================================================

def extract_sql_queries(messages):

    queries = []

    for message in messages:

        tool_calls = getattr(
            message,
            "tool_calls",
            None
        )

        if not tool_calls:
            continue

        for call in tool_calls:

            if call.get("name") == "execute_sql":

                sql = call.get(
                    "args",
                    {}
                ).get("sql")

                if sql:
                    queries.append(sql)

    return queries


# ============================================================
# 8. LOAD LAST SQL RESULT AS DATA
# ============================================================

def sql_to_records(sql: str):

    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        return []

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        df = pd.read_sql_query(sql, conn)

    return df.head(200).to_dict(
        orient="records"
    )


# ============================================================
# 9. MAIN AGENT FUNCTION
# ============================================================

def ask_data_agent(
    question,
    history
):
    # ========================================================
    # HARD GUARD: ambiguous revenue definition
    # Do not rely on the LLM to obey this rule.
    # ========================================================

    q = question.lower()

    mentions_revenue = "revenue" in q

    explicitly_price = (
        "sum(price)" in q
        or "sum of price" in q
        or "using price" in q
    )

    explicitly_payment = (
        "sum(payment_value)" in q
        or "sum of payment_value" in q
        or "using payment_value" in q
    )

    if mentions_revenue and not explicitly_price and not explicitly_payment:
        return {
            "answer": (
                "Revenue is ambiguous in this transformed dataset. "
                "Which definition would you like to use?\n\n"
                "1. **SUM(price)** — sum of prices on the retained order-item rows.\n"
                "2. **SUM(payment_value)** — sum of recorded retained payment values.\n\n"
                "Neither should automatically be interpreted as complete original-order revenue."
            ),
            "sql": [],
            "data": [],
            "insights": [],
            "strategies": [],
            "limitations": [],
            "chart": {
                "type": "none",
                "x": None,
                "y": None,
                "title": None
            }
        }

    agent, structured_llm = build_agent()

    messages = history[-10:] + [
        {
            "role": "user",
            "content": question
        }
    ]

    result = agent.invoke({
        "messages": messages
    })

    final_answer = result[
        "messages"
    ][-1].content

    sql_queries = extract_sql_queries(
        result["messages"]
    )

    # If no SQL was executed, the agent may be asking
    # a clarification or explaining an unsupported metric.
    # Preserve its answer exactly instead of sending empty
    # SQL data to the formatting LLM.
    if not sql_queries:
        return {
            "answer": final_answer,
            "sql": [],
            "data": [],
            "insights": [],
            "strategies": [],
            "limitations": [],
            "chart": {
                "type": "none",
                "x": None,
                "y": None,
                "title": None
            }
        }

    data = []

    if sql_queries:
        try:
            data = sql_to_records(
                sql_queries[-1]
            )
        except Exception:
            data = []

    chart_requested = any(
        word in question.lower()
        for word in [
            "chart",
            "graph",
            "plot",
            "visual",
            "visualization",
            "biểu đồ",
            "vẽ"
        ]
    )

    format_prompt = f"""
Convert the completed data analysis into structured output.

USER QUESTION:
{question}

AGENT ANSWER:
{final_answer}

SQL DATA:
{json.dumps(data[:50], ensure_ascii=False, default=str)}

Rules:

- Preserve all numerical facts from SQL/database.
- Never invent numbers.
- Never invent currency.

BASIC INSIGHT:
- Always provide one concise basic insight.
- It must directly summarize the most important pattern supported by SQL.

PARADOXICAL INSIGHT:
- Always provide one paradoxical insight section.
- Look for a surprising contrast, tension, exception, reversal, or counter-intuitive pattern supported by the SQL evidence.
- Do NOT manufacture a paradox.
- If the current SQL evidence is insufficient, explicitly say:
  "No defensible paradoxical pattern can be established from this query alone."

STRATEGY:
Always return three horizons:

1. short_term:
   an immediate operational action.

2. medium_term:
   an action requiring additional analysis, process change, or resource allocation.

3. long_term:
   a structural or strategic action.

Every strategy must be connected to the observed evidence.

If the query is too simple to support a business recommendation,
do NOT invent one. Instead state what additional analysis is needed before acting.

Do not infer demand, customer preference, causality, or profitability
from an aggregate ranking alone.

Limitations should mention only material limitations.

Chart requested: {chart_requested}

If chart_requested is False:
chart.type MUST be "none".

If chart_requested is True:
choose a chart only when SQL data supports it.

Chart x and y MUST exactly match column names in SQL DATA.

Chart requested: {chart_requested}

If chart_requested is False:
chart.type MUST be "none".

If chart_requested is True:
choose a chart only when the SQL data can support it.

Chart x and y MUST exactly match column names in SQL DATA.
"""

    try:
        presentation = structured_llm.invoke(
            format_prompt
        )

    except Exception:

        presentation = Presentation(
            answer=final_answer,
            basic_insight="Insufficient structured evidence for an additional insight.",
            paradoxical_insight=(
                "No defensible paradoxical pattern can be established "
                "from this query alone."
            ),
            strategy=StrategyPlan(
                short_term="No immediate action should be taken from this result alone.",
                medium_term="Run a more detailed segmented analysis before making a decision.",
                long_term="Use repeated evidence across multiple analyses before changing long-term strategy."
            ),
            limitations=[],
            chart=ChartSpec(type="none")
        )

    # ------------------------------------------------------------
    # Deterministic chart fallback
    # ------------------------------------------------------------
    
    if not chart_requested:
        presentation.chart = ChartSpec(type="none")
    
    elif data:
    
        df_chart = pd.DataFrame(data)
    
        numeric_cols = df_chart.select_dtypes(
            include="number"
        ).columns.tolist()
    
        categorical_cols = [
            col for col in df_chart.columns
            if col not in numeric_cols
        ]
    
        current_chart = presentation.chart
    
        # Check whether LLM-generated chart specification is usable
        chart_valid = (
            current_chart.type != "none"
            and current_chart.x in df_chart.columns
            and current_chart.y in df_chart.columns
        )
    
        # If not, Python chooses a sensible chart
        if not chart_valid:
    
            q_lower = question.lower()
    
            # Relationship between two numeric variables -> scatter
            if (
                ("relationship" in q_lower or "scatter" in q_lower)
                and len(numeric_cols) >= 2
            ):
                presentation.chart = ChartSpec(
                    type="scatter",
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title="Data Relationship"
                )
    
            # Category + numeric measure -> bar
            elif categorical_cols and numeric_cols:
                presentation.chart = ChartSpec(
                    type="bar",
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                )
    
            # Two numeric columns -> scatter
            elif len(numeric_cols) >= 2:
                presentation.chart = ChartSpec(
                    type="scatter",
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title="Data Relationship"
                )
    
            else:
                presentation.chart = ChartSpec(type="none")

    return {
        "answer": presentation.answer,
        "sql": sql_queries,
        "data": data,
        "basic_insight": presentation.basic_insight,
        "paradoxical_insight": presentation.paradoxical_insight,
        "strategy": presentation.strategy.model_dump(),
        "limitations": presentation.limitations,
        "chart": presentation.chart.model_dump()
    }


# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="E-commerce AI Data Analyst",
    layout="wide"
)

st.title("E-commerce AI Data Analyst")

st.caption(
    "Groq + LangChain + SQLite • Read-only SQL"
)


# ============================================================
# MULTI-CHAT STATE
# ============================================================

import uuid


def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": []
    }

    st.session_state.current_chat_id = chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    create_new_chat()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Chats")

if st.sidebar.button(
    "＋ New chat",
    use_container_width=True
):
    create_new_chat()
    st.rerun()


st.sidebar.markdown("---")

for chat_id, chat in reversed(
    list(st.session_state.chats.items())
):

    title = chat["title"]

    if st.sidebar.button(
        title,
        key=f"chat_{chat_id}",
        use_container_width=True
    ):
        st.session_state.current_chat_id = chat_id
        st.rerun()


current_chat = st.session_state.chats[
    st.session_state.current_chat_id
]

history = current_chat["messages"]


# DISPLAY PREVIOUS CHAT
for msg in history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# USER QUESTION
question = st.chat_input(
    "Ask a question about the e-commerce data..."
)


if question:

    with st.chat_message("user"):
        st.markdown(question)

    prior_history = list(history)
    
    with st.spinner(
        "Analyzing database..."
    ):

        result = ask_data_agent(
            question,
            prior_history
        )

    answer = result["answer"]

    with st.chat_message("assistant"):

        st.markdown(answer)

        # DATA TABLE
        if result["data"]:

            df_result = pd.DataFrame(
                result["data"]
            )

            st.subheader("Data")
            st.dataframe(
                df_result,
                use_container_width=True
            )

            # CHART
            chart = result["chart"]

            chart_type = chart.get("type")
            x = chart.get("x")
            y = chart.get("y")

            if (
                chart_type != "none"
                and x in df_result.columns
                and y in df_result.columns
            ):

                st.subheader("Visualization")

                if chart_type == "bar":
                    fig = px.bar(
                        df_result,
                        x=x,
                        y=y,
                        title=chart.get("title")
                    )

                elif chart_type == "line":
                    fig = px.line(
                        df_result,
                        x=x,
                        y=y,
                        title=chart.get("title")
                    )

                elif chart_type == "scatter":
                    fig = px.scatter(
                        df_result,
                        x=x,
                        y=y,
                        title=chart.get("title")
                    )

                elif chart_type == "pie":
                    fig = px.pie(
                        df_result,
                        names=x,
                        values=y,
                        title=chart.get("title")
                    )

                else:
                    fig = None

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        # ============================================================
        # INSIGHTS
        # ============================================================
        
        st.subheader("Insights")
        
        st.markdown("**Basic Insight**")
        st.write(result["basic_insight"])
        
        st.markdown("**Paradoxical Insight**")
        st.write(result["paradoxical_insight"])
        
        
        # ============================================================
        # STRATEGY
        # ============================================================
        
        st.subheader("Strategy")
        
        strategy = result["strategy"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Short-term")
            st.write(strategy.get("short_term", ""))
        
        with col2:
            st.markdown("### Medium-term")
            st.write(strategy.get("medium_term", ""))
        
        with col3:
            st.markdown("### Long-term")
            st.write(strategy.get("long_term", ""))

        # LIMITATIONS
        if result["limitations"]:

            with st.expander(
                "Limitations"
            ):
                for item in result["limitations"]:
                    st.write("•", item)

        # SQL TRANSPARENCY
        if result["sql"]:

            with st.expander(
                "SQL used"
            ):

                for i, sql in enumerate(
                    result["sql"],
                    start=1
                ):

                    st.code(
                        sql,
                        language="sql"
                    )


    # Save conversation context
    history.append({
        "role": "user",
        "content": question
    })
    
    history.append({
        "role": "assistant",
        "content": answer
    })

    # Automatically name a new conversation
    if current_chat["title"] == "New chat":
    
        clean_title = question.strip()
    
        if len(clean_title) > 32:
            clean_title = clean_title[:32] + "..."
    
        current_chat["title"] = clean_title
