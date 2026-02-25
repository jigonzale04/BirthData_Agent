import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

st.set_page_config(layout="wide")
st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")


# =============================
# DATA LOADING
# =============================

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("Provisional_Natality_2025_CDC.csv")
    except Exception:
        return None


df_raw = load_data()

if df_raw is None:
    st.error("Dataset file not found in repository.")
    st.stop()

df = df_raw.copy()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

required_cols = [
    "state_of_residence",
    "month",
    "month_code",
    "year_code",
    "sex_of_infant",
    "births",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["births"] = pd.to_numeric(df["births"], errors="coerce")
df = df.dropna(subset=["births"])


# =============================
# SIDEBAR FILTERS
# =============================

st.sidebar.header("Filters")

states = sorted(df["state_of_residence"].dropna().unique())
genders = sorted(df["sex_of_infant"].dropna().unique())
months = sorted(df["month"].dropna().unique())

state_sel = st.sidebar.multiselect("State", ["All"] + states, default=["All"])
gender_sel = st.sidebar.multiselect("Gender", ["All"] + genders, default=["All"])
month_sel = st.sidebar.multiselect("Month", ["All"] + months, default=["All"])

filtered = df.copy()

if "All" not in state_sel:
    filtered = filtered[filtered["state_of_residence"].isin(state_sel)]

if "All" not in gender_sel:
    filtered = filtered[filtered["sex_of_infant"].isin(gender_sel)]

if "All" not in month_sel:
    filtered = filtered[filtered["month"].isin(month_sel)]

if filtered.empty:
    st.warning("No data matches selected filters.")
    st.stop()


# =============================
# VISUALIZATION
# =============================

agg = (
    filtered.groupby(["state_of_residence", "sex_of_infant"])["births"]
    .sum()
    .reset_index()
)

fig = px.bar(
    agg,
    x="state_of_residence",
    y="births",
    color="sex_of_infant",
    title="Total Births by State and Gender",
    template="plotly_white",
)

fig.update_layout(
    xaxis_title="State",
    yaxis_title="Births",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Records")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True, hide_index=True)


# =============================
# AI DATA ANALYST SECTION
# =============================

st.markdown("---")
st.header("AI Data Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask an analytical question about the data...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Precompute structured aggregates
    total_births = int(filtered["births"].sum())

    state_totals = (
        filtered.groupby("state_of_residence")["births"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    gender_totals = (
        filtered.groupby("sex_of_infant")["births"]
        .sum()
        .to_dict()
    )

    monthly_totals = (
        filtered.groupby("month")["births"]
        .sum()
        .to_dict()
    )

    context = {
        "total_births": total_births,
        "state_totals": state_totals,
        "gender_totals": gender_totals,
        "monthly_totals": monthly_totals,
        "filters_applied": {
            "states": state_sel,
            "gender": gender_sel,
            "months": month_sel,
        },
    }

    system_prompt = f"""
You are a senior data analyst writing executive insights.

Use ONLY the structured dataset context provided below.
Do NOT fabricate numbers.
If the answer cannot be determined from the data, clearly say so.

Your response must:
- Be text only
- Provide interpretation, not raw data dumps
- Explain trends and comparisons
- Highlight meaningful differences
- Suggest implications when relevant
- Be concise (3–5 short paragraphs max)

Dataset Context:
{json.dumps(context, indent=2)}
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['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']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "temperature": 0.2,
            },
        )

        result = response.json()["choices"][0]["message"]["content"]

    except Exception:
        result = "The AI analyst is currently unavailable."

    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
