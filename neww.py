import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Aspect Marketing Adviser", layout="wide")
DEFAULT_FILE = "campaign data since may.xlsx"   # you can change this
CURRENCY = "£"
DEFAULT_TARGET_CPA = 100.0                      # your benchmark; editable in UI

# =========================
# DATA LOADING & METRICS
# =========================
@st.cache_data(show_spinner=False)
def load_data(file_or_path):
    """Load Excel/CSV and normalize columns + metrics."""
    if hasattr(file_or_path, "read"):   # UploadedFile
        df = pd.read_excel(file_or_path)
    else:
        if not os.path.exists(file_or_path):
            raise FileNotFoundError(f"File not found: {file_or_path}")
        if str(file_or_path).lower().endswith(".csv"):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_excel(file_or_path)

    # normalize columns
    df = df.rename(columns={
        "Cost": "Spend",
        "Impr.": "Impr",
        "Impressions": "Impr"
    })
    req = ["Day", "Campaign", "Spend", "Conversions", "Clicks", "Impr"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {req}")

    # types
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    df = df.dropna(subset=["Day"]).copy()
    for c in ["Spend", "Conversions", "Clicks", "Impr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # period columns
    df["Month"] = df["Day"].dt.to_period("M").astype(str)

    # derived metrics (row-level)
    df["CPC"] = np.where(df["Clicks"] > 0, df["Spend"]/df["Clicks"], np.nan)
    df["CTR"] = np.where(df["Impr"]   > 0, df["Clicks"]/df["Impr"],   np.nan)
    df["CVR"] = np.where(df["Clicks"] > 0, df["Conversions"]/df["Clicks"], np.nan)
    df["CPA"] = np.where(df["Conversions"] > 0, df["Spend"]/df["Conversions"], np.nan)
    return df

def add_profitability_labels(df, target_cpa):
    df = df.copy()
    def decide(conv, cpa):
        if conv == 0 or pd.isna(cpa):
            return "Bad (No conversions)"
        return "Good" if cpa < target_cpa else "Bad (Unprofitable)"
    df["Profitability"] = [decide(c, a) for c, a in zip(df["Conversions"], df["CPA"])]
    return df

def month_summary(df, month):
    m = df[df["Month"] == month]
    total = float(m["Spend"].sum())
    good  = float(m.loc[m["Profitability"]=="Good", "Spend"].sum())
    bad   = float(m.loc[m["Profitability"].str.contains("Bad", na=False), "Spend"].sum())
    return {"total": total, "good": good, "bad": bad, "savings": bad}

def money(x): return f"{CURRENCY}{x:,.2f}"

# =========================
# BOT (offline intent router; optional LLM)
# =========================
def bot_answer(df, month, target_cpa, text):
    q = (text or "").strip().lower()
    s = month_summary(df, month)
    if any(k in q for k in ["waste", "saving", "bad"]):
        pct = (s["bad"]/s["total"]*100) if s["total"]>0 else 0
        return f"{month}: Bad spend {money(s['bad'])} ({pct:.1f}%). Potential savings {money(s['savings'])}."
    if "good" in q:
        return f"{month}: Good spend {money(s['good'])} of total {money(s['total'])}."
    if "cpa" in q:
        m = df[df["Month"]==month]
        cpa = m["CPA"].mean(skipna=True)
        return f"{month}: Avg CPA ~ {money(cpa) if not np.isnan(cpa) else 'n/a'} vs target {money(target_cpa)}."
    if "top" in q and "campaign" in q:
        m = df[df["Month"]==month]
        top = (m.groupby(["Campaign","Profitability"], dropna=False)
                 .agg(spend=("Spend","sum"), conv=("Conversions","sum"))
                 .reset_index()
                 .sort_values("spend", ascending=False)
                 .head(10))
        lines = [f"- {r.Campaign}: {money(r.spend)} • {r.Profitability} • conv {int(r.conv)}"
                 for _, r in top.iterrows()]
        return "Top campaigns by spend:\n" + "\n".join(lines)
    return f"{month}: Total {money(s['total'])} • Good {money(s['good'])} • Bad {money(s['bad'])} • Savings {money(s['savings'])}."

def llm_answer(context, question):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a helpful marketing analyst. Answer clearly and briefly using the context.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# =========================
# UI
# =========================
st.title("📊 Aspect Marketing Adviser")

# ---- Sidebar: file + settings
st.sidebar.header("Data")
default_choice = DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None
sel = st.sidebar.selectbox("Excel file path", [default_choice] if default_choice else [])
uploaded = st.sidebar.file_uploader("…or upload Excel (.xlsx)", type=["xlsx"])
target_cpa = st.sidebar.number_input("Target CPA (customer value)", value=float(DEFAULT_TARGET_CPA), step=10.0)

# Load
if uploaded is not None:
    df_raw = load_data(uploaded)
elif sel:
    df_raw = load_data(sel)
else:
    st.info("Upload or select your Excel to begin.")
    st.stop()

# Labels
df = add_profitability_labels(df_raw, target_cpa)

# Month picker
months = sorted(df["Month"].unique())
sel_month = st.sidebar.selectbox("Select month", months, index=len(months)-1)

# =========================
# KPIs
# =========================
st.subheader(f"Executive Summary — {sel_month}")
k1, k2, k3, k4 = st.columns(4)
summ = month_summary(df, sel_month)
k1.metric("Total Spend",           money(summ["total"]))
k2.metric("Good Spend (BOOST)",    money(summ["good"]))
k3.metric("Bad Spend (PAUSE)",     money(summ["bad"]))
k4.metric("Potential Savings",     money(summ["savings"]))

st.divider()

# =========================
# TRENDS
# =========================
st.subheader("Trends")

# Daily trends (selected month)
md = df[df["Month"] == sel_month].copy()
daily = (md.groupby("Day")
           .agg(Spend=("Spend","sum"),
                Clicks=("Clicks","sum"),
                Conversions=("Conversions","sum"),
                Impr=("Impr","sum"))
           .reset_index())
daily["CPC"] = np.where(daily["Clicks"]>0, daily["Spend"]/daily["Clicks"], np.nan)
daily["CPA"] = np.where(daily["Conversions"]>0, daily["Spend"]/daily["Conversions"], np.nan)
daily["CTR"] = np.where(daily["Impr"]>0, daily["Clicks"]/daily["Impr"], np.nan)
daily["CVR"] = np.where(daily["Clicks"]>0, daily["Conversions"]/daily["Clicks"], np.nan)

left, right = st.columns(2)

with left:
    st.markdown("**Daily Spend / Clicks / Conversions**")
    base = alt.Chart(daily).encode(x="Day:T")
    chart1 = alt.layer(
        base.mark_line().encode(y=alt.Y("Spend:Q", title="Spend")),
        base.mark_line().encode(y=alt.Y("Clicks:Q", title="Clicks")),
        base.mark_line().encode(y=alt.Y("Conversions:Q", title="Conversions")),
    ).resolve_scale(y='independent').interactive()
    st.altair_chart(chart1, use_container_width=True)

with right:
    st.markdown("**Daily CPA vs Target + CPC/CTR/CVR**")
    target_line = pd.DataFrame({"y":[target_cpa], "label":["Target CPA"]})
    chart2 = alt.Chart(daily).mark_line().encode(x="Day:T", y=alt.Y("CPA:Q", title="CPA"))
    line_target = alt.Chart(target_line).mark_rule().encode(y="y:Q").properties(title="CPA (rule) vs Target")
    st.altair_chart(alt.layer(chart2, line_target).interactive(), use_container_width=True)
    # Secondary small multiples
    smalls = daily.melt("Day", value_vars=["CPC","CTR","CVR"], var_name="Metric", value_name="Value")
    st.altair_chart(
        alt.Chart(smalls).mark_line().encode(x="Day:T", y="Value:Q", color="Metric:N").interactive(),
        use_container_width=True
    )

# Month-over-month
mom = (df.groupby("Month")
         .agg(Spend=("Spend","sum"),
              Clicks=("Clicks","sum"),
              Conversions=("Conversions","sum"),
              Impr=("Impr","sum"))
         .reset_index())
mom["CPC"] = np.where(mom["Clicks"]>0, mom["Spend"]/mom["Clicks"], np.nan)
mom["CPA"] = np.where(mom["Conversions"]>0, mom["Spend"]/mom["Conversions"], np.nan)
mom["CTR"] = np.where(mom["Impr"]>0, mom["Clicks"]/mom["Impr"], np.nan)
mom["CVR"] = np.where(mom["Clicks"]>0, mom["Conversions"]/mom["Clicks"], np.nan)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Month-over-Month Spend & Conversions**")
    m1 = alt.Chart(mom).transform_fold(
        ["Spend","Conversions"], as_=["Metric","Value"]
    ).mark_bar().encode(
        x="Month:N", y="Value:Q", color="Metric:N"
    ).interactive()
    st.altair_chart(m1, use_container_width=True)
with c2:
    st.markdown("**Month-over-Month Efficiency (CPC/CPA/CTR/CVR)**")
    m2 = mom.melt("Month", value_vars=["CPC","CPA","CTR","CVR"], var_name="Metric", value_name="Value")
    st.altair_chart(
        alt.Chart(m2).mark_line(point=True).encode(x="Month:N", y="Value:Q", color="Metric:N").interactive(),
        use_container_width=True
    )

st.divider()

# =========================
# BOOST vs PAUSE (tabs)
# =========================
st.subheader("Boost vs Pause — Focus Lists")

camp = (md.groupby(["Campaign","Profitability"], dropna=False)
          .agg(spend=("Spend","sum"),
               conv=("Conversions","sum"),
               clicks=("Clicks","sum"),
               impr=("Impr","sum"))
          .reset_index())

camp["CPC"] = np.where(camp["clicks"]>0, camp["spend"]/camp["clicks"], np.nan)
camp["CTR"] = np.where(camp["impr"]>0, camp["clicks"]/camp["impr"], np.nan)
camp["CVR"] = np.where(camp["clicks"]>0, camp["conv"]/camp["clicks"], np.nan)
camp["CPA"] = np.where(camp["conv"]>0, camp["spend"]/camp["conv"], np.nan)

boost = camp[camp["Profitability"]=="Good"].sort_values("spend", ascending=False).reset_index(drop=True)
pause = camp[camp["Profitability"].str.contains("Bad")].sort_values("spend", ascending=False).reset_index(drop=True)

tab1, tab2 = st.tabs(["🚀 BOOST (Good)", "⛔ PAUSE (Bad)"])

with tab1:
    st.markdown("**Top campaigns to BOOST (profitable under Target CPA)**")
    topb = boost.head(12)
    b_chart = alt.Chart(topb).mark_bar().encode(
        x=alt.X("spend:Q", title="Spend"),
        y=alt.Y("Campaign:N", sort="-x"),
        color=alt.value("#16a34a")
    )
    st.altair_chart(b_chart, use_container_width=True)
    st.dataframe(topb[["Campaign","spend","conv","CPA","CVR","CTR","CPC"]], use_container_width=True)
    st.download_button(
        "Download BOOST list (CSV)",
        data=boost.to_csv(index=False).encode("utf-8"),
        file_name=f"boost_{sel_month}.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown("**Top campaigns to PAUSE (unprofitable or no conversions)**")
    topp = pause.head(12)
    p_chart = alt.Chart(topp).mark_bar().encode(
        x=alt.X("spend:Q", title="Spend"),
        y=alt.Y("Campaign:N", sort="-x"),
        color=alt.value("#dc2626")
    )
    st.altair_chart(p_chart, use_container_width=True)
    st.dataframe(topp[["Campaign","spend","conv","CPA","CVR","CTR","CPC"]], use_container_width=True)
    st.download_button(
        "Download PAUSE list (CSV)",
        data=pause.to_csv(index=False).encode("utf-8"),
        file_name=f"pause_{sel_month}.csv",
        mime="text/csv"
    )

st.divider()

# =========================
# BOT
# =========================
st.subheader("🤖 Adviser Bot")
q = st.text_input("Ask me something (e.g., 'wasted spend', 'avg cpa', 'top campaigns')")
if q:
    base = bot_answer(df, sel_month, target_cpa, q)
    context = f"Month={sel_month}; Total={money(summ['total'])}; Good={money(summ['good'])}; Bad={money(summ['bad'])}; TargetCPA={money(target_cpa)}"
    llm = llm_answer(context, q)  # only replies if OPENAI_API_KEY is set
    st.success(llm or base)

# =========================
# EXPORTS
# =========================
st.download_button(
    "Download selected month rows (CSV)",
    data=df[df["Month"]==sel_month][["Day","Campaign","Spend","Clicks","Impr","Conversions","CPC","CPA","CTR","CVR","Profitability"]].to_csv(index=False).encode("utf-8"),
    file_name=f"adviser_rows_{sel_month}.csv",
    mime="text/csv"
)
 