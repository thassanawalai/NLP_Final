import streamlit as st
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Analytics Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_history_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state["history"])

def render_pie_chart(df: pd.DataFrame) -> None:
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["หมวดหมู่", "จำนวน"]

    fig = px.pie(
        cat_counts,
        names="หมวดหมู่",
        values="จำนวน",
        title="สัดส่วนคำเลี่ยงตามหมวดหมู่",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

def render_top5_bar_chart(df: pd.DataFrame) -> None:
    word_counts = df["word"].value_counts().head(5).reset_index()
    word_counts.columns = ["คำเลี่ยง", "จำนวนครั้ง"]

    fig = px.bar(
        word_counts,
        x="คำเลี่ยง",
        y="จำนวนครั้ง",
        title="Top 5 คำเลี่ยงที่ตรวจพบบ่อยที่สุด",
        text="จำนวนครั้ง",
        color="จำนวนครั้ง",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="คำเลี่ยง / คำวิบัติ", yaxis_title="จำนวนครั้ง", coloraxis_showscale=False, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)