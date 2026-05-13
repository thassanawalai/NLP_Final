import streamlit as st
import pandas as pd

from rag import init_chroma, add_to_collection, delete_from_collection
from cleaner import rule_based_clean
from llm import decode_with_gemini
from analytics import build_history_df, render_pie_chart, render_top5_bar_chart

CATEGORIES = [
    "การพนัน",
    "การค้าขาย",
    "คำหยาบคาย",
    "การเงิน/สินเชื่อ",
    "ยาเสพติด",
    "เพศ/อนาจาร",
    "อื่นๆ",
]


def main():
    st.set_page_config(
        page_title="Thai Text Normalizer AI",
        page_icon="🔍",
        layout="wide",
    )

    # ── Init ChromaDB (cached singleton) ─────────────────────────────────────
    collection = init_chroma()

    # ── Init Session State ────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔍 Thai Text Normalizer AI")
    st.markdown(
        "เครื่องมือ AI สำหรับ **ถอดรหัสคำวิบัติ** คำเลี่ยงเซ็นเซอร์ "
        "และอักขระที่ถูกดัดแปลง ให้กลับเป็นคำดั้งเดิม\n\n"
        "**Pipeline:** 🧹 Regex Cleaning → 🗄️ RAG (ChromaDB) → 🤖 Gemini AI"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="รับ API Key ได้จาก Google AI Studio",
        )
        if not api_key:
            st.warning("⚠️ กรุณาใส่ Gemini API Key เพื่อเริ่มใช้งาน")
        else:
            st.success("✅ API Key พร้อมใช้งาน")

        st.divider()
        st.markdown("**🔄 Processing Pipeline:**")
        st.markdown(
            """
            1. **Regex Cleaning** (Layer 1)
            2. **RAG Retrieval** → ChromaDB
            3. **Gemini AI** (Layer 2, `temperature=0.0`)
            """
        )

        st.divider()

        # ── Dictionary Manager ────────────────────────────────────────────────
        with st.expander("📚 จัดการพจนานุกรมคำศัพท์", expanded=False):
            st.caption(f"มีคำในฐานข้อมูล: **{collection.count()} คำ**")

            new_slang_word = st.text_input(
                "คำวิบัติ (Slang)",
                placeholder="เช่น บ@ค@ร่@",
                key="dict_slang",
            )
            new_meaning = st.text_input(
                "คำแปลที่ถูกต้อง (Meaning)",
                placeholder="เช่น บาคาร่า",
                key="dict_meaning",
            )
            new_category = st.selectbox(
                "หมวดหมู่ (Category)",
                CATEGORIES,
                key="dict_category",
            )

            if st.button("➕ เพิ่มลงฐานข้อมูล", use_container_width=True):
                if new_slang_word.strip() and new_meaning.strip():
                    add_to_collection(
                        collection,
                        slang=new_slang_word.strip(),
                        meaning=new_meaning.strip(),
                        category=new_category,
                    )
                    # เคลียร์ cache เพื่อให้ count อัปเดตใน sidebar
                    st.success(
                        f"✅ เพิ่ม \"{new_slang_word}\" → \"{new_meaning}\" "
                        f"[{new_category}] สำเร็จ!"
                    )
                    st.rerun()
                else:
                    st.warning("กรุณากรอกคำวิบัติและคำแปลให้ครบ")

            st.divider()
            delete_slang_word = st.text_input(
                "ลบคำวิบัติ (ระบุคำที่ต้องการลบ)",
                placeholder="เช่น บ@ค@ร่@",
                key="dict_delete",
            )
            if st.button("🗑️ ลบจากฐานข้อมูล", use_container_width=True):
                if delete_slang_word.strip():
                    success = delete_from_collection(collection, delete_slang_word.strip())
                    if success:
                        st.success(f"✅ ลบ \"{delete_slang_word}\" สำเร็จ!")
                        st.rerun()
                    else:
                        st.error(f"❌ ไม่พบคำว่า \"{delete_slang_word}\" ในฐานข้อมูล")

            # แสดง Dictionary ทั้งหมดในฐานข้อมูล
            if st.button("📖 ดูพจนานุกรมทั้งหมด", use_container_width=True):
                st.session_state["show_dict"] = True

        if st.session_state.get("show_dict"):
            with st.sidebar:
                all_docs = collection.get()
                if all_docs["documents"]:
                    rows = [
                        {"คำเลี่ยง": doc, **meta}
                        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
                    ]
                    df_dict = pd.DataFrame(rows)
                    df_dict.columns = ["คำเลี่ยง", "คำแปล", "หมวดหมู่"]
                    st.dataframe(df_dict, use_container_width=True, hide_index=True)
                if st.button("ซ่อน", key="hide_dict"):
                    st.session_state["show_dict"] = False
                    st.rerun()

        st.divider()

        # ── Analytics quick stat + clear ─────────────────────────────────────
        st.metric("คำเลี่ยงสะสม (session)", len(st.session_state["history"]))
        if st.button("🗑️ ล้างข้อมูล Analytics", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    # ── Main Tabs ─────────────────────────────────────────────────────────────
    tab_decode, tab_analytics = st.tabs(
        ["🔍 ถอดรหัสข้อความ", "📊 สถิติเชิงลึก (Analytics)"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 — ถอดรหัสข้อความ
    # ══════════════════════════════════════════════════════════════════════════
    with tab_decode:
        input_text = st.text_area(
            "📝 ป้อนข้อความที่ต้องการถอดรหัส",
            height=180,
            placeholder=(
                "ตัวอย่าง: สมัคร เล่น ก.า.ร.พ.น.ั.น ออนไลน์ รับ โ-บ-นั-ส ทุกวัน "
                "หรือ วันนี้ V า ย ของดี ราคาถูก ติดต่อ inbox"
            ),
            key="main_input_text",  # เพิ่ม key เพื่อให้ state ไม่หายตอน rerun
        )

        decode_btn = st.button(
            "🔓 ถอดรหัสข้อความ",
            disabled=not bool(api_key),
            use_container_width=True,
            type="primary",
        )

        if not api_key:
            st.info("💡 ใส่ Gemini API Key ในแถบซ้ายมือเพื่อเปิดใช้งานปุ่มถอดรหัส")

        if decode_btn:
            if not input_text.strip():
                st.warning("⚠️ กรุณาป้อนข้อความที่ต้องการประมวลผล")
                st.stop()

            with st.spinner("กำลังประมวลผล...  Regex → RAG (ChromaDB) → Gemini AI"):
                try:
                    # Layer 1: Regex Cleaning
                    cleaned = rule_based_clean(input_text)

                    # Layer 2 + RAG: ดึง context จาก ChromaDB แล้วส่งให้ Gemini
                    result, rag_context = decode_with_gemini(api_key, cleaned, collection)

                except ValueError as ve:
                    st.error(f"❌ ไม่สามารถ Parse ผลลัพธ์จาก Gemini ได้\n\n{ve}")
                    st.stop()
                except Exception as e:
                    err = str(e)
                    if "API_KEY" in err.upper() or "api key" in err.lower() or "invalid" in err.lower():
                        st.error("❌ API Key ไม่ถูกต้องหรือหมดอายุ")
                    elif "quota" in err.lower() or "429" in err:
                        st.error("❌ เกิน Quota การใช้งาน Gemini API")
                    elif "safety" in err.lower():
                        st.error("❌ Gemini Safety Filter บล็อกคำขอนี้")
                    else:
                        st.error(f"❌ เกิดข้อผิดพลาด: {err}")
                    st.stop()
            
            # ตรวจสอบและจัดการกรณีที่ Gemini คืนค่ามาเป็น list แทนที่จะเป็น dict
            if isinstance(result, list):
                result = result[0] if len(result) > 0 else {}
                
            if not isinstance(result, dict):
                result = {}

            # บันทึก detected_slang เข้า session history สำหรับ Analytics
            new_slang: list = result.get("detected_slang", [])
            if new_slang:
                st.session_state["history"].extend(new_slang)

            # ── Results ───────────────────────────────────────────────────────
            st.success("✅ ถอดรหัสสำเร็จ!")
            st.divider()

            col_orig, col_norm = st.columns(2, gap="large")
            with col_orig:
                st.subheader("📄 ข้อความต้นฉบับ (Original)")
                st.info(result.get("original", input_text))
            with col_norm:
                st.subheader("✨ ข้อความที่ Normalize แล้ว")
                st.success(result.get("normalized", "—"))

                if result.get("normalized"):
                    # ปุ่มคัดลอก/ดาวน์โหลดข้อความที่ทำความสะอาดแล้ว
                    st.download_button("💾 ดาวน์โหลดข้อความ (TXT)", result.get("normalized"), file_name="normalized_text.txt")

            # Debug expanders
            with st.expander("🗄️ ดู RAG Context ที่ดึงจาก ChromaDB"):
                st.caption(
                    "ข้อมูลเหล่านี้คือ Top-3 document จาก Vector DB "
                    "ที่ถูก inject เข้า Gemini Prompt อัตโนมัติ"
                )
                st.code(rag_context, language=None)

            with st.expander("🧹 ดูผลลัพธ์หลัง Layer 1 — Regex Cleaning"):
                st.caption("ข้อความที่ส่งให้ Gemini หลังกรองอักขระซ่อนเร้นออกแล้ว")
                st.code(cleaned, language=None)

            st.divider()

            st.subheader("📊 คำเลี่ยงที่ตรวจพบในข้อความนี้")
            if new_slang:
                df_this = pd.DataFrame(new_slang)
                df_this.index = range(1, len(df_this) + 1)
                df_this.columns = ["คำเลี่ยง / คำวิบัติ", "คำแปล", "หมวดหมู่"]
                st.dataframe(
                    df_this,
                    use_container_width=True,
                    column_config={
                        "คำเลี่ยง / คำวิบัติ": st.column_config.TextColumn(width="medium"),
                        "คำแปล": st.column_config.TextColumn(width="medium"),
                        "หมวดหมู่": st.column_config.TextColumn(width="medium"),
                    },
                )
                st.info(
                    f"💡 พบคำเลี่ยงใหม่ {len(new_slang)} คำ — "
                    "ไปที่แท็บ **📊 สถิติเชิงลึก** เพื่อดู Dashboard"
                )
            else:
                st.info("ℹ️ ไม่พบคำเลี่ยงหรือคำวิบัติในข้อความนี้")

            with st.expander("🗂️ ดู Raw JSON จาก Gemini"):
                st.json(result)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 — Analytics Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.subheader("📊 Real-Time Analytics Dashboard")
        st.caption("ข้อมูลสะสมจากทุกข้อความที่ประมวลผลใน session นี้")

        history: list = st.session_state["history"]

        if not history:
            st.info(
                "ℹ️ ยังไม่มีข้อมูลสถิติ\n\n"
                "กรุณาไปที่แท็บ **🔍 ถอดรหัสข้อความ** แล้วประมวลผลข้อความอย่างน้อย 1 ครั้ง"
            )
        else:
            df_all = build_history_df()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🔎 จำนวนคำเลี่ยงทั้งหมดที่ตรวจพบ", len(df_all))
            with m2:
                st.metric("🗂️ หมวดหมู่ที่พบ", df_all["category"].nunique())
            with m3:
                top_cat = df_all["category"].value_counts().idxmax()
                st.metric("🏆 หมวดหมู่ยอดนิยม", top_cat)
            
            st.write("")
            csv_data = df_all.to_csv(index=False).encode('utf-8-sig') # ใช้ utf-8-sig เพื่อให้ Excel อ่านภาษาไทยได้
            st.download_button("📥 ดาวน์โหลดข้อมูลสถิติ (CSV)", data=csv_data, file_name="slang_analytics.csv", mime="text/csv", use_container_width=True)

            st.divider()

            col_pie, col_bar = st.columns(2, gap="large")
            with col_pie:
                render_pie_chart(df_all)
            with col_bar:
                render_top5_bar_chart(df_all)

            st.divider()

            with st.expander("📋 ดูข้อมูลดิบทั้งหมด (All Detected Slang)"):
                df_display = df_all.copy()
                df_display.index = range(1, len(df_display) + 1)
                df_display.columns = ["คำเลี่ยง / คำวิบัติ", "คำแปล", "หมวดหมู่"]
                st.dataframe(df_display, use_container_width=True)


if __name__ == "__main__":
    main()
