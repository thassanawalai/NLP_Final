import google.generativeai as genai
import json
import re
import streamlit as st
import chromadb
from rag import retrieve_context

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Contextual Decoding + RAG (Gemini API)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "คุณคือผู้เชี่ยวชาญด้านภาษาศาสตร์ NLP "
    "หน้าที่ของคุณคือแปลงข้อความภาษาไทยที่มีคำวิบัติ คำเลี่ยงเซ็นเซอร์ "
    "หรือคำหยาบที่ถูกดัดแปลงอักขระ ให้กลับเป็นคำศัพท์ดั้งเดิม (Root word) "
    "ที่ถูกต้องตามพจนานุกรม โดยคงบริบทเดิมไว้ ห้ามเปลี่ยนความหมายประโยค "
    "นอกจากนี้ให้ทำการจัดหมวดหมู่ของคำเลี่ยงที่พบด้วย "
    "โดยแบ่งหมวดหมู่ตามความเหมาะสม เช่น การพนัน, การค้าขาย, คำหยาบคาย ฯลฯ"
)

@st.cache_data(show_spinner=False)
def _cached_gemini_call(api_key: str, prompt: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def decode_with_gemini(
    api_key: str,
    cleaned_text: str,
    collection: chromadb.Collection,
) -> tuple[dict, str]:
    lexicon_context = retrieve_context(collection, cleaned_text, n_results=3)

    prompt = f"""คุณคือผู้เชี่ยวชาญด้านภาษาศาสตร์ NLP หน้าที่ของคุณคือถอดรหัสข้อความภาษาไทยที่มีคำวิบัติ คำเลี่ยงฟิลเตอร์เซ็นเซอร์ ให้กลับเป็นคำศัพท์ดั้งเดิม (Root word) ที่ถูกต้องตามพจนานุกรม โดยต้องคงบริบทเดิมไว้ ห้ามเปลี่ยนความหมายของประโยค

ข้อควรจำ (Lexicon Reference) — ดึงมาจาก Vector Database อัตโนมัติ:
{lexicon_context}

ตัวอย่างการทำงาน (Few-Shot Examples):
Input: "วันนี้ไปเดินตลาดเจอร้าน V า ย ของเยอะมาก แต่แอบได้ยินคนด่ากันว่า คสย เอ้ย"
Output: {{
  "original": "วันนี้ไปเดินตลาดเจอร้าน V า ย ของเยอะมาก แต่แอบได้ยินคนด่ากันว่า คสย เอ้ย",
  "normalized": "วันนี้ไปเดินตลาดเจอร้านขายของเยอะมาก แต่แอบได้ยินคนด่ากันว่า ควย เอ้ย",
  "detected_slang": [
    {{"word": "V า ย", "meaning": "ขาย", "category": "การค้าขาย"}},
    {{"word": "คสย", "meaning": "ควย", "category": "คำหยาบคาย"}}
  ]
}}

จงประมวลผลข้อความต่อไปนี้ และตอบกลับเป็น JSON เท่านั้น โดยระบุ 'meaning' ของแต่ละคำ และจัดหมวดหมู่คำเลี่ยงตามความเหมาะสม:
Input: "{cleaned_text}"
"""
    raw = _cached_gemini_call(api_key, prompt)

    try:
        return json.loads(raw), lexicon_context
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            return json.loads(match.group(1)), lexicon_context
        raise ValueError(f"ไม่สามารถ parse JSON ได้จากผลลัพธ์:\n{raw}")