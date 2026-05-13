import re

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Rule-based Cleaning ด้วย Regex
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_clean(text: str) -> str:
    """
    ทำความสะอาดข้อความเบื้องต้นก่อนส่งให้ AI ประมวลผล
    กำจัดอักขระที่มองไม่เห็นและสัญลักษณ์แทรกกลางคำที่ใช้เลี่ยงฟิลเตอร์
    คงเครื่องหมายวรรคตอนและ Emoji ที่มีความหมายไว้
    """
    # ลบ Zero-Width Characters: ZWS, ZWNJ, ZWJ, SHY, BOM
    text = re.sub(r'[​‌‍­﻿]', '', text)
    # ลบสัญลักษณ์แทรกกลางคำ เช่น ห-ี → หี, ก.า.ร.พ.น.ั.น → การพนัน
    text = re.sub(r'(?<=[ก-๙a-zA-Z0-9])[*\-_./|\\]+(?=[ก-๙a-zA-Z0-9])', '', text)
    # บีบช่องว่างระหว่างอักษรไทยให้เหลือ 1 ช่อง
    text = re.sub(r'(?<=[ก-๙])\s{2,}(?=[ก-๙])', ' ', text)
    # ลบ trailing/leading whitespace แต่ละบรรทัด และบีบ whitespace ซ้ำซ้อน
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()