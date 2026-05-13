import streamlit as st
import chromadb
import uuid

# ─────────────────────────────────────────────────────────────────────────────
# RAG Layer: ChromaDB — Vector Database สำหรับ Lexicon
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def init_chroma() -> chromadb.Collection:
    """
    สร้าง ChromaDB PersistentClient และ collection 'slang_dict' ครั้งเดียว
    @st.cache_resource ทำให้ object นี้ถูกแชร์ข้ามทุก user session
    และไม่ถูก re-initialize ทุกครั้งที่ Streamlit rerun script
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="slang_dict",
        metadata={"hnsw:space": "cosine"},  # cosine similarity เหมาะกับ text embedding
    )
    _seed_collection(collection)
    return collection

def _seed_collection(collection: chromadb.Collection) -> None:
    """
    เพิ่มข้อมูลเริ่มต้นเข้า collection เฉพาะเมื่อ DB ว่างเปล่าเท่านั้น
    ป้องกัน duplicate เมื่อ app ถูก restart หลายครั้ง
    """
    if collection.count() > 0:
        return  # มีข้อมูลแล้ว ไม่ต้อง seed ซ้ำ

    seed_docs = ["Vาย, ข า ย", "คสย, ค ว ย", "บ@ค@ร่@", "กี, Kee", "พ.นัน, ป ล่ อ ย กู้"]
    seed_metas = [
        {"meaning": "ขาย",               "category": "การค้าขาย"},
        {"meaning": "ควย",               "category": "คำหยาบคาย"},
        {"meaning": "บาคาร่า",           "category": "การพนัน"},
        {"meaning": "หี",                "category": "คำหยาบคาย"},
        {"meaning": "พนัน / ปล่อยกู้",   "category": "การพนัน/การเงิน"},
    ]
    collection.add(
        documents=seed_docs,
        metadatas=seed_metas,
        ids=[str(uuid.uuid4()) for _ in seed_docs],
    )

def retrieve_context(collection: chromadb.Collection, query: str, n_results: int = 3) -> str:
    count = collection.count()
    if count == 0:
        return "- (ไม่มีข้อมูลใน Vector Database)"
    results = collection.query(query_texts=[query], n_results=min(n_results, count))
    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lines.append(f'- "{doc}" หมายถึง "{meta["meaning"]}" (หมวด: {meta["category"]})')
    return "\n".join(lines)

def add_to_collection(collection: chromadb.Collection, slang: str, meaning: str, category: str) -> None:
    """เพิ่มคำเลี่ยงใหม่เข้า ChromaDB collection พร้อม metadata"""
    collection.add(
        documents=[slang],
        metadatas=[{"meaning": meaning, "category": category}],
        ids=[str(uuid.uuid4())],
    )

def delete_from_collection(collection: chromadb.Collection, slang: str) -> bool:
    """ค้นหาและลบคำเลี่ยงออกจาก ChromaDB คืนค่า True หากลบสำเร็จ"""
    all_docs = collection.get()
    ids_to_delete = [
        doc_id for doc_id, doc in zip(all_docs["ids"], all_docs["documents"]) 
        if doc == slang
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        return True
    return False