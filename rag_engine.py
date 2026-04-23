"""
rag_engine.py
RAGシステム：FAISSインデックスの構築・検索モジュール
"""

import os
import re
import pickle

import faiss
import numpy as np
import tiktoken
from openai import OpenAI

# インデックスファイルのパス
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE  = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.pkl")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "bakery_expert_knowledge.md")

# Embeddingモデル
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536  # text-embedding-3-small の次元数


# ────────────────────────────────────────────────
# チャンク分割
# ────────────────────────────────────────────────

def _split_chunks(text: str) -> list[str]:
    """## 見出し単位でテキストをチャンク分割する"""
    # ## で始まるブロックごとに分割
    parts = re.split(r'(?=^##+ )', text, flags=re.MULTILINE)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) > 30:  # 短すぎるものは除外
            chunks.append(part)
    return chunks


# ────────────────────────────────────────────────
# Embedding取得
# ────────────────────────────────────────────────

def _get_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """テキストリストをEmbeddingベクトルに変換して返す"""
    client = OpenAI(api_key=api_key)
    # OpenAI APIはバッチで複数テキストを処理できる
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def _get_single_embedding(text: str, api_key: str) -> np.ndarray:
    """1テキストのEmbeddingを返す"""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return np.array([response.data[0].embedding], dtype=np.float32)


# ────────────────────────────────────────────────
# 公開関数
# ────────────────────────────────────────────────

def get_chunk_count() -> int:
    """インデックス済みチャンク数を返す（未構築時は0）"""
    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            return len(chunks)
        except Exception:
            return 0
    return 0


def build_index(api_key: str) -> int:
    """
    ナレッジベースをチャンク分割→Embedding→FAISSインデックス構築。
    既存ファイルが存在する場合もロードせず再構築する（明示的な再構築用）。
    構築済みチャンク数を返す。
    """
    # ナレッジベース読み込み
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        content = f.read()

    # チャンク分割
    chunks = _split_chunks(content)

    # Embeddingを取得
    vectors = _get_embeddings(chunks, api_key)

    # FAISSインデックス構築（内積→コサイン類似度）
    faiss.normalize_L2(vectors)  # L2正規化でコサイン類似度を実現
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    # 保存
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def _load_index():
    """保存済みインデックスとチャンクを読み込む"""
    index  = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """tiktokenでトークン数をカウントする"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def retrieve(query: str, api_key: str, top_k: int = 3) -> str:
    """
    クエリに近いチャンクをFAISSで検索し、
    最大1000トークンに収めて返す。
    インデックス未構築の場合は自動でbuild_indexを実行する。
    """
    # インデックスが存在しない場合は自動構築
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        build_index(api_key)

    index, chunks = _load_index()

    # クエリをEmbeddingに変換
    query_vec = _get_single_embedding(query, api_key)
    faiss.normalize_L2(query_vec)

    # 類似チャンクを検索
    distances, indices = index.search(query_vec, top_k)

    # トークン上限1000以内で結果を連結
    result_parts = []
    total_tokens = 0
    token_limit  = 1000

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        tokens = _count_tokens(chunk)
        if total_tokens + tokens > token_limit:
            # 残りトークン分だけ切り詰めて追加
            remaining = token_limit - total_tokens
            if remaining > 50:
                enc = tiktoken.get_encoding("cl100k_base")
                truncated = enc.decode(enc.encode(chunk)[:remaining])
                result_parts.append(truncated)
            break
        result_parts.append(chunk)
        total_tokens += tokens

    return "\n\n---\n\n".join(result_parts)
