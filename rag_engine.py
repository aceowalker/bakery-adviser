"""
rag_engine.py
RAGシステム：FAISSインデックスの構築・ハイブリッド検索モジュール
クエリ書き換え（Query Rewriting）とハイブリッド検索（FAISS + BM25 + RRF）を実装
"""

import os
import re
import pickle

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE     = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_FILE    = os.path.join(BASE_DIR, "chunks.pkl")
BM25_FILE      = os.path.join(BASE_DIR, "bm25_corpus.pkl")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "bakery_expert_knowledge.md")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
CHAT_MODEL  = "gpt-4o-mini"


# ────────────────────────────────────────────────
# チャンク分割
# ────────────────────────────────────────────────

def _split_chunks(text: str) -> list[str]:
    """## 見出し単位でテキストをチャンク分割する"""
    parts = re.split(r'(?=^##+ )', text, flags=re.MULTILINE)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) > 30:
            chunks.append(part)
    return chunks


# ────────────────────────────────────────────────
# BM25用トークン化
# ────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    """BM25用にテキストをトークン化する（ASCII単語 + 日本語文字bigram）"""
    tokens = []
    # ASCII英数字を単語単位で取得
    tokens.extend(w.lower() for w in re.findall(r'[A-Za-z0-9]+', text))
    # 日本語文字（ひらがな・カタカナ・漢字）をunigram + bigramで取得
    jp_text = re.sub(r'[^぀-ゟ゠-ヿ一-鿿]', '', text)
    if jp_text:
        tokens.extend(list(jp_text))
        for i in range(len(jp_text) - 1):
            tokens.append(jp_text[i:i + 2])
    return tokens if tokens else text.split()


# ────────────────────────────────────────────────
# Embedding取得
# ────────────────────────────────────────────────

def _get_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """テキストリストをEmbeddingベクトルに変換して返す"""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def _get_single_embedding(text: str, api_key: str) -> np.ndarray:
    """1テキストのEmbeddingを返す"""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype=np.float32)


# ────────────────────────────────────────────────
# クエリ書き換え
# ────────────────────────────────────────────────

def _rewrite_query(query: str, api_key: str) -> list[str]:
    """ユーザー入力を3方向のクエリに書き換える。失敗時は元のクエリのみ返す。"""
    client = OpenAI(api_key=api_key)
    prompt = (
        "あなたはベーカリー経営のナレッジベースを検索するクエリ生成AIです。\n"
        "ユーザーの質問に対して、適切な情報を確実に取得するための\n"
        "検索クエリを3つ生成してください。\n\n"
        "【生成するクエリの方向】\n"
        "1. 元の質問の言い換え\n"
        "2. 経営・販売戦略の専門用語への変換（例：「売れ筋」→「ABC分析Aランク商品」）\n"
        "3. ナレッジベースの章立てを意識した具体的なキーワード\n\n"
        "【出力形式】\n"
        "クエリ1:（テキスト）\n"
        "クエリ2:（テキスト）\n"
        "クエリ3:（テキスト）\n\n"
        "それ以外の文章は一切出力しないこと。\n\n"
        f"ユーザーの質問: {query}"
    )
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        queries = [query]
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('クエリ') and ':' in line:
                q = line.split(':', 1)[1].strip()
                if q and q not in queries:
                    queries.append(q)
        return queries if len(queries) > 1 else [query]
    except Exception:
        return [query]


# ────────────────────────────────────────────────
# ハイブリッド検索・RRF
# ────────────────────────────────────────────────

def _search_bm25(
    tokenized_query: list[str],
    tokenized_corpus: list[list[str]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """BM25で検索し、(チャンクインデックス, スコア)のリストを返す"""
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def _rrf_merge(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[int]:
    """RRF（Reciprocal Rank Fusion）で複数ランキングを統合し、降順インデックスリストを返す"""
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)


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
    BM25用トークン化コーパスも同時に保存する。
    構築済みチャンク数を返す。
    """
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        content = f.read()

    chunks = _split_chunks(content)

    # FAISSインデックス構築（内積→コサイン類似度）
    vectors = _get_embeddings(chunks, api_key)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)

    # チャンク保存
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    # BM25用トークン化コーパス保存
    tokenized_corpus = [_tokenize_for_bm25(c) for c in chunks]
    with open(BM25_FILE, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    return len(chunks)


def _load_index():
    """保存済みFAISSインデックス・チャンク・BM25コーパスを読み込む"""
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    if os.path.exists(BM25_FILE):
        with open(BM25_FILE, "rb") as f:
            tokenized_corpus = pickle.load(f)
    else:
        # 後方互換：旧インデックスにBM25データがない場合は再生成
        tokenized_corpus = [_tokenize_for_bm25(c) for c in chunks]
        with open(BM25_FILE, "wb") as f:
            pickle.dump(tokenized_corpus, f)

    return index, chunks, tokenized_corpus


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """tiktokenでトークン数をカウントする"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def retrieve(query: str, api_key: str, top_k: int = 5) -> str:
    """
    クエリ書き換え（3方向）→ハイブリッド検索（FAISS + BM25）→RRF統合で
    上位チャンクを取得し、最大1500トークンに収めて返す。
    インデックス未構築の場合は自動でbuild_indexを実行する。
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        build_index(api_key)

    index, chunks, tokenized_corpus = _load_index()

    # クエリ書き換えで最大3クエリ生成（元クエリ含む）
    queries = _rewrite_query(query, api_key)

    # 各クエリのFAISS + BM25結果を収集
    all_ranked_lists: list[list[tuple[int, float]]] = []
    for q in queries:
        # FAISS検索（top_k=5）
        q_vec = _get_single_embedding(q, api_key)
        faiss.normalize_L2(q_vec)
        distances, indices = index.search(q_vec, top_k)
        faiss_results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]
        all_ranked_lists.append(faiss_results)

        # BM25検索（top_k=5）
        bm25_results = _search_bm25(_tokenize_for_bm25(q), tokenized_corpus, top_k)
        all_ranked_lists.append(bm25_results)

    # RRF（k=60）で統合し上位top_k件を取得
    merged_indices = _rrf_merge(all_ranked_lists)[:top_k]

    # トークン上限1500以内で結果を連結
    result_parts = []
    seen: set[int] = set()
    total_tokens = 0
    token_limit = 1500

    for idx in merged_indices:
        if idx < 0 or idx >= len(chunks) or idx in seen:
            continue
        seen.add(idx)
        chunk = chunks[idx]
        tokens = _count_tokens(chunk)
        if total_tokens + tokens > token_limit:
            remaining = token_limit - total_tokens
            if remaining > 50:
                enc = tiktoken.get_encoding("cl100k_base")
                truncated = enc.decode(enc.encode(chunk)[:remaining])
                result_parts.append(truncated)
            break
        result_parts.append(chunk)
        total_tokens += tokens

    return "\n\n---\n\n".join(result_parts)
