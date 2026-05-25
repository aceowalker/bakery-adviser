"""
rag_engine.py
RAGシステム：階層FAISS（章単位 + 詳細チャンク）＋ハイブリッド検索＋クエリ書き換え＋章選択エージェント
"""

import json
import os
import pickle
import re

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi

from chapter_indexer import build_chapter_index, load_chapter_index

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "bakery_expert_knowledge.md")

# ── レガシー（後方互換）
INDEX_FILE  = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.pkl")
BM25_FILE   = os.path.join(BASE_DIR, "bm25_corpus.pkl")

# ── 階層インデックス
CHAPTER_FAISS_FILE  = os.path.join(BASE_DIR, "chapter_faiss.bin")
CHAPTER_META_FILE   = os.path.join(BASE_DIR, "chapter_meta.pkl")
DETAIL_FAISS_FILE   = os.path.join(BASE_DIR, "detail_faiss.bin")
DETAIL_CHUNKS_FILE  = os.path.join(BASE_DIR, "detail_chunks.pkl")
DETAIL_BM25_FILE    = os.path.join(BASE_DIR, "detail_bm25.pkl")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
CHAT_MODEL  = "gpt-4o-mini"


# ────────────────────────────────────────────────
# テキスト分割
# ────────────────────────────────────────────────

def _split_by_chapter(text: str) -> list[dict]:
    """## 見出しで章単位に分割し、メタデータ付きdictを返す"""
    parts = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    chapters = []
    for part in parts:
        part = part.strip()
        m = re.match(r'^## (\d+)\.\s+(.+?)$', part, re.MULTILINE)
        if not m:
            continue
        chapters.append({
            "chapter_number": int(m.group(1)),
            "chapter_title":  m.group(2).strip(),
            "text":           part,
        })
    return chapters


def _split_detail_chunks(chapter: dict, max_tokens: int = 500) -> list[dict]:
    """
    章テキストを ### 見出し単位でさらに分割し、
    500トークンを超える場合はさらに分割する。
    各チャンクに chapter_number / chapter_title / section_title を付与。
    """
    enc = tiktoken.get_encoding("cl100k_base")
    text = chapter["text"]
    ch_num   = chapter["chapter_number"]
    ch_title = chapter["chapter_title"]

    # ### 見出しで分割
    sections = re.split(r'(?=^### )', text, flags=re.MULTILINE)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 20:
            continue
        m = re.match(r'^### (.+?)$', sec, re.MULTILINE)
        sec_title = m.group(1).strip() if m else ch_title

        tokens = enc.encode(sec)
        if len(tokens) <= max_tokens:
            chunks.append({
                "chapter_number": ch_num,
                "chapter_title":  ch_title,
                "section_title":  sec_title,
                "text":           sec,
            })
        else:
            # 500トークン単位でスライス
            for i in range(0, len(tokens), max_tokens):
                piece = enc.decode(tokens[i: i + max_tokens])
                chunks.append({
                    "chapter_number": ch_num,
                    "chapter_title":  ch_title,
                    "section_title":  sec_title,
                    "text":           piece,
                })
    # ### 見出しがない場合は章全体を1チャンクとして追加
    if not chunks:
        chunks.append({
            "chapter_number": ch_num,
            "chapter_title":  ch_title,
            "section_title":  ch_title,
            "text":           text,
        })
    return chunks


# ────────────────────────────────────────────────
# BM25 トークン化
# ────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    tokens = []
    tokens.extend(w.lower() for w in re.findall(r'[A-Za-z0-9]+', text))
    jp_text = re.sub(r'[^぀-ゟ゠-ヿ一-鿿]', '', text)
    if jp_text:
        tokens.extend(list(jp_text))
        for i in range(len(jp_text) - 1):
            tokens.append(jp_text[i:i + 2])
    return tokens if tokens else text.split()


# ────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────

def _get_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _get_single_embedding(text: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype=np.float32)


# ────────────────────────────────────────────────
# クエリ書き換え
# ────────────────────────────────────────────────

def _rewrite_query(query: str, api_key: str) -> list[str]:
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
# 章選択エージェント（STEP 3）
# ────────────────────────────────────────────────

def select_relevant_chapters(query: str, api_key: str, chapter_index: list[dict]) -> list[int]:
    """
    chapter_index.json の概要をシステムプロンプトに渡し、
    クエリに関連する章番号を最大3つ選択して返す。
    パース失敗時は全章番号をフォールバックとして返す。
    """
    all_numbers = [ch["chapter_number"] for ch in chapter_index]

    index_text = "\n".join(
        f"{ch['chapter_number']}章: {ch['title']} — {ch['summary']} "
        f"[キーワード: {', '.join(ch['keywords'])}]"
        for ch in chapter_index
    )

    prompt = (
        "あなたはベーカリー経営の専門AIです。\n"
        "ユーザーの質問に対して、以下のナレッジベースの章一覧から\n"
        "回答に必要な章を最大3つ選んでください。\n\n"
        "【章一覧】\n"
        f"{index_text}\n\n"
        "【ユーザーの質問】\n"
        f"{query}\n\n"
        "【出力形式】\n"
        "関連章: [章番号1, 章番号2, 章番号3]\n"
        "理由: （1行で）\n\n"
        "それ以外の文章は出力しないこと。\n"
        "章番号は整数のみで出力すること。"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()

        for line in text.split('\n'):
            if '関連章' in line and ':' in line:
                raw = line.split(':', 1)[1].strip()
                nums = re.findall(r'\d+', raw)
                selected = [int(n) for n in nums if int(n) in all_numbers]
                if selected:
                    return selected[:3]
    except Exception:
        pass

    return all_numbers  # フォールバック：全章


# ────────────────────────────────────────────────
# BM25 / FAISS 検索ユーティリティ
# ────────────────────────────────────────────────

def _search_bm25(
    tokenized_query: list[str],
    tokenized_corpus: list[list[str]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def _rrf_merge(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[int]:
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)


# ────────────────────────────────────────────────
# インデックス構築（公開）
# ────────────────────────────────────────────────

def get_chunk_count() -> int:
    """詳細チャンク数を返す（未構築時は0）"""
    if os.path.exists(DETAIL_CHUNKS_FILE):
        try:
            with open(DETAIL_CHUNKS_FILE, "rb") as f:
                return len(pickle.load(f))
        except Exception:
            pass
    # レガシー互換
    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE, "rb") as f:
                return len(pickle.load(f))
        except Exception:
            pass
    return 0


def build_index(api_key: str) -> int:
    """
    階層FAISSインデックスを構築する。
    ・chapter_faiss  : 章単位（レベル1）
    ・detail_faiss   : 詳細チャンク（レベル2）
    ・detail_bm25    : 詳細チャンクのBM25コーパス
    構築済み詳細チャンク数を返す。
    """
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        content = f.read()

    # ── レベル1：章単位
    chapter_dicts = _split_by_chapter(content)
    chapter_texts = [c["text"] for c in chapter_dicts]
    chapter_metas = [
        {"chapter_number": c["chapter_number"], "chapter_title": c["chapter_title"]}
        for c in chapter_dicts
    ]

    ch_vectors = _get_embeddings(chapter_texts, api_key)
    faiss.normalize_L2(ch_vectors)
    ch_index = faiss.IndexFlatIP(EMBED_DIM)
    ch_index.add(ch_vectors)
    faiss.write_index(ch_index, CHAPTER_FAISS_FILE)
    with open(CHAPTER_META_FILE, "wb") as f:
        pickle.dump(chapter_metas, f)

    # ── レベル2：詳細チャンク
    detail_chunks: list[dict] = []
    for ch in chapter_dicts:
        detail_chunks.extend(_split_detail_chunks(ch))

    detail_texts = [c["text"] for c in detail_chunks]
    dt_vectors = _get_embeddings(detail_texts, api_key)
    faiss.normalize_L2(dt_vectors)
    dt_index = faiss.IndexFlatIP(EMBED_DIM)
    dt_index.add(dt_vectors)
    faiss.write_index(dt_index, DETAIL_FAISS_FILE)
    with open(DETAIL_CHUNKS_FILE, "wb") as f:
        pickle.dump(detail_chunks, f)

    # BM25コーパス
    tokenized = [_tokenize_for_bm25(c["text"]) for c in detail_chunks]
    with open(DETAIL_BM25_FILE, "wb") as f:
        pickle.dump(tokenized, f)

    # レガシーファイルも維持（後方互換）
    legacy_texts = [c["text"] for c in detail_chunks]
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(legacy_texts, f)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(tokenized, f)
    faiss.write_index(dt_index, INDEX_FILE)

    # chapter_index.json の再生成
    from chapter_indexer import INDEX_FILE as CI_FILE
    if os.path.exists(CI_FILE):
        os.remove(CI_FILE)
    build_chapter_index(api_key)

    # メモリキャッシュをクリア
    invalidate_cache()

    return len(detail_chunks)


_hierarchy_cache: dict = {}


def _load_hierarchy():
    """
    階層インデックスをメモリキャッシュ付きで読み込む。
    同一プロセス内で2回目以降はディスクI/Oを省略する。
    """
    if _hierarchy_cache:
        return (
            _hierarchy_cache["dt_index"],
            _hierarchy_cache["detail_chunks"],
            _hierarchy_cache["tokenized_corpus"],
        )

    dt_index = faiss.read_index(DETAIL_FAISS_FILE)
    with open(DETAIL_CHUNKS_FILE, "rb") as f:
        detail_chunks: list[dict] = pickle.load(f)

    if os.path.exists(DETAIL_BM25_FILE):
        with open(DETAIL_BM25_FILE, "rb") as f:
            tokenized_corpus = pickle.load(f)
    else:
        tokenized_corpus = [_tokenize_for_bm25(c["text"]) for c in detail_chunks]
        with open(DETAIL_BM25_FILE, "wb") as f:
            pickle.dump(tokenized_corpus, f)

    _hierarchy_cache["dt_index"]         = dt_index
    _hierarchy_cache["detail_chunks"]    = detail_chunks
    _hierarchy_cache["tokenized_corpus"] = tokenized_corpus

    return dt_index, detail_chunks, tokenized_corpus


def invalidate_cache() -> None:
    """インデックス再構築後にメモリキャッシュをクリアする"""
    _hierarchy_cache.clear()


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ────────────────────────────────────────────────
# 階層ハイブリッド検索（STEP 4）
# ────────────────────────────────────────────────

def hierarchical_agent_search(
    query: str,
    api_key: str,
    chapter_index: list[dict],
    top_k: int = 5,
) -> tuple[list[dict], list[int]]:
    """
    章選択エージェント → 選択章内でFAISS＋BM25 → RRF統合。
    (チャンクdictリスト, 選択章番号リスト) を返す。
    """
    dt_index, detail_chunks, tokenized_corpus = _load_hierarchy()

    # 章選択
    selected_chapters = select_relevant_chapters(query, api_key, chapter_index)

    # 選択章に属するチャンクのインデックス
    target_indices = [
        i for i, c in enumerate(detail_chunks)
        if c["chapter_number"] in selected_chapters
    ]
    if not target_indices:
        target_indices = list(range(len(detail_chunks)))

    # FAISS 検索（全体から取得し、章フィルタ）
    q_vec = _get_single_embedding(query, api_key)
    faiss.normalize_L2(q_vec)
    search_k = min(len(detail_chunks), top_k * 5)
    distances, indices = dt_index.search(q_vec, search_k)
    faiss_results = [
        (int(idx), float(dist))
        for idx, dist in zip(indices[0], distances[0])
        if idx >= 0 and int(idx) in target_indices
    ][:top_k]

    # BM25 検索（章フィルタ済みコーパスで実施）
    target_set = set(target_indices)
    filtered_corpus = [
        (i, tokenized_corpus[i]) for i in target_indices
    ]
    if filtered_corpus:
        q_tokens = _tokenize_for_bm25(query)
        bm25 = BM25Okapi([c for _, c in filtered_corpus])
        scores = bm25.get_scores(q_tokens)
        top_local = np.argsort(scores)[::-1][:top_k]
        bm25_results = [
            (filtered_corpus[j][0], float(scores[j])) for j in top_local
        ]
    else:
        bm25_results = []

    # RRF 統合
    merged = _rrf_merge([faiss_results, bm25_results])[:top_k]
    result_chunks = [detail_chunks[i] for i in merged if i < len(detail_chunks)]
    return result_chunks, selected_chapters


# ────────────────────────────────────────────────
# メイン検索エントリポイント（公開）
# ────────────────────────────────────────────────

def retrieve(
    query: str,
    api_key: str,
    top_k: int = 5,
) -> tuple[str, dict]:
    """
    クエリ書き換え（3方向）→ 章選択 → 階層ハイブリッド検索 → RRF統合。
    最大1500トークンのコンテキスト文字列と検索ログdictを返す。
    インデックス未構築の場合は自動でbuild_indexを実行する。
    """
    needs_build = (
        not os.path.exists(DETAIL_FAISS_FILE)
        or not os.path.exists(DETAIL_CHUNKS_FILE)
    )
    if needs_build:
        build_index(api_key)

    # chapter_index の取得（キャッシュ優先）
    chapter_index = load_chapter_index()
    if not chapter_index:
        chapter_index = build_chapter_index(api_key)

    # クエリ書き換え
    queries = _rewrite_query(query, api_key)

    # 全クエリ分の結果を収集
    all_ranked: list[list[tuple[int, float]]] = []
    all_selected_chapters: list[int] = []
    all_chunks_map: dict[int, dict] = {}

    dt_index, detail_chunks, tokenized_corpus = _load_hierarchy()
    token_counts_log: dict[str, int] = {}

    for q in queries:
        result_chunks, sel_chaps = hierarchical_agent_search(
            q, api_key, chapter_index, top_k=top_k
        )
        all_selected_chapters.extend(sel_chaps)

        # 結果チャンクをグローバルインデックスで参照
        for chunk in result_chunks:
            # detail_chunks 内のインデックスを特定
            for gi, dc in enumerate(detail_chunks):
                if dc is chunk or (dc["text"] == chunk["text"] and dc["chapter_number"] == chunk["chapter_number"]):
                    all_chunks_map[gi] = dc
                    all_ranked.append([(gi, 1.0)])
                    break

    # 重複除去のためフラット化してRRF
    flat: list[list[tuple[int, float]]] = []
    seen_gi: set[int] = set()
    for gi, chunk in all_chunks_map.items():
        if gi not in seen_gi:
            flat.append([(gi, 1.0)])
            seen_gi.add(gi)

    merged_indices = _rrf_merge(flat)[:top_k]

    # トークン上限1500でコンテキスト構築
    result_parts: list[str] = []
    sources: list[dict] = []
    total_tokens = 0
    token_limit  = 1500

    for idx in merged_indices:
        if idx >= len(detail_chunks):
            continue
        chunk = detail_chunks[idx]
        t = _count_tokens(chunk["text"])
        if total_tokens + t > token_limit:
            remaining = token_limit - total_tokens
            if remaining > 50:
                enc = tiktoken.get_encoding("cl100k_base")
                truncated = enc.decode(enc.encode(chunk["text"])[:remaining])
                result_parts.append(truncated)
                sources.append({
                    "chapter_number": chunk["chapter_number"],
                    "chapter_title":  chunk["chapter_title"],
                    "section_title":  chunk["section_title"],
                })
            break
        result_parts.append(chunk["text"])
        sources.append({
            "chapter_number": chunk["chapter_number"],
            "chapter_title":  chunk["chapter_title"],
            "section_title":  chunk["section_title"],
        })
        total_tokens += t

    context_text = "\n\n---\n\n".join(result_parts)

    # 選択章（重複除去・ソート）
    unique_chapters = sorted(set(all_selected_chapters))
    chapter_labels = [
        f"{n}章: {next((c['title'] for c in chapter_index if c['chapter_number'] == n), str(n))}"
        for n in unique_chapters
    ]

    search_log = {
        "rewritten_queries":   queries,
        "selected_chapters":   unique_chapters,
        "chapter_labels":      chapter_labels,
        "sources":             sources,
        "total_context_tokens": total_tokens,
    }

    return context_text, search_log
