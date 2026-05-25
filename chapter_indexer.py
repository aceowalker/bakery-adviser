"""
chapter_indexer.py
bakery_expert_knowledge.md の章構造を抽出し chapter_index.json を生成・キャッシュする。
"""

import json
import os
import re

from openai import OpenAI

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "bakery_expert_knowledge.md")
INDEX_FILE     = os.path.join(BASE_DIR, "chapter_index.json")

CHAT_MODEL = "gpt-4o-mini"


def _extract_chapters(text: str) -> list[dict]:
    """## レベルの見出しで章を分割し、番号・タイトル・本文を返す"""
    parts = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    chapters = []
    for part in parts:
        part = part.strip()
        m = re.match(r'^## (\d+)\.\s+(.+?)$', part, re.MULTILINE)
        if not m:
            continue
        num   = int(m.group(1))
        title = m.group(2).strip()
        body  = part[m.end():].strip()
        chapters.append({"chapter_number": num, "title": title, "body": body})
    return chapters


def _generate_keywords_and_summary(chapter: dict, client: OpenAI) -> dict:
    """gpt-4o-mini でキーワードとサマリーを生成する"""
    prompt = (
        "以下のベーカリー経営ナレッジの章を読んで、\n"
        "検索キーワード（5〜8個）と1行サマリー（30字以内）を生成してください。\n\n"
        "【章タイトル】\n"
        f"{chapter['title']}\n\n"
        "【章の内容（抜粋）】\n"
        f"{chapter['body'][:600]}\n\n"
        "【出力形式（JSON）】\n"
        '{"keywords": ["キーワード1", ...], "summary": "1行サマリー"}\n\n'
        "JSON以外は一切出力しないこと。"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    # コードブロック除去
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        return {
            "keywords": data.get("keywords", []),
            "summary":  data.get("summary", ""),
        }
    except json.JSONDecodeError:
        return {"keywords": [], "summary": chapter["title"]}


def build_chapter_index(api_key: str) -> list[dict]:
    """
    chapter_index.json が存在すればそれを返す（キャッシュ）。
    なければナレッジベースから章を抽出しLLMでメタデータを生成して保存する。
    """
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data["chapters"]

    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        content = f.read()

    chapters_raw = _extract_chapters(content)
    client = OpenAI(api_key=api_key)

    chapters_out = []
    for ch in chapters_raw:
        meta = _generate_keywords_and_summary(ch, client)
        chapters_out.append({
            "chapter_number": ch["chapter_number"],
            "title":          ch["title"],
            "keywords":       meta["keywords"],
            "summary":        meta["summary"],
        })

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump({"chapters": chapters_out}, f, ensure_ascii=False, indent=2)

    return chapters_out


def load_chapter_index() -> list[dict]:
    """保存済み chapter_index.json を読み込む（未生成時は空リスト）"""
    if not os.path.exists(INDEX_FILE):
        return []
    with open(INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)["chapters"]
