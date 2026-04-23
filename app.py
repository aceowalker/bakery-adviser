"""
app.py
Bakery AI Adviser - Streamlit メインアプリ
"""

import os
import streamlit as st
from dotenv import load_dotenv

# ────────────────────────────────────────────────
# ページ設定（最初に呼ぶ必要がある）
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="🍞 Bakery AI Adviser",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────
# カスタムCSS
# ────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap" rel="stylesheet">
<script>
(function() {
  var css = [
    "html, body, [class*='css'] { font-family: 'Noto Sans JP', sans-serif; }",
    ".stApp { background-color: #FDF6EC; }",
    "[data-testid='stSidebar'] { background-color: #FEFAF4; border-right: 1px solid #EDD9C0; }",
    "[data-testid='stSidebar'] > div:first-child { background-color: #FEFAF4; }",
    "[data-testid='stSidebar'] .stMarkdown p { color: #3D2B1F; }",
    "h1 { color: #3D2B1F !important; font-weight: 700 !important; }",
    "[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarAssistant']) { background-color: #ffffff; border-radius: 16px; padding: 16px; box-shadow: 0 4px 16px rgba(61,43,31,0.10); border-left: 4px solid #C8860A; margin-bottom: 12px; }",
    "[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) { background-color: #2C3E6B; border-radius: 16px; padding: 16px; box-shadow: 0 4px 16px rgba(44,62,107,0.20); margin-bottom: 12px; }",
    "[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) p { color: #ffffff !important; }",
    ".stButton > button { background-color: #FEFAF4; border: 1px solid #D4B896; color: #3D2B1F; border-radius: 20px; font-size: 13px; font-family: 'Noto Sans JP', sans-serif; transition: all 0.2s; width: 100%; }",
    ".stButton > button:hover { background-color: #C8860A; color: #ffffff; border-color: #C8860A; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(200,134,10,0.3); }",
    "[data-testid='stChatInput'] { border-color: #D4B896 !important; background-color: #FEFAF4 !important; border-radius: 24px !important; }",
    "hr { border-color: #EDD9C0; }",
    ".stAlert { background-color: #FFF8EE; border-color: #C8860A; border-radius: 12px; }",
    ".stSpinner > div { border-top-color: #C8860A !important; }",
    ".sidebar-section { font-size: 13px; font-weight: 700; color: #8B5E08; text-transform: uppercase; letter-spacing: 0.08em; margin: 16px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #EDD9C0; }"
  ].join(" ");
  var el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
})();
</script>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# モジュールのインポート
# ────────────────────────────────────────────────
from data_loader import check_files, get_data_summary
from rag_engine import build_index, retrieve, get_chunk_count
from advisor_agent import generate_advice, generate_quick_analysis

# ────────────────────────────────────────────────
# APIキーの取得（優先順位：st.secrets → .env）
# ────────────────────────────────────────────────
def _get_api_key() -> str:
    """st.secrets → 環境変数の順でAPIキーを取得する"""
    # Streamlit Cloud 環境
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key and key.startswith("sk-"):
            return key
    except Exception:
        pass
    # ローカル環境（.env）
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key.startswith("sk-"):
        return key
    return ""

# ────────────────────────────────────────────────
# session_state の初期化
# ────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "api_key" not in st.session_state:
    st.session_state.api_key = _get_api_key()

# ────────────────────────────────────────────────
# データサマリーをキャッシュして取得
# ────────────────────────────────────────────────
@st.cache_data
def cached_data_summary() -> dict:
    """起動時に1回だけ実行してキャッシュする"""
    return get_data_summary()

# ────────────────────────────────────────────────
# ════════════════════════════════════════════════
# サイドバー
# ════════════════════════════════════════════════
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍞 Bakery AI Adviser")

    # ① 接続ステータス
    st.markdown('<div class="sidebar-section">接続ステータス</div>', unsafe_allow_html=True)
    if st.session_state.api_key:
        st.markdown("🟢 **AI接続済み**", unsafe_allow_html=False)
    else:
        st.markdown("🔴 **未接続**", unsafe_allow_html=False)
        st.error("OpenAI APIキーが設定されていません。\n`.env` ファイルに `OPENAI_API_KEY` を設定してください。")
        st.stop()

    # ② データファイルの状態
    st.markdown('<div class="sidebar-section">📂 データファイル</div>', unsafe_allow_html=True)
    files = check_files()
    st.write("✅ 売上推移データ" if files["sales"]     else "❌ 売上推移データ")
    st.write("✅ ABC分析データ"  if files["products"]  else "❌ ABC分析データ")
    st.write("✅ 客層データ"     if files["customers"] else "❌ 客層データ")
    st.write("✅ 専門ナレッジ"   if files["knowledge"] else "❌ 専門ナレッジ")

    # ③ RAGナレッジ状態
    st.markdown('<div class="sidebar-section">🧠 RAGナレッジ</div>', unsafe_allow_html=True)

    # 起動時にインデックスを自動ロード（未存在なら構築）
    if not st.session_state.rag_ready:
        chunk_count = get_chunk_count()
        if chunk_count == 0:
            with st.spinner("インデックスを構築中..."):
                chunk_count = build_index(st.session_state.api_key)
        st.session_state.rag_ready = True

    chunk_count = get_chunk_count()
    st.write(f"📚 インデックス済み：{chunk_count} チャンク")

    if st.button("🔄 インデックス再構築"):
        with st.spinner("再構築中..."):
            n = build_index(st.session_state.api_key)
        st.success(f"再構築完了：{n} チャンク")
        st.rerun()

    # ④ クイック分析ボタン
    st.markdown('<div class="sidebar-section">⚡ クイック分析</div>', unsafe_allow_html=True)

    quick_buttons = [
        ("📊 ABC分析から売り込み戦略",  "ABC分析の結果をもとに、売り込み戦略を提案してください。"),
        ("🌸 新商品アイデア提案",        "客層データと売上トレンドをもとに、新商品のアイデアを提案してください。"),
        ("🎪 イベント企画提案",          "売上向上につながるイベント企画を提案してください。"),
        ("📈 今月の売上向上施策",        "今月すぐに実施できる売上向上施策を提案してください。"),
        ("🔍 総合データ分析を実行",      "売上・商品・客層データを総合的に分析して、改善提案をしてください。"),
    ]
    for label, message in quick_buttons:
        if st.button(label, key=f"quick_{label}"):
            st.session_state.user_input = message
            st.rerun()

    # ⑤ 会話リセット
    st.markdown('<div class="sidebar-section">設定</div>', unsafe_allow_html=True)
    if st.button("🗑️ 会話をリセット"):
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.rerun()

# ────────────────────────────────────────────────
# ════════════════════════════════════════════════
# メインエリア
# ════════════════════════════════════════════════
# ────────────────────────────────────────────────

# ヘッダー
st.title("🍞 Bakery AI Adviser")
st.caption("powered by GPT-4o-mini × RAG")
st.divider()

# データサマリーをロード（キャッシュ）
if st.session_state.data_summary is None:
    st.session_state.data_summary = cached_data_summary()
data_summary_text = st.session_state.data_summary.get("ai_summary_text", "")

# ウェルカムメッセージ（初回のみ）
if not st.session_state.chat_history:
    st.info(
        "売上データ・ABC分析・客層情報をAIが総合分析し、"
        "専門コンサルタント知識ベース（RAG）をもとに具体的な売上向上アドバイスを提供します。"
    )

    # クイック選択タグ（4列）
    tag_cols = st.columns(4)
    quick_tags = [
        ("📊 ABC分析・売り込み",  "ABC分析の結果をもとに、売り込み戦略を提案してください。"),
        ("🌿 新商品開発",          "客層データと売上トレンドをもとに、新商品のアイデアを提案してください。"),
        ("🎉 イベント企画",        "売上向上につながるイベント企画を提案してください。"),
        ("💰 売上アップ施策",      "今月すぐに実施できる売上向上施策を提案してください。"),
    ]
    for col, (label, message) in zip(tag_cols, quick_tags):
        with col:
            if st.button(label, key=f"tag_{label}"):
                st.session_state.user_input = message
                st.rerun()

# チャット履歴の表示
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
        else:
            st.write(msg["content"])

# ────────────────────────────────────────────────
# メッセージ入力と送信処理
# ────────────────────────────────────────────────

# クイックボタン経由のメッセージを処理
pending_input = st.session_state.user_input

# チャット入力欄（画面下部に固定）
user_typed = st.chat_input(
    placeholder="質問を入力してください（Enterで送信）",
)

# 入力元の確定（直接入力 or クイックボタン）
user_message = user_typed or pending_input

if user_message:
    # クイックボタン入力をリセット
    st.session_state.user_input = ""

    # ユーザーメッセージを履歴に追加して即時表示
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    # AI回答を生成
    with st.chat_message("assistant"):
        with st.spinner("🍞 アドバイスを生成中..."):
            # RAGコンテキストを取得
            rag_context = retrieve(user_message, st.session_state.api_key)

            # AIアドバイスを生成
            ai_response = generate_advice(
                user_message=user_message,
                history=st.session_state.chat_history[:-1],  # 今回分を除いた履歴
                data_summary=data_summary_text,
                rag_context=rag_context,
                api_key=st.session_state.api_key,
            )

        st.markdown(ai_response)

    # AI回答を履歴に追加
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
