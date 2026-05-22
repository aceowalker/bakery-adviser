"""
advisor_agent.py
AIエージェント：売上向上アドバイス生成モジュール
"""

from openai import OpenAI

# 使用モデル（コスト最適化）
MODEL = "gpt-4o-mini"

# システムプロンプトのテンプレート
SYSTEM_PROMPT_TEMPLATE = """あなたはベーカリー専門の売上向上コンサルタントです。
以下の店舗データと専門ナレッジを参考に、具体的で実践的なアドバイスをしてください。

【店舗データサマリー】
{data_summary}

【内部参考資料（以下のルールを厳守すること）】
・以下の資料はあなたの思考の参考にのみ使用すること
・資料の文章を回答に直接引用・転載・要約転載することを禁止する
・資料に書かれた内容は、あなた自身がベーカリーコンサルタントとして
培った知見として、完全に自分の言葉で表現し直して回答すること
・「資料によると」「ナレッジベースには」などの出典を示す表現も禁止する
{rag_context}

回答は必ず以下の構造で返してください：

**【現状分析】**
（2〜3行で現状を端的に分析）

**【具体的アドバイス】**
- （アドバイス1）
- （アドバイス2）
- （アドバイス3）
- （アドバイス4）
- （アドバイス5）

**【優先アクション ⭐】**
（最重要の1点を強調して記載）
"""

# クイック分析用プロンプトのテンプレート
QUICK_ANALYSIS_TEMPLATE = """あなたはベーカリー専門の売上向上コンサルタントです。
以下の店舗データと専門ナレッジをもとに、総合的な分析レポートを作成してください。

【店舗データサマリー】
{data_summary}

【内部参考資料（以下のルールを厳守すること）】
・以下の資料はあなたの思考の参考にのみ使用すること
・資料の文章を回答に直接引用・転載・要約転載することを禁止する
・資料に書かれた内容は、あなた自身がベーカリーコンサルタントとして
培った知見として、完全に自分の言葉で表現し直して回答すること
・「資料によると」「ナレッジベースには」などの出典を示す表現も禁止する
{rag_context}

以下の構成で詳細なレポートを作成してください：

**【総合評価】**
（店舗の現状を総括、強みと課題を明確に）

**【売上分析】**
（売上トレンドと曜日別パターンから読み取れる示唆）

**【商品戦略分析】**
（ABC分析結果をもとにした商品構成の評価と改善提案）

**【客層分析】**
（客層データから読み取れるターゲット像と訴求ポイント）

**【今すぐできる改善アクション TOP3】**
1. （最優先アクション）
2. （2番目のアクション）
3. （3番目のアクション）
"""


def generate_advice(
    user_message: str,
    history: list,
    data_summary: str,
    rag_context: str,
    api_key: str,
) -> str:
    """
    ユーザーの質問に対してAIアドバイスを生成する。

    Args:
        user_message: ユーザーの質問テキスト
        history: 会話履歴（[{"role": ..., "content": ...}, ...] 形式）
        data_summary: 店舗データのテキストサマリー
        rag_context: RAGで取得した関連ナレッジ
        api_key: OpenAI APIキー

    Returns:
        AIが生成したアドバイステキスト
    """
    try:
        client = OpenAI(api_key=api_key)

        # システムプロンプトを組み立て
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            data_summary=data_summary or "（データなし）",
            rag_context=rag_context or "（ナレッジなし）",
        )

        # messagesを構成（system + 直近10件の履歴 + 今回の質問）
        messages = [{"role": "system", "content": system_prompt}]

        # 直近10件のみ渡す（トークン節約）
        recent_history = history[-10:] if len(history) > 10 else history
        messages.extend(recent_history)

        # 今回のユーザーメッセージを追加
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        # エラー時は日本語のフォールバックメッセージを返す
        return (
            f"⚠️ アドバイスの生成中にエラーが発生しました。\n\n"
            f"エラー内容：{str(e)}\n\n"
            "しばらく時間をおいて再度お試しください。"
            "問題が続く場合はAPIキーの設定をご確認ください。"
        )


def generate_quick_analysis(
    data_summary: str,
    rag_context: str,
    api_key: str,
) -> str:
    """
    売上・ABC分析・客層を総合した自動分析レポートを生成する。

    Args:
        data_summary: 店舗データのテキストサマリー
        rag_context: RAGで取得した関連ナレッジ
        api_key: OpenAI APIキー

    Returns:
        総合分析レポートのテキスト
    """
    try:
        client = OpenAI(api_key=api_key)

        prompt = QUICK_ANALYSIS_TEMPLATE.format(
            data_summary=data_summary or "（データなし）",
            rag_context=rag_context or "（ナレッジなし）",
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return (
            f"⚠️ 分析レポートの生成中にエラーが発生しました。\n\n"
            f"エラー内容：{str(e)}\n\n"
            "しばらく時間をおいて再度お試しください。"
        )
