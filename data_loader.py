"""
data_loader.py
CSVデータの読み込み・前処理・サマリー生成モジュール
"""

import os
import pandas as pd
import numpy as np

# ファイルパス定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_FILE     = os.path.join(BASE_DIR, "sales_transaction_utf8.csv")
PRODUCTS_FILE  = os.path.join(BASE_DIR, "product_master_utf8.csv")
CUSTOMERS_FILE = os.path.join(BASE_DIR, "customer_attributes_utf8.csv")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "bakery_expert_knowledge.md")


def check_files() -> dict:
    """4ファイルの存在確認を行い、結果をdictで返す"""
    return {
        "sales":     os.path.exists(SALES_FILE),
        "products":  os.path.exists(PRODUCTS_FILE),
        "customers": os.path.exists(CUSTOMERS_FILE),
        "knowledge": os.path.exists(KNOWLEDGE_FILE),
    }


# ────────────────────────────────────────────────
# 売上データの読み込み・集計
# ────────────────────────────────────────────────

def _load_sales() -> pd.DataFrame:
    """売上CSVを読み込み、定休日除外・日付変換を行う"""
    df = pd.read_csv(SALES_FILE, encoding="cp932", sep="\t")

    # is_closed が 1 の行（定休日）を除外
    df = df[df["is_closed"] != 1].copy()

    # date列をdatetime型に変換
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")

    # 数値列をfloatに統一
    for col in ["sales_amount", "customer_count", "average_spend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["date", "sales_amount"])


def _summarize_sales(df: pd.DataFrame) -> dict:
    """月別・曜日別集計、直近トレンドを生成する"""
    df = df.copy()
    df["month"]      = df["date"].dt.to_period("M").astype(str)
    df["day_name"]   = df["date"].dt.dayofweek  # 0=月曜

    # 月別売上集計
    monthly = (
        df.groupby("month")["sales_amount"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total", "mean": "avg", "count": "days"})
    )
    monthly_str = monthly.tail(6).to_string()

    # 曜日別平均売上
    DOW_MAP = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    weekly = (
        df.groupby("day_name")["sales_amount"]
        .mean()
        .rename(index=DOW_MAP)
    )
    best_day  = weekly.idxmax()
    worst_day = weekly.idxmin()

    # 直近30日・90日トレンド
    latest_date = df["date"].max()
    last30 = df[df["date"] >= latest_date - pd.Timedelta(days=30)]
    last90 = df[df["date"] >= latest_date - pd.Timedelta(days=90)]
    prev30 = df[
        (df["date"] >= latest_date - pd.Timedelta(days=60)) &
        (df["date"] <  latest_date - pd.Timedelta(days=30))
    ]

    avg30  = last30["sales_amount"].mean()
    avg90  = last90["sales_amount"].mean()
    avg_p30 = prev30["sales_amount"].mean() if not prev30.empty else avg30

    trend30 = ((avg30 - avg_p30) / avg_p30 * 100) if avg_p30 else 0
    trend_str = (
        f"直近30日平均売上: {avg30:,.0f}円/日 "
        f"({'↑' if trend30 >= 0 else '↓'}{abs(trend30):.1f}% 前月比) / "
        f"直近90日平均: {avg90:,.0f}円/日"
    )

    return {
        "monthly_summary": monthly_str,
        "best_day":        best_day,
        "worst_day":       worst_day,
        "trend_text":      trend_str,
        "total_days":      len(df),
        "overall_avg":     df["sales_amount"].mean(),
    }


# ────────────────────────────────────────────────
# 商品データの読み込み・集計
# ────────────────────────────────────────────────

def _load_products() -> pd.DataFrame:
    """商品マスタCSVを読み込み、原価率を追加する"""
    df = pd.read_csv(PRODUCTS_FILE, encoding="utf-8-sig")

    # NaN行を除去
    df = df.dropna(subset=["sales_rank", "product_name"]).copy()

    # 原価率を計算（cost / price）
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["cost"]  = pd.to_numeric(df["cost"],  errors="coerce")
    df["cost_ratio"] = (df["cost"] / df["price"] * 100).round(1)

    # sales_composition_ratio を数値化
    df["comp_ratio_num"] = (
        df["sales_composition_ratio"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    return df


def _summarize_products(df: pd.DataFrame) -> dict:
    """ABC別集計・上位10商品リストを生成する"""
    # ABCランク別集計
    abc = (
        df.groupby("abc_rank")
        .agg(
            商品数=("product_name", "count"),
            売上構成比合計=("comp_ratio_num", "sum"),
            平均原価率=("cost_ratio", "mean"),
        )
        .round(1)
    )

    # 上位10商品リスト
    top10 = df.nsmallest(10, "sales_rank")[
        ["sales_rank", "product_name", "price", "comp_ratio_num", "abc_rank", "cost_ratio"]
    ]
    top10_str = top10.to_string(index=False)

    return {
        "abc_summary": abc.to_string(),
        "top10_str":   top10_str,
        "total_products": len(df),
        "a_count": len(df[df["abc_rank"] == "A"]),
        "b_count": len(df[df["abc_rank"] == "B"]),
        "c_count": len(df[df["abc_rank"] == "C"]),
    }


# ────────────────────────────────────────────────
# 客層データの読み込み・集計
# ────────────────────────────────────────────────

def _load_customers() -> pd.DataFrame:
    """客層CSVを読み込む"""
    df = pd.read_csv(CUSTOMERS_FILE, encoding="utf-8-sig")
    df["ratio_num"] = (
        df["ratio"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df


def _summarize_customers(df: pd.DataFrame) -> dict:
    """上位5客層・性別比率・年代分布を集計する"""
    # 上位5客層
    top5 = df.nsmallest(5, "rank")[["customer_age", "customer_gender", "ratio"]]
    top5_str = " / ".join(
        f"{r['customer_age']}_{r['customer_gender']}({r['ratio']})"
        for _, r in top5.iterrows()
    )

    # 性別比率
    gender_ratio = (
        df.groupby("customer_gender")["ratio_num"]
        .sum()
        .round(1)
    )
    gender_str = " / ".join(f"{g}:{v}%" for g, v in gender_ratio.items())

    # 年代分布（上位3位）
    age_ratio = (
        df.groupby("customer_age")["ratio_num"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    age_str = " / ".join(f"{a}:{v:.1f}%" for a, v in age_ratio.items())

    return {
        "top5_str":    top5_str,
        "gender_str":  gender_str,
        "top_ages":    age_str,
    }


# ────────────────────────────────────────────────
# メイン公開関数
# ────────────────────────────────────────────────

def get_data_summary() -> dict:
    """
    3CSVの分析結果をまとめてdictで返す。
    ai_summary_text キーに500文字以内のテキストサマリーを含む。
    """
    result = {}

    # 売上データ
    try:
        df_s = _load_sales()
        sales_info = _summarize_sales(df_s)
        result["sales"] = sales_info
    except Exception as e:
        result["sales"] = {"error": str(e)}
        sales_info = {}

    # 商品データ
    try:
        df_p = _load_products()
        prod_info = _summarize_products(df_p)
        result["products"] = prod_info
    except Exception as e:
        result["products"] = {"error": str(e)}
        prod_info = {}

    # 客層データ
    try:
        df_c = _load_customers()
        cust_info = _summarize_customers(df_c)
        result["customers"] = cust_info
    except Exception as e:
        result["customers"] = {"error": str(e)}
        cust_info = {}

    # AIに渡す抽象化テキストサマリー（具体的数値・商品名・個人情報を除外）
    lines = []

    # ── 売上傾向 ──
    lines.append("【売上傾向】")
    if sales_info and "error" not in sales_info:
        best = sales_info.get("best_day", "")
        worst = sales_info.get("worst_day", "")
        weekend = {"土", "日"}
        if best in weekend:
            lines.append("・週末（土日）に売上が集中し、平日との差が大きい")
        else:
            lines.append(f"・{best}曜日に売上が高く、週内で売上の差が見られる")
        if worst not in weekend:
            lines.append(f"・平日（特に{worst}曜日）は売上が低い傾向がある")
        trend_text = sales_info.get("trend_text", "")
        if "↑" in trend_text:
            lines.append("・直近1ヶ月は前月比で上昇傾向にある")
        elif "↓" in trend_text:
            lines.append("・直近1ヶ月は前月比でやや下降傾向にある")
        else:
            lines.append("・直近1ヶ月の売上は前月比で横ばい傾向にある")

    # ── 商品構成 ──
    lines.append("【商品構成】")
    if prod_info and "error" not in prod_info:
        try:
            abc_ratios = df_p.groupby("abc_rank")["comp_ratio_num"].sum()

            def _fmt_pct(val: float) -> str:
                """5%単位に丸める。5%未満は「数」で表現する"""
                rounded = round(float(val) / 5) * 5
                return str(rounded) if rounded >= 5 else "数"

            a_r = _fmt_pct(abc_ratios.get("A", 0))
            b_r = _fmt_pct(abc_ratios.get("B", 0))
            c_r = _fmt_pct(abc_ratios.get("C", 0))
            a_cost = df_p[df_p["abc_rank"] == "A"]["cost_ratio"].mean()
            lines.append(f"・Aランク商品（全体売上の約{a_r}%）：売上貢献度の高い定番・主力商品が中心")
            lines.append(f"・Bランク商品（全体売上の約{b_r}%）：中程度の売上を持つ準主力商品が中心")
            lines.append(f"・Cランク商品（全体売上の約{c_r}%）：売上比率は低いが品揃えに貢献する商品が中心")
            if not pd.isna(a_cost):
                lines.append(f"・Aランク商品の平均原価率は約{round(a_cost / 5) * 5}%程度")
        except Exception:
            lines.append(
                f"・Aランク:{prod_info.get('a_count',0)}品 / "
                f"Bランク:{prod_info.get('b_count',0)}品 / "
                f"Cランク:{prod_info.get('c_count',0)}品"
            )

    # ── 客層傾向 ──
    lines.append("【客層傾向】")
    if cust_info and "error" not in cust_info:
        try:
            # 女性比率を10%単位に丸める
            gender_ratio = df_c.groupby("customer_gender")["ratio_num"].sum()
            female_key = next((k for k in gender_ratio.index if "女" in str(k)), None)
            female_pct = gender_ratio[female_key] if female_key else 0
            female_approx = round(float(female_pct) / 10) * 10

            # 上位3年代を特定して主要年代範囲を生成
            age_ratios = (
                df_c.groupby("customer_age")["ratio_num"]
                .sum()
                .sort_values(ascending=False)
            )
            top_ages = age_ratios.head(3).index.tolist()
            lower_bounds = [
                int(str(a).strip().split("~")[0])
                for a in top_ages
                if str(a).strip().split("~")[0].isdigit()
            ]
            if lower_bounds:
                main_min = (min(lower_bounds) // 10) * 10
                main_max = (max(lower_bounds) // 10) * 10 + 9
                main_range = f"{main_min}〜{main_max}歳"
            else:
                main_range = "30〜50代"

            # 全年代の下限値から年齢分布範囲を生成
            all_lower = [
                int(str(a).strip().split("~")[0])
                for a in df_c["customer_age"].unique()
                if str(a).strip().split("~")[0].isdigit()
            ]
            dist_min = min(all_lower) if all_lower else 15
            dist_max = max(all_lower) + 9 if all_lower else 69

            lines.append(f"・主要客層は{main_range}の女性")
            lines.append(f"・女性客が全体の約{female_approx}%を占める")
            lines.append(f"・{dist_min}〜{dist_max}歳に広く分布しており、ファミリー層の来店も多い")
        except Exception:
            lines.append(f"・{cust_info.get('gender_str', '')}")

    result["ai_summary_text"] = "\n".join(lines)

    return result
