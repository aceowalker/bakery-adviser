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

    # AIに渡す圧縮テキストサマリー（500文字以内）
    lines = []
    if sales_info and "error" not in sales_info:
        lines.append(f"【売上】{sales_info.get('trend_text', '')}")
        lines.append(f"売上上位曜日:{sales_info.get('best_day','')} / 下位:{sales_info.get('worst_day','')}")
    if prod_info and "error" not in prod_info:
        lines.append(
            f"【商品】全{prod_info.get('total_products',0)}品 "
            f"A:{prod_info.get('a_count',0)}品 B:{prod_info.get('b_count',0)}品 C:{prod_info.get('c_count',0)}品"
        )
        lines.append(f"上位商品: {prod_info.get('top10_str','')[:120]}")
    if cust_info and "error" not in cust_info:
        lines.append(f"【客層】{cust_info.get('top5_str','')}")
        lines.append(f"性別:{cust_info.get('gender_str','')} / 上位年代:{cust_info.get('top_ages','')}")

    summary_text = "\n".join(lines)
    # 500文字に切り詰め
    result["ai_summary_text"] = summary_text[:500]

    return result
