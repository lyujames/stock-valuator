"""取数层：yfinance（美股）+ 本地示例 JSON + 缓存。

yfinance 返回的字段名偶尔会变，所以每个指标都按「候选标签列表」依次尝试。
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE, "samples")
CACHE_DIR = os.path.join(BASE, "data", "cache")
CACHE_TTL = 12 * 3600
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,20}$")

# 报表字段 → yfinance 里可能出现的行标签
_INCOME = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "rd": ["Research And Development"],
    "shares": ["Diluted Average Shares", "Basic Average Shares"],
}
_CASHFLOW = {
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "depreciation": ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"],
}
_BALANCE = {
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
    "inventory": ["Inventory"],
}


class DataError(Exception):
    pass


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _rows(df, labels: list[str]) -> dict[int, float]:
    """从 DataFrame（行=科目，列=报告期）取第一个存在的科目，返回 {年份: 数值}。"""
    if df is None or getattr(df, "empty", True):
        return {}
    for lab in labels:
        if lab in df.index:
            out = {}
            for col in df.columns:
                v = _num(df.loc[lab, col])
                if v is not None:
                    out[col.year] = v
            if out:
                return out
    return {}


def normalize_yf(t: Any, ticker: str, risk_free: Optional[float] = None) -> dict:
    """把 yfinance.Ticker 对象整理成引擎使用的公司数据结构。"""
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    series: dict[str, dict[int, float]] = {}
    for df, mapping in ((t.income_stmt, _INCOME), (t.cashflow, _CASHFLOW), (t.balance_sheet, _BALANCE)):
        for key, labels in mapping.items():
            series[key] = _rows(df, labels)

    years = sorted(set().union(*[set(s) for s in series.values()]))
    rows = []
    for y in years:
        row = {"year": y}
        for key, s in series.items():
            if y in s:
                row[key] = s[y]
        if "capex" in row:
            row["capex"] = abs(row["capex"])
        if row.get("revenue") is not None or row.get("net_income") is not None:
            rows.append(row)
    rows = rows[-6:]
    if not rows:
        raise DataError(f"没有取到 {ticker} 的年度财报（代码是否正确？）")

    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice")) or _num(info.get("previousClose"))
    if price is None:
        try:
            price = _num(t.history(period="5d")["Close"].iloc[-1])
        except Exception:
            price = None
    if price is None:
        raise DataError(f"没有取到 {ticker} 的当前股价")

    shares = _num(info.get("sharesOutstanding")) or rows[-1].get("shares")

    analyst = None
    try:
        ge = t.growth_estimates
        for lab in ("LTG", "+5y", "+1y"):
            if lab in ge.index:
                for col in (ticker.upper(), "stockTrend", ge.columns[0]):
                    if col in ge.columns and _num(ge.loc[lab, col]) is not None:
                        analyst = _num(ge.loc[lab, col])
                        break
            if analyst is not None:
                break
    except Exception:
        analyst = None

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector": info.get("sector"),
        "currency": info.get("financialCurrency") or info.get("currency") or "USD",
        "price": price,
        "shares": shares,
        "eps_ttm": _num(info.get("trailingEps")),
        "forward_eps": _num(info.get("forwardEps")),
        "dividend_rate": _num(info.get("dividendRate")),
        "analyst_growth": analyst,
        "risk_free": risk_free,
        "years": rows,
        "source": "yfinance",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _risk_free() -> Optional[float]:
    import yfinance as yf
    try:
        v = _num(yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1])
        if v is not None:
            v = v / 100 if v > 1 else v
            if 0 < v < 0.15:
                return v
    except Exception:
        pass
    return None


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise DataError("代码格式不正确")
    return name


def list_samples() -> list[dict]:
    out = []
    if os.path.isdir(SAMPLES_DIR):
        for fn in sorted(os.listdir(SAMPLES_DIR)):
            if fn.endswith(".json"):
                with open(os.path.join(SAMPLES_DIR, fn), encoding="utf-8") as f:
                    c = json.load(f)
                out.append({"id": fn[:-5], "ticker": c.get("ticker"), "name": c.get("name")})
    return out


def load_sample(sample_id: str) -> dict:
    _check_name(sample_id)
    path = os.path.join(SAMPLES_DIR, sample_id + ".json")
    if not os.path.isfile(path):
        raise DataError("找不到该示例")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_us(ticker: str, refresh: bool = False) -> dict:
    ticker = _check_name(ticker.strip()).upper()
    cache_path = os.path.join(CACHE_DIR, ticker + ".json")
    if not refresh and os.path.isfile(cache_path) and time.time() - os.path.getmtime(cache_path) < CACHE_TTL:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    try:
        import yfinance as yf
    except ImportError:
        raise DataError("未安装 yfinance，请先运行：pip install -r requirements.txt")
    try:
        company = normalize_yf(yf.Ticker(ticker), ticker, _risk_free())
    except DataError:
        raise
    except Exception as e:  # 网络、限流等
        raise DataError(f"从 yfinance 取数失败：{e}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(company, f, ensure_ascii=False, indent=1)
    return company
