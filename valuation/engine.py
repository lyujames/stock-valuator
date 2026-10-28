"""估值引擎：巴菲特流派 / 成长股 / GARP。

纯标准库实现。入口 valuate(company, params) 返回可直接 JSON 化的结果。
所有金额为公司报表币种的绝对值（如美元），比率为小数（0.1 = 10%）。
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Optional

DEFAULT_PARAMS: dict[str, Any] = {
    "discount_rate": 0.10,        # 折现率
    "terminal_growth": 0.025,     # 永续增长率
    "high_growth_years": 5,       # 高增长期年数
    "fade_years": 5,              # 增速衰减期年数（线性降到永续增长率）
    "growth_override": None,      # 自定义增速（覆盖自动估计）
    "safety_margin": 0.25,        # 安全边际要求
    "required_return": 0.12,      # 成长股要求的年化回报
    "exit_pe": None,              # 5 年后退出 PE（None = 按类型自动）
    "peg_fair": 1.0,              # 合理 PEG
    "buffett_growth_cap": 0.10,   # 巴菲特流派对增速的保守上限
    "growth_cap": 0.35,           # 自动估计增速的上限
    "scenario_g_delta": 0.04,     # 悲观/乐观情景：增速偏移
    "scenario_r_delta": 0.01,     # 悲观/乐观情景：折现率偏移
    "price_override": None,
    "risk_free_override": None,
    "company_type_override": None,
}

_RANGES = {
    "discount_rate": (0.04, 0.25),
    "terminal_growth": (0.0, 0.05),
    "high_growth_years": (1, 10),
    "fade_years": (0, 10),
    "safety_margin": (0.0, 0.6),
    "required_return": (0.04, 0.40),
    "peg_fair": (0.3, 3.0),
    "buffett_growth_cap": (0.0, 0.20),
    "growth_cap": (0.05, 0.60),
    "scenario_g_delta": (0.0, 0.15),
    "scenario_r_delta": (0.0, 0.05),
}
_OPTIONAL_NUM = ("growth_override", "exit_pe", "price_override", "risk_free_override")
_TYPES = ("mature", "mid_growth", "high_growth", "loss", "financial")

TYPE_LABELS = {
    "mature": "成熟稳健型",
    "mid_growth": "中速成长型",
    "high_growth": "高速成长型",
    "loss": "亏损 / 盈利不稳型",
    "financial": "金融股",
}

# 各类型下三个流派的权重
_WEIGHTS = {
    "mature":      {"buffett": 0.60, "garp": 0.25, "growth": 0.15},
    "mid_growth":  {"buffett": 0.35, "garp": 0.40, "growth": 0.25},
    "high_growth": {"buffett": 0.15, "garp": 0.35, "growth": 0.50},
    "loss":        {"buffett": 0.00, "garp": 0.00, "growth": 1.00},
    "financial":   {"buffett": 1.00, "garp": 0.00, "growth": 0.00},
}
_DEFAULT_EXIT_PE = {"mature": 15.0, "mid_growth": 20.0, "high_growth": 25.0, "loss": 25.0}


# ---------------------------------------------------------------- helpers
def _f(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _cagr(first: Optional[float], last: Optional[float], n: int) -> Optional[float]:
    if first is None or last is None or n <= 0 or first <= 0 or last <= 0:
        return None
    return (last / first) ** (1 / n) - 1


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def clean_params(raw: Optional[dict]) -> dict:
    p = dict(DEFAULT_PARAMS)
    for k, v in (raw or {}).items():
        if k not in p:
            continue
        if k in _RANGES:
            fv = _f(v)
            if fv is not None:
                lo, hi = _RANGES[k]
                p[k] = _clamp(fv, lo, hi)
        elif k in _OPTIONAL_NUM:
            p[k] = _f(v)
        elif k == "company_type_override":
            p[k] = v if v in _TYPES else None
    p["high_growth_years"] = int(round(p["high_growth_years"]))
    p["fade_years"] = int(round(p["fade_years"]))
    p["terminal_growth"] = min(p["terminal_growth"], p["discount_rate"] - 0.01)
    return p


# ------------------------------------------------------------ core models
def dcf_per_share(base_cf, g1, r, g_term, high_years, fade_years, net_cash, shares):
    """多阶段 DCF：高增长期 g1 → 线性衰减到 g_term → 永续。返回 {value, tv_share}。"""
    if base_cf is None or base_cf <= 0 or not shares or shares <= 0:
        return None
    if r <= g_term:
        return None
    total = high_years + fade_years
    cf, pv, g = base_cf, 0.0, g1
    for t in range(1, total + 1):
        if t <= high_years:
            g = g1
        else:
            g = g1 + (g_term - g1) * (t - high_years) / (fade_years + 1)
        cf *= 1 + g
        pv += cf / (1 + r) ** t
    tv = cf * (1 + g_term) / (r - g_term)
    pv_tv = tv / (1 + r) ** total
    equity = pv + pv_tv + (net_cash or 0.0)
    return {"value": max(equity / shares, 0.0), "tv_share": pv_tv / (pv + pv_tv)}


def solve_implied_growth(price, base_cf, r, g_term, high_years, fade_years, net_cash, shares):
    """反向 DCF：二分法求使 DCF 价值 = 当前股价的高增长期增速。"""
    lo, hi = -0.20, 0.60

    def val(g):
        res = dcf_per_share(base_cf, g, r, g_term, high_years, fade_years, net_cash, shares)
        return None if res is None else res["value"]

    vlo, vhi = val(lo), val(hi)
    if vlo is None or vhi is None:
        return {"status": "n/a", "growth": None}
    if vlo > price:
        return {"status": "below", "growth": lo}
    if vhi < price:
        return {"status": "above", "growth": hi}
    for _ in range(60):
        mid = (lo + hi) / 2
        if val(mid) < price:
            lo = mid
        else:
            hi = mid
    return {"status": "ok", "growth": (lo + hi) / 2}


def graham_number(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


# ---------------------------------------------------------------- derive
def derive(company: dict) -> list[dict]:
    rows = sorted(company.get("years") or [], key=lambda r: r["year"])
    out: list[dict] = []
    prev: Optional[dict] = None
    for r in rows:
        rev, ni, gp, op = (_f(r.get(k)) for k in ("revenue", "net_income", "gross_profit", "operating_income"))
        ocf, capex, da = _f(r.get("ocf")), _f(r.get("capex")), _f(r.get("depreciation"))
        eq, debt, cash = _f(r.get("equity")), _f(r.get("total_debt")), _f(r.get("cash"))
        sh, inv, rd = _f(r.get("shares")), _f(r.get("inventory")), _f(r.get("rd"))
        eps = _f(r.get("eps"))
        capex = abs(capex) if capex is not None else None
        if eps is None and ni is not None and sh:
            eps = ni / sh
        d: dict[str, Any] = {"year": r["year"], "revenue": rev, "net_income": ni, "eps": eps,
                             "shares": sh, "equity": eq, "inventory": inv, "operating_income": op}
        d["gross_margin"] = gp / rev if gp is not None and rev else None
        d["op_margin"] = op / rev if op is not None and rev else None
        d["net_margin"] = ni / rev if ni is not None and rev else None
        d["fcf"] = ocf - capex if ocf is not None and capex is not None else None
        d["fcf_margin"] = d["fcf"] / rev if d["fcf"] is not None and rev else None
        d["fcf_ni"] = d["fcf"] / ni if d["fcf"] is not None and ni and ni > 0 else None
        d["capex_ocf"] = capex / ocf if capex is not None and ocf and ocf > 0 else None
        d["rd_ratio"] = rd / rev if rd is not None and rev else None
        d["de"] = debt / eq if debt is not None and eq and eq > 0 else None
        d["bvps"] = eq / sh if eq is not None and sh else None
        d["roe"] = None
        if ni is not None and eq and eq > 0:
            peq = prev.get("equity") if prev else None
            d["roe"] = ni / ((eq + peq) / 2 if peq and peq > 0 else eq)
        ic = eq + debt - cash if None not in (eq, debt, cash) else None
        d["roic"] = op * (1 - 0.21) / ic if op is not None and ic and ic > 0 else None  # 税率按 21% 近似
        if ni is not None and da is not None and capex is not None:
            d["owner_earnings"] = ni + da - min(capex, da)   # 维持性资本开支≈min(资本开支, 折旧摊销)
        else:
            d["owner_earnings"] = d["fcf"]
        d["net_cash"] = cash - debt if cash is not None and debt is not None else None
        for key, src in (("rev_yoy", "revenue"), ("ni_yoy", "net_income"), ("eps_yoy", "eps"), ("inv_yoy", "inventory")):
            a, b = d.get(src), prev.get(src) if prev else None
            d[key] = a / b - 1 if a is not None and b and b > 0 else None
        out.append(d)
        prev = d
    return out


def estimate_growth(company: dict, d: list[dict], params: dict) -> tuple[float, str]:
    if params["growth_override"] is not None:
        return _clamp(params["growth_override"], -0.2, 0.6), "自定义"
    cap = params["growth_cap"]
    analyst = _f(company.get("analyst_growth"))
    if analyst is not None:
        return _clamp(analyst, -0.1, cap), "分析师预期"
    n = len(d) - 1
    parts = []
    if n >= 1:
        parts.append(_cagr(d[0]["revenue"], d[-1]["revenue"], n))
        eps0 = next((x["eps"] for x in d if x["eps"] and x["eps"] > 0), None)
        parts.append(_cagr(eps0, d[-1]["eps"], n) if eps0 else None)
    g = _mean(parts)
    if g is None:
        return 0.05, "默认值（缺少历史数据）"
    return _clamp(g, -0.05, cap), f"历史 {n} 年 CAGR（营收与 EPS 均值）"


def classify(company: dict, d: list[dict], g: float, params: dict) -> str:
    if params["company_type_override"]:
        return params["company_type_override"]
    sector = (company.get("sector") or "").lower()
    if "financial" in sector or "bank" in sector or "insurance" in sector:
        return "financial"
    last = d[-1] if d else {}
    eps = _f(company.get("eps_ttm")) or last.get("eps")
    if (last.get("net_income") is not None and last["net_income"] <= 0) or (eps is not None and eps <= 0):
        return "loss"
    if g >= 0.25:
        return "high_growth"
    if g >= 0.10:
        return "mid_growth"
    return "mature"


# ----------------------------------------------------------------- checks
def _chk(name, value, target, ok, fmt="pct"):
    if value is None:
        return {"name": name, "value_text": "数据缺失", "target_text": target, "pass": None}
    txt = _pct(value) if fmt == "pct" else f"{value:.2f}"
    return {"name": name, "value_text": txt, "target_text": target, "pass": bool(ok(value))}


def _score(checks) -> Optional[int]:
    ps = [c for c in checks if c["pass"] is not None]
    return round(100 * sum(1 for c in ps if c["pass"]) / len(ps)) if ps else None


def _vals(d, key):
    return [x[key] for x in d if x.get(key) is not None]


def quality_checks(d: list[dict], ctype: str) -> list[dict]:
    roes = _vals(d, "roe")
    c = [
        _chk("平均 ROE", _mean(roes), "≥ 15%", lambda v: v >= 0.15),
        _chk("最低年度 ROE", min(roes) if roes else None, "≥ 10%", lambda v: v >= 0.10),
        _chk("各年均盈利（最低净利率）", min(_vals(d, "net_margin")) if _vals(d, "net_margin") else None, "> 0", lambda v: v > 0),
    ]
    if ctype == "financial":
        return c  # 银行/保险不看毛利率、FCF、资本开支
    gms = _vals(d, "gross_margin")
    n = len(d) - 1
    eps0 = next((x["eps"] for x in d if x["eps"] and x["eps"] > 0), None)
    eps_cagr = _cagr(eps0, d[-1]["eps"], n) if eps0 and n >= 1 else None
    c += [
        _chk("平均毛利率（护城河代理）", _mean(gms), "≥ 35%", lambda v: v >= 0.35),
        _chk("毛利率波动（最大 − 最小）", (max(gms) - min(gms)) if len(gms) >= 2 else None, "≤ 6 个百分点", lambda v: v <= 0.06),
        _chk("平均净利率", _mean(_vals(d, "net_margin")), "≥ 10%", lambda v: v >= 0.10),
        _chk("自由现金流 ÷ 净利润（盈利含金量）", _mean(_vals(d, "fcf_ni")), "≥ 0.8", lambda v: v >= 0.8, "x"),
        _chk("资本开支 ÷ 经营现金流（轻资产）", _mean(_vals(d, "capex_ocf")), "≤ 50%", lambda v: v <= 0.5),
        _chk("平均 ROIC（税率按 21% 近似）", _mean(_vals(d, "roic")), "≥ 12%", lambda v: v >= 0.12),
        _chk("负债 ÷ 权益（最新）", d[-1].get("de"), "≤ 0.8", lambda v: v <= 0.8, "x"),
        _chk("EPS 年化增速", eps_cagr, "> 0", lambda v: v > 0),
    ]
    return c


def growth_checks(d: list[dict]) -> list[dict]:
    n = len(d) - 1
    rev_cagr = _cagr(d[0]["revenue"], d[-1]["revenue"], n) if n >= 1 else None
    last_yoy = d[-1].get("rev_yoy")
    gm0, gm1 = d[0].get("gross_margin"), d[-1].get("gross_margin")
    ni_cagr = _cagr(d[0]["net_income"], d[-1]["net_income"], n) if n >= 1 else None
    sh_cagr = _cagr(d[0]["shares"], d[-1]["shares"], n) if n >= 1 else None
    r40 = last_yoy + d[-1]["fcf_margin"] if last_yoy is not None and d[-1].get("fcf_margin") is not None else None
    slow_target = f"≥ 0.7 × CAGR（{_pct(0.7 * rev_cagr)}）" if rev_cagr is not None else "≥ 0.7 × CAGR"
    return [
        _chk("营收 CAGR", rev_cagr, "≥ 15%", lambda v: v >= 0.15),
        _chk("最近一年营收增速（未明显放缓）", last_yoy, slow_target, lambda v: rev_cagr is not None and v >= 0.7 * rev_cagr),
        _chk("毛利率变化（最新 − 最早）", gm1 - gm0 if gm0 is not None and gm1 is not None else None, "≥ −2 个百分点", lambda v: v >= -0.02),
        _chk("经营杠杆（利润 CAGR − 营收 CAGR）", ni_cagr - rev_cagr if ni_cagr is not None and rev_cagr is not None else None, "≥ 0", lambda v: v >= 0),
        _chk("Rule of 40（营收增速 + FCF 利润率）", r40, "≥ 40%", lambda v: v >= 0.40),
        _chk("股本年化变化（稀释）", sh_cagr, "≤ 2%", lambda v: v <= 0.02),
        _chk("最新自由现金流利润率（未烧钱）", d[-1].get("fcf_margin"), "> 0", lambda v: v > 0),
    ]


def garp_checks(d: list[dict], g: float) -> list[dict]:
    yoys = _vals(d, "rev_yoy")
    last = d[-1]
    n = len(d) - 1
    eps0 = next((x["eps"] for x in d if x["eps"] and x["eps"] > 0), None)
    eps_cagr = _cagr(eps0, last["eps"], n) if eps0 and n >= 1 else None
    inv_gap = last["inv_yoy"] - last["rev_yoy"] if last.get("inv_yoy") is not None and last.get("rev_yoy") is not None else None
    return [
        _chk("负债 ÷ 权益（最新）", last.get("de"), "≤ 1.0", lambda v: v <= 1.0, "x"),
        _chk("库存增速 − 营收增速（林奇警示）", inv_gap, "≤ 5 个百分点", lambda v: v <= 0.05),
        _chk("营收增速稳定性（标准差）", statistics.pstdev(yoys) if len(yoys) >= 3 else None, "≤ 8 个百分点", lambda v: v <= 0.08),
        _chk("EPS 年化增速（需为正）", eps_cagr, "> 0", lambda v: v > 0),
        _chk("增速处于 PEG 适用区间", g, "8% – 35%", lambda v: 0.08 <= v <= 0.35),
    ]


# ------------------------------------------------------------------ school
def _verdict(price, base, mos):
    if base is None or base <= 0:
        return "na", "无法判断"
    if price <= base * (1 - mos):
        return "under", "低估"
    if price <= base * 1.15:
        return "fair", "合理"
    return "over", "高估"


def _finalize(res: dict, price: float, mos: float) -> dict:
    counted = [m for m in res["methods"] if m.get("counted")]
    if counted:
        res["value"] = {k: sum(m[k] for m in counted) / len(counted) for k in ("low", "base", "high")}
        res["premium"] = price / res["value"]["base"] - 1 if res["value"]["base"] > 0 else None
        res["verdict"], res["verdict_text"] = _verdict(price, res["value"]["base"], mos)
    else:
        res["value"], res["premium"] = None, None
        res["verdict"], res["verdict_text"] = "na", "无法估值"
    return res


def _scen(params):
    dg, dr = params["scenario_g_delta"], params["scenario_r_delta"]
    return {"low": (-dg, dr), "base": (0.0, 0.0), "high": (dg, -dr)}


def _rr(r, dr, term):
    return max(r + dr, term + 0.01)


def _dcf_method(name, base_cf, g, params, net_cash, shares, note, cap_g=None):
    sc = _scen(params)
    vals, tv = {}, None
    for k, (dg, dr) in sc.items():
        gg = g + dg
        if cap_g is not None:
            gg = min(gg, cap_g)
        res = dcf_per_share(base_cf, gg, _rr(params["discount_rate"], dr, params["terminal_growth"]),
                            params["terminal_growth"], params["high_growth_years"], params["fade_years"], net_cash, shares)
        if res is None:
            return None
        vals[k] = res["value"]
        if k == "base":
            tv = res["tv_share"]
    return {"name": name, **vals, "counted": True, "note": note, "tv_share": tv}


def buffett_school(company, d, g, ctype, params, price, shares, net_cash, warnings):
    res = {"key": "buffett", "name": "巴菲特流派", "subtitle": "好公司 + 合理价格 + 安全边际",
           "methods": [], "reference": [], "applicable": True, "note": ""}
    res["checks"] = quality_checks(d, ctype)
    res["quality_score"] = _score(res["checks"])
    r, term = params["discount_rate"], params["terminal_growth"]
    last = d[-1]
    eps_norm = _mean([x["eps"] for x in d[-3:]])
    bvps = last.get("bvps")
    rf = _f(params["risk_free_override"]) or _f(company.get("risk_free"))

    if ctype == "financial":
        roe = _mean(_vals(d, "roe"))
        gf = _clamp(g, 0.0, 0.05)
        if roe is not None and bvps and bvps > 0:
            sc = _scen(params)
            vals = {}
            for k, (dg, dr) in sc.items():
                roe_k = roe * {"low": 0.85, "base": 1.0, "high": 1.10}[k]
                rk = _rr(r, dr, gf)
                vals[k] = max(bvps * (roe_k - gf) / (rk - gf), 0.0)
            res["methods"].append({"name": "合理市净率模型 PB=(ROE−g)/(r−g)", **vals, "counted": True,
                                   "note": f"平均 ROE {_pct(roe)}，长期增速 {_pct(gf)}；悲观/乐观情景把 ROE 乘 0.85/1.10"})
        res["reference"].append({"name": "当前市净率", "text": f"{price / bvps:.2f}" if bvps else "数据缺失"})
        res["note"] = "金融股不适用 FCF / DCF，改用 ROE 驱动的合理市净率。"
    else:
        oes = _vals(d, "owner_earnings")[-3:]
        base_oe = _mean(oes)
        gb = _clamp(g, 0.0, params["buffett_growth_cap"])
        if base_oe and base_oe > 0:
            m = _dcf_method("所有者收益 DCF", base_oe, gb, params, net_cash, shares,
                            f"取近 {len(oes)} 年平均所有者收益；增速按保守上限取 {_pct(gb)}", cap_g=params["buffett_growth_cap"])
            if m:
                res["methods"].append(m)
        gn = graham_number(eps_norm, bvps)
        if gn:
            res["reference"].append({"name": "格雷厄姆数（保守底线，不计入估值）", "text": f"{gn:.2f}"})
        if eps_norm and price:
            ey = eps_norm / price
            sp = f"，比无风险利率 {_pct(rf)} 高 {(ey - rf) * 100:.1f} 个百分点" if rf is not None else ""
            res["reference"].append({"name": "盈利收益率（近 3 年平均 EPS ÷ 股价）", "text": _pct(ey) + sp})
    q = res["quality_score"]
    if q is not None and q < 50:
        res["note"] = (res["note"] + " 质量分偏低：便宜可能是价值陷阱，估值结论需打折看待。").strip()
    return _finalize(res, price, params["safety_margin"])


def growth_school(company, d, g, ctype, params, price, shares, net_cash, warnings):
    res = {"key": "growth", "name": "成长股", "subtitle": "增长是否被合理定价",
           "methods": [], "reference": [], "applicable": ctype != "financial", "note": ""}
    res["checks"] = growth_checks(d)
    res["quality_score"] = _score(res["checks"])
    res["reverse_dcf"] = None
    if ctype == "financial":
        res["note"] = "金融股不适用 FCF DCF 与未来 PE 回推。"
        return _finalize(res, price, params["safety_margin"])
    last = d[-1]
    fcf = last.get("fcf")
    eps = _f(company.get("eps_ttm")) or last.get("eps")
    r, term = params["discount_rate"], params["terminal_growth"]

    if fcf and fcf > 0:
        m = _dcf_method("多阶段 DCF（自由现金流）", fcf, g, params, net_cash, shares,
                        f"最新 FCF 起算；前 {params['high_growth_years']} 年增速 {_pct(g)}，之后 {params['fade_years']} 年线性衰减至 {_pct(term)}")
        if m:
            res["methods"].append(m)
            if m["tv_share"] and m["tv_share"] > 0.75:
                warnings.append(f"DCF 终值占总价值 {m['tv_share'] * 100:.0f}%，结论对永续增长和折现率非常敏感。")
        rev = solve_implied_growth(price, fcf, r, term, params["high_growth_years"], params["fade_years"], net_cash, shares)
        rev["assumed"] = g
        res["reverse_dcf"] = rev
        if last.get("net_income") is not None and last["net_income"] <= 0:
            warnings.append("净利润为负但 FCF 为正：FCF 可能因股权激励等非现金费用被高估，DCF 偏乐观。")
    elif fcf is not None:
        res["note"] = "最新自由现金流为负，FCF DCF 不适用。"

    if eps and eps > 0:
        pe_x = params["exit_pe"] or _DEFAULT_EXIT_PE.get(ctype, 20.0)
        sc = _scen(params)
        vals, rets = {}, {}
        for k, (dg, _dr) in sc.items():
            pe_k = pe_x * {"low": 0.8, "base": 1.0, "high": 1.2}[k]
            fut = eps * (1 + g + dg) ** 5 * pe_k
            vals[k] = fut / (1 + params["required_return"]) ** 5
            rets[k] = (fut / price) ** 0.2 - 1 if price > 0 else None
        res["methods"].append({"name": "未来 PE 回推（目标回报定价）", **vals, "counted": True,
                               "note": f"5 年后 EPS × 退出 PE {pe_x:.0f} 倍，按要求回报 {_pct(params['required_return'])} 折回今天"})
        res["expected_return"] = rets
        res["reference"].append({"name": "按当前股价持有 5 年的年化回报（不含分红）",
                                 "text": f"悲观 {_pct(rets['low'])} / 基准 {_pct(rets['base'])} / 乐观 {_pct(rets['high'])}（要求 {_pct(params['required_return'])}）"})
    rev_t = last.get("revenue")
    if rev_t and price and shares:
        ev = price * shares - (net_cash or 0.0)
        evs = ev / rev_t
        res["reference"].append({"name": "EV / 营收", "text": f"{evs:.1f} 倍" + (f"；按增速折算 {evs / (g * 100):.2f}（每 1% 增速对应的倍数）" if g > 0 else "")})
    if last.get("rev_yoy") is not None and last.get("fcf_margin") is not None:
        res["reference"].append({"name": "Rule of 40", "text": _pct(last["rev_yoy"] + last["fcf_margin"])})
    if ctype == "loss" and not res["methods"]:
        res["note"] = "盈利和自由现金流均不为正，盈利/现金流模型不适用；仅提供 EV/营收等参考指标，请结合增速与毛利率自行判断。"
    return _finalize(res, price, params["safety_margin"])


def garp_school(company, d, g, ctype, params, price, shares, net_cash, warnings):
    res = {"key": "garp", "name": "GARP", "subtitle": "增速与估值是否匹配",
           "methods": [], "reference": [], "applicable": ctype != "financial", "note": ""}
    res["checks"] = garp_checks(d, g)
    res["quality_score"] = _score(res["checks"])
    res["in_range"] = 0.08 <= g <= 0.35
    if ctype == "financial":
        res["note"] = "金融股不使用 PEG。"
        return _finalize(res, price, params["safety_margin"])
    eps = _f(company.get("eps_ttm")) or d[-1].get("eps")
    if eps and eps > 0 and g > 0:
        pe = price / eps
        gp = g * 100
        res["peg"] = pe / gp
        res["reference"].append({"name": "市盈率 PE（TTM）", "text": f"{pe:.1f}"})
        res["reference"].append({"name": "PEG = PE ÷ 增速", "text": f"{pe / gp:.2f}（< 1 偏便宜，1–1.5 合理，> 2 偏贵）"})
        feps = _f(company.get("forward_eps"))
        if feps and feps > 0:
            res["reference"].append({"name": "前瞻 PEG（用预期 EPS）", "text": f"{price / feps / gp:.2f}"})
        dy = (_f(company.get("dividend_rate")) or 0.0) / price if price else 0.0
        res["reference"].append({"name": "林奇比率 (增速% + 股息率%) ÷ PE", "text": f"{(gp + dy * 100) / pe:.2f}（> 2 较好，< 1 较差）"})
        sc = _scen(params)
        vals = {}
        for k, (dg, _dr) in sc.items():
            gk = max(g + dg, 0.0) * 100
            vals[k] = eps * _clamp(params["peg_fair"] * gk, 5.0, 40.0)
        res["methods"].append({"name": f"PEG = {params['peg_fair']:.1f} 的合理价", **vals, "counted": True,
                               "note": f"合理 PE = {params['peg_fair']:.1f} × 增速{gp:.1f}（限制在 5–40 倍）× 最新 EPS"})
        if not res["in_range"]:
            res["note"] = "增速在 8%–35% 之外，PEG 容易失真，综合时权重减半。"
    else:
        res["note"] = "EPS 或增速不为正，PEG 不适用。"
    return _finalize(res, price, params["safety_margin"])


# ----------------------------------------------------------------- main
def valuate(company: dict, params: Optional[dict] = None) -> dict:
    p = clean_params(params)
    d = derive(company)
    if not d:
        return {"error": "缺少财务数据（years 为空）"}
    price = p["price_override"] or _f(company.get("price"))
    if not price or price <= 0:
        return {"error": "缺少股价"}
    warnings: list[str] = []
    if company.get("sample"):
        warnings.append("当前为虚构示例数据，仅用于演示，不代表任何真实公司。")
    if len(d) < 4:
        warnings.append(f"只有 {len(d)} 年财务数据，历史指标和增速估计的可靠性有限。")

    shares = _f(company.get("shares")) or d[-1].get("shares")
    if not shares:
        return {"error": "缺少股本数据"}
    net_cash = d[-1].get("net_cash")
    if net_cash is None:
        net_cash = 0.0
        warnings.append("缺少现金或债务数据，DCF 未计入净现金。")

    g, g_src = estimate_growth(company, d, p)
    ctype = classify(company, d, g, p)

    nis = _vals(d, "net_income")
    declines = sum(1 for a, b in zip(nis, nis[1:]) if b < a)
    if len(nis) >= 4 and declines >= 1 and _mean(nis) and _mean(nis) > 0 and statistics.pstdev(nis) / _mean(nis) > 0.5:
        warnings.append("各年净利润波动很大，可能是周期股：不要用当前利润外推，建议改用周期平均盈利。")
    if g_src.startswith("历史"):
        warnings.append("增速来自历史 CAGR（回头看），不是预期；可在左侧填入自己的判断。")

    schools = {
        "buffett": buffett_school(company, d, g, ctype, p, price, shares, net_cash, warnings),
        "growth": growth_school(company, d, g, ctype, p, price, shares, net_cash, warnings),
        "garp": garp_school(company, d, g, ctype, p, price, shares, net_cash, warnings),
    }

    # 综合
    weights = dict(_WEIGHTS[ctype])
    if ctype != "financial" and not schools["garp"].get("in_range", True):
        # 增速低于 8% 时 PEG 合理价失真（被压到 5 倍 PE 下限），直接不计入；高于 35% 则权重减半
        weights["garp"] *= 0.0 if g < 0.08 else 0.5
        if g < 0.08:
            schools["garp"]["note"] = "增速低于 8%，PEG 合理价失真，不计入综合结果。"
            schools["garp"]["excluded"] = True
    used = {k: w for k, w in weights.items() if w > 0 and schools[k]["value"]}
    overall = None
    if used:
        tot = sum(used.values())
        val = {b: sum(schools[k]["value"][b] * w / tot for k, w in used.items()) for b in ("low", "base", "high")}
        verdict, vtext = _verdict(price, val["base"], p["safety_margin"])
        overall = {"value": val, "premium": price / val["base"] - 1, "verdict": verdict, "verdict_text": vtext,
                   "weights": {k: w / tot for k, w in used.items()}}
        bases = [schools[k]["value"]["base"] for k in used]
        if len(bases) >= 2 and _mean(bases) and (max(bases) - min(bases)) / _mean(bases) > 0.5:
            warnings.append("各流派给出的内在价值分歧很大：这家公司是否便宜，取决于你认同哪一种逻辑。")
    else:
        warnings.append("没有任何模型能给出内在价值，请查看各流派的参考指标。")

    # 敏感性分析：优先用成长股 DCF，否则用所有者收益 DCF
    sens = None
    fcf = d[-1].get("fcf")
    base_cf, g_s, label = (fcf, g, "成长股 DCF（FCF）") if fcf and fcf > 0 and ctype != "financial" else (None, g, "")
    if base_cf is None and ctype != "financial":
        oe = _mean(_vals(d, "owner_earnings")[-3:])
        if oe and oe > 0:
            base_cf, g_s, label = oe, _clamp(g, 0.0, p["buffett_growth_cap"]), "所有者收益 DCF"
    if base_cf:
        rates = [max(p["discount_rate"] + x, p["terminal_growth"] + 0.01) for x in (-0.02, -0.01, 0, 0.01, 0.02)]
        growths = [g_s + x for x in (-0.06, -0.03, 0, 0.03, 0.06)]
        grid = []
        for gg in growths:
            row = []
            for rr in rates:
                m = dcf_per_share(base_cf, gg, rr, p["terminal_growth"], p["high_growth_years"], p["fade_years"], net_cash, shares)
                row.append(m["value"] if m else None)
            grid.append(row)
        sens = {"label": label, "rates": rates, "growths": growths, "values": grid, "price": price}

    last = d[-1]
    eps = _f(company.get("eps_ttm")) or last.get("eps")
    summary = {
        "ticker": company.get("ticker"), "name": company.get("name"), "sector": company.get("sector"),
        "currency": company.get("currency", "USD"), "price": price, "shares": shares,
        "market_cap": price * shares, "net_cash": net_cash,
        "pe": price / eps if eps and eps > 0 else None,
        "pb": price / last["bvps"] if last.get("bvps") and last["bvps"] > 0 else None,
        "type": ctype, "type_label": TYPE_LABELS[ctype],
        "growth": g, "growth_source": g_src,
        "source": company.get("source"), "fetched_at": company.get("fetched_at"),
        "sample": bool(company.get("sample")), "risk_free": _f(company.get("risk_free")),
    }
    schools_out = {k: v for k, v in schools.items()}
    return {
        "summary": summary, "schools": schools_out, "overall": overall,
        "reverse_dcf": schools["growth"].get("reverse_dcf"),
        "sensitivity": sens, "history": d, "warnings": warnings, "params": p,
    }
