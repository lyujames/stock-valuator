import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuation import valuate, clean_params
from valuation.engine import dcf_per_share, solve_implied_growth, graham_number
from valuation.data import list_samples, load_sample, normalize_yf


class TestModels(unittest.TestCase):
    def test_dcf_flat_perpetuity(self):
        # 现金流恒定 100、折现率 10%、无增长 → 价值 = 100 / 0.10 = 1000
        r = dcf_per_share(100, 0.0, 0.10, 0.0, 5, 5, 0.0, 1)
        self.assertAlmostEqual(r["value"], 1000.0, places=6)

    def test_dcf_net_cash_and_shares(self):
        r = dcf_per_share(100, 0.0, 0.10, 0.0, 5, 5, 500.0, 10)
        self.assertAlmostEqual(r["value"], 150.0, places=6)

    def test_dcf_invalid_inputs(self):
        self.assertIsNone(dcf_per_share(-5, 0.05, 0.1, 0.02, 5, 5, 0, 1))
        self.assertIsNone(dcf_per_share(100, 0.05, 0.02, 0.03, 5, 5, 0, 1))  # r <= g_term
        self.assertIsNone(dcf_per_share(100, 0.05, 0.1, 0.02, 5, 5, 0, 0))

    def test_reverse_dcf_roundtrip(self):
        price = dcf_per_share(100, 0.08, 0.10, 0.025, 5, 5, 200.0, 20)["value"]
        s = solve_implied_growth(price, 100, 0.10, 0.025, 5, 5, 200.0, 20)
        self.assertEqual(s["status"], "ok")
        self.assertAlmostEqual(s["growth"], 0.08, places=4)

    def test_reverse_dcf_out_of_range(self):
        self.assertEqual(solve_implied_growth(1e9, 100, 0.10, 0.025, 5, 5, 0, 1)["status"], "above")

    def test_graham_number(self):
        self.assertAlmostEqual(graham_number(5, 30), math.sqrt(22.5 * 5 * 30))
        self.assertIsNone(graham_number(-1, 30))

    def test_clean_params_clamps(self):
        p = clean_params({"discount_rate": 5, "terminal_growth": 0.2, "growth_override": "", "company_type_override": "x"})
        self.assertEqual(p["discount_rate"], 0.25)
        self.assertLess(p["terminal_growth"], p["discount_rate"])
        self.assertIsNone(p["growth_override"])
        self.assertIsNone(p["company_type_override"])
        self.assertEqual(clean_params({"bogus": 1}), clean_params({}))


class TestSamples(unittest.TestCase):
    EXPECT = {"mature": "mature", "garp": "mid_growth", "growth": "high_growth", "loss": "loss", "bank": "financial"}

    def test_types(self):
        for sid, t in self.EXPECT.items():
            r = valuate(load_sample(sid))
            self.assertNotIn("error", r, sid)
            self.assertEqual(r["summary"]["type"], t, sid)

    def test_ordering_and_json(self):
        import json
        for sid in self.EXPECT:
            r = valuate(load_sample(sid))
            json.dumps(r)  # 必须可序列化
            for sc in r["schools"].values():
                if sc["value"]:
                    v = sc["value"]
                    self.assertLessEqual(v["low"], v["base"] + 1e-9, sid)
                    self.assertLessEqual(v["base"], v["high"] + 1e-9, sid)
            if r["overall"]:
                self.assertAlmostEqual(sum(r["overall"]["weights"].values()), 1.0)

    def test_loss_has_no_value(self):
        r = valuate(load_sample("loss"))
        self.assertIsNone(r["overall"])

    def test_price_override_changes_verdict(self):
        c = load_sample("mature")
        cheap = valuate(c, {"price_override": 20})
        self.assertEqual(cheap["overall"]["verdict"], "under")
        dear = valuate(c, {"price_override": 500})
        self.assertEqual(dear["overall"]["verdict"], "over")

    def test_growth_override_monotonic(self):
        c = load_sample("garp")
        lo = valuate(c, {"growth_override": 0.08})["schools"]["growth"]["value"]["base"]
        hi = valuate(c, {"growth_override": 0.25})["schools"]["growth"]["value"]["base"]
        self.assertLess(lo, hi)

    def test_sensitivity_center_matches_base(self):
        r = valuate(load_sample("garp"))
        s = r["sensitivity"]
        dcf_base = r["schools"]["growth"]["methods"][0]["base"]
        self.assertAlmostEqual(s["values"][2][2], dcf_base, places=6)

    def test_errors(self):
        self.assertIn("error", valuate({"years": []}))
        c = load_sample("mature")
        c["price"] = None
        self.assertIn("error", valuate(c))

    def test_samples_listed(self):
        self.assertEqual(len(list_samples()), 5)


class TestYfinanceMapping(unittest.TestCase):
    def test_normalize_with_fake_ticker(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas 未安装")
        cols = [pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")]  # yfinance 习惯：新→旧
        inc = pd.DataFrame({cols[0]: [1000.0, 600.0, 300.0, 200.0, 2.0, 100.0, 100.0],
                            cols[1]: [900.0, 530.0, 260.0, 170.0, 1.7, 90.0, 100.0]},
                           index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS",
                                  "Research And Development", "Diluted Average Shares"])
        cf = pd.DataFrame({cols[0]: [250.0, -60.0, 40.0], cols[1]: [220.0, -55.0, 38.0]},
                          index=["Operating Cash Flow", "Capital Expenditure", "Depreciation And Amortization"])
        bs = pd.DataFrame({cols[0]: [800.0, 300.0, 150.0], cols[1]: [700.0, 320.0, 120.0]},
                          index=["Stockholders Equity", "Total Debt", "Cash Cash Equivalents And Short Term Investments"])

        class Fake:
            info = {"currentPrice": 50.0, "sharesOutstanding": 100.0, "longName": "Fake Inc", "sector": "Technology",
                    "trailingEps": 2.0, "forwardEps": 2.3, "dividendRate": 0.5}
            income_stmt, cashflow, balance_sheet = inc, cf, bs
            growth_estimates = None

        c = normalize_yf(Fake(), "fake", 0.04)
        self.assertEqual([r["year"] for r in c["years"]], [2024, 2025])
        last = c["years"][-1]
        self.assertEqual(last["capex"], 60.0)  # 取绝对值
        self.assertEqual(last["revenue"], 1000.0)
        self.assertEqual(c["price"], 50.0)
        self.assertEqual(c["ticker"], "FAKE")
        r = valuate(c)
        self.assertNotIn("error", r)


if __name__ == "__main__":
    unittest.main()
