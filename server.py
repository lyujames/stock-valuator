"""本地服务：python server.py  →  浏览器打开 http://127.0.0.1:8765

只用标准库。默认只监听本机（127.0.0.1），不对局域网开放。
  GET  /                         页面
  GET  /api/samples              示例公司列表
  GET  /api/company?ticker=AAPL  取美股数据（&refresh=1 跳过缓存）
  GET  /api/company?sample=mature 载入示例
  POST /api/valuate              {company, params} → 估值结果
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from valuation import valuate
from valuation.data import DataError, fetch_us, list_samples, load_sample

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
MAX_BODY = 2 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "StockValuator/1.0"

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/api/samples":
                return self._json(200, list_samples())
            if url.path == "/api/company":
                if "sample" in q:
                    return self._json(200, load_sample(q["sample"][0]))
                if "ticker" in q:
                    return self._json(200, fetch_us(q["ticker"][0], refresh=q.get("refresh", ["0"])[0] == "1"))
                return self._json(400, {"error": "缺少 ticker 或 sample 参数"})
            return self._json(404, {"error": "not found"})
        except DataError as e:
            return self._json(502, {"error": str(e)})
        except Exception as e:  # 不把堆栈暴露给页面
            self.log_error("GET %s failed: %r", self.path, e)
            return self._json(500, {"error": "服务内部错误"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/valuate":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json(400, {"error": "请求体为空或过大"})
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            company = body.get("company")
            if not isinstance(company, dict):
                return self._json(400, {"error": "缺少 company"})
            result = valuate(company, body.get("params"))
            return self._json(400 if "error" in result else 200, result)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"error": "请求不是有效的 JSON"})
        except Exception as e:
            self.log_error("POST failed: %r", e)
            return self._json(500, {"error": "计算出错，请检查数据是否完整"})

    def log_message(self, fmt, *args):
        if not os.environ.get("QUIET"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description="美股估值计算器")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"估值计算器已启动：{url}   （Ctrl+C 退出）")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
