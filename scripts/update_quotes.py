"""Holt aktuelle Kursdaten (Yahoo Finance) und schreibt sie nach quotes.js.

Ohne externe Abhängigkeiten (nur Standardbibliothek).
Aufruf:  python scripts/update_quotes.py
Die Webseite lädt quotes.js per <script>, funktioniert also auch lokal per file://.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Ticker -> Anzeigename. "group" verbindet den Kurs mit der Zeile im Fahrplan (index.html).
TICKERS = [
    ("AMZN", "Amazon", "AMZN"),
    ("AAPL", "Apple", "AAPL"),
    ("NVDA", "NVIDIA", "NVDA"),
    ("AMD", "AMD", "AMD/ASML"),
    ("ASML", "ASML", "AMD/ASML"),
    ("WMT", "Walmart", "WMT"),
    ("COST", "Costco", "COST"),
    ("DE", "Deere", "DE"),
    ("IWM", "Russell 2000 ETF", "IWM"),
    ("NEM", "Newmont", "NEM/GOLD"),
    ("B", "Barrick (ex GOLD)", "NEM/GOLD"),
    ("NTR", "Nutrien", "NTR/CF"),
    ("CF", "CF Industries", "NTR/CF"),
    ("BKNG", "Booking", "BKNG/DAL"),
    ("DAL", "Delta Air Lines", "BKNG/DAL"),
    ("HD", "Home Depot", "HD/LOW"),
    ("LOW", "Lowe's", "HD/LOW"),
    ("XOM", "ExxonMobil", "Öl/VLO"),
    ("CVX", "Chevron", "Öl/VLO"),
    ("VLO", "Valero", "Öl/VLO"),
    ("XLV", "Health Care ETF", "Defensiv"),
    ("XLP", "Basiskonsum ETF", "Defensiv"),
    ("NOVN.SW", "Novartis", "Defensiv"),
    ("NESN.SW", "Nestlé", "Defensiv"),
    ("RO.SW", "Roche", "Defensiv"),
    ("URTH", "MSCI World ETF", "URTH"),
    ("TSLA", "Tesla", "Watchlist"),
    ("ANT", "Anthropic", "Watchlist"),
    ("ZAL.DE", "Zalando", "Watchlist"),
    ("^GSPC", "S&P 500", "Markt"),
]

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=1y&interval=1d"
OUT = Path(__file__).resolve().parent.parent / "quotes.js"


def fetch(symbol):
    req = urllib.request.Request(URL.format(urllib.request.quote(symbol)),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    res = data["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = [(t, h, l, c) for t, h, l, c in
            zip(res["timestamp"], q["high"], q["low"], q["close"]) if c is not None]
    # Bei ganz neuen Titeln (z.B. kurz nach dem Börsengang) liefert Yahoo teils
    # mehrere Punkte pro Tag – auf einen Schlusskurs pro Handelstag reduzieren.
    off = res["meta"].get("gmtoffset", 0)
    daily = {}
    for r in rows:
        daily[(r[0] + off) // 86400] = r
    return res["meta"], [daily[k] for k in sorted(daily)]


def pct(a, b):
    return None if not b else round((a / b - 1) * 100, 2)


def close_before(rows, days):
    """Schlusskurs am letzten Handelstag, der mindestens `days` Kalendertage zurückliegt."""
    cutoff = rows[-1][0] - days * 86400
    older = [r for r in rows if r[0] <= cutoff]
    return older[-1][3] if older else None


def sma(closes, n):
    return round(sum(closes[-n:]) / n, 4) if len(closes) >= n else None


def atr(rows, n=14):
    if len(rows) < n + 1:
        return None
    trs = []
    for prev, cur in zip(rows[-n - 1:-1], rows[-n:]):
        _, h, l, _c = cur
        pc = prev[3]
        if h is None or l is None:
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return round(sum(trs) / len(trs), 4) if trs else None


def build(symbol, name, group):
    meta, rows = fetch(symbol)
    closes = [r[3] for r in rows]
    price = meta.get("regularMarketPrice") or closes[-1]
    # Letzte Zeile = aktueller bzw. letzter Handelstag, also ist closes[-2] der Vortag
    prev = closes[-2] if len(closes) >= 2 else meta.get("chartPreviousClose")
    year = datetime.fromtimestamp(rows[-1][0], timezone.utc).year
    first_of_year = next((r[3] for r in rows
                          if datetime.fromtimestamp(r[0], timezone.utc).year == year), None)
    return {
        "symbol": symbol,
        "name": name,
        "group": group,
        "currency": meta.get("currency"),
        "price": round(price, 4),
        "time": meta.get("regularMarketTime"),
        "day": pct(price, prev),
        "week": pct(price, close_before(rows, 7)),
        "month": pct(price, close_before(rows, 30)),
        "ytd": pct(price, first_of_year),
        "sma20": sma(closes, 20),
        "sma50": sma(closes, 50),
        "sma200": sma(closes, 200),
        "atr14": atr(rows),
        "high52": meta.get("fiftyTwoWeekHigh"),
        "low52": meta.get("fiftyTwoWeekLow"),
        "spark": [round(c, 4) for c in closes[-30:]],
    }


def main():
    out, failed = [], []
    for sym, name, group in TICKERS:
        for attempt in range(3):  # bis zu 3 Versuche bei Verbindungsfehlern
            try:
                out.append(build(sym, name, group))
                break
            except Exception as e:  # einzelne Ausfälle sollen nicht alles stoppen
                if attempt == 2:
                    failed.append(sym)
                    print(f"FEHLER {sym}: {e}", file=sys.stderr)
                else:
                    time.sleep(2)
        time.sleep(0.3)
    if not out:
        sys.exit("Keine Kursdaten erhalten – quotes.js bleibt unverändert.")
    payload = {"updated": int(time.time()), "failed": failed, "quotes": out}
    OUT.write_text("// automatisch erzeugt von scripts/update_quotes.py – nicht von Hand ändern\n"
                   "window.QUOTES = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"{len(out)} Kurse geschrieben nach {OUT.name}" + (f", fehlgeschlagen: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
