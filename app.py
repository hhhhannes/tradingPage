import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from streamlit_autorefresh import st_autorefresh
import zoneinfo

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════
# ── SINGLE SOURCE OF TRUTH: Alle Assets werden NUR hier definiert ───────────
# Um ein neues Asset hinzuzufügen: einfach einen neuen Eintrag anhängen.
# Alles andere (Tabs, Börsenzeiten, Portfolio-Strategie, Benchmarks, …)
# wird automatisch daraus abgeleitet.
#
# Felder:
#   name             Anzeigename für Tab-Titel & Übersicht (Pflicht)
#   strategy_label   Kurzname für die Strategie-Tabelle (optional, Default = name)
#   tz               Zeitzone der Börse, None = 24/7 (z.B. Krypto)
#   open / close     Handelszeiten als (Stunde, Minute)
#   market_name      Anzeigename der Börse (z.B. "NASDAQ", "Krypto 24/7")
#   is_stock         True = Fundamentaldaten (KGV, KBV, …) werden geladen
#   pe_benchmark     Branchen-Benchmark für KGV (optional, nur bei is_stock)
#   pb_benchmark     Branchen-Benchmark für KBV (optional, nur bei is_stock)
#   weight           Ziel-Portfoliogewichtung 0..1, None = nicht im Kernportfolio
#   category         Kategorie für Strategie-Tabelle (nur nötig wenn weight gesetzt)
#   risk             Risikoeinstufung für Strategie-Tabelle
#   goal             Anlageziel für Strategie-Tabelle
# ═══════════════════════════════════════════════════════════════════════════
ASSET_CONFIG = {
    "EURCHF=X": {
        "name": "EUR/CHF Kurs",
        "tz": None, "open": (8, 0), "close": (22,0), "market_name": "Forex",
        "is_stock": False,
        "currency_symbol": "",  # Wechselkurs, kein Dollarpreis
        "weight": None,
    },
    "SLV": {
        "name": "Silber (SLV)",
        "strategy_label": "Silber",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NYSE",
        "is_stock": False,
        "weight": 0.10, "category": "Rohstoff", "risk": "Mittel", "goal": "Inflationsschutz",
    },
    "GLD": {
        "name": "Gold (GLD)",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NYSE",
        "is_stock": False,
        "weight": 0.08, "category": "Sicherheit", "risk": "Niedrig", "goal": "Sicherheit",
    },
    "BTC-USD": {
        "name": "Bitcoin (BTC-USD)",
        "strategy_label": "Bitcoin",
        "tz": None, "open": (0, 0), "close": (23, 59), "market_name": "Krypto 24/7",
        "is_stock": False,
        "weight": 0.12, "category": "Krypto", "risk": "Sehr Hoch", "goal": "Asymm. Upside",
    },
    "URTH": {
        "name": "MSCI World (URTH)",
        "strategy_label": "MSCI World ETF",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NYSE",
        "is_stock": False,
        "weight": 0.30, "category": "Core ETF", "risk": "Niedrig", "goal": "Breite Diversifikation",
    },
    "NVDA": {
        "name": "NVIDIA",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NASDAQ",
        "is_stock": True, "pe_benchmark": 40, "pb_benchmark": 20,
        "weight": 0.15, "category": "KI-Wachstum", "risk": "Hoch", "goal": "KI-Wachstum",
    },
    "AAPL": {
        "name": "Apple",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NASDAQ",
        "is_stock": True, "pe_benchmark": 28, "pb_benchmark": 40,
        "weight": 0.15, "category": "Tech Blue-Chip", "risk": "Mittel", "goal": "Stabile Rendite",
    },
    "BW": {
        "name": "Babcock & Wilcox",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NYSE",
        "is_stock": True, "pe_benchmark": 20, "pb_benchmark": 2,
        "weight": 0.05, "category": "Spezialwert", "risk": "Hoch", "goal": "Nischenwachstum",
    },
    "SPY": {
        "name": "S&P 500 (SPY)",
        "strategy_label": "S&P 500 ETF",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NYSE",
        "is_stock": False,
        "weight": 0.05, "category": "Core ETF", "risk": "Niedrig", "goal": "US-Markt Kern",
    },
    "MSFT": {
        "name": "Microsoft",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NASDAQ",
        "is_stock": True, "pe_benchmark": 32, "pb_benchmark": 12,
        "weight": None,
    },
    "AMZN": {
        "name": "Amazon",
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0), "market_name": "NASDAQ",
        "is_stock": True, "pe_benchmark": 45,
        "weight": None,
    },
    # 👉 Neues Asset hinzufügen? Einfach hier einen weiteren Eintrag ergänzen —
    #    Tabs, Börsenstatus, Fundamentaldaten & Strategie-Tabelle ziehen automatisch nach.
}

# ── Abgeleitete Strukturen (bitte nicht direkt editieren, siehe ASSET_CONFIG) ─
ASSETS = {cfg["name"]: ticker for ticker, cfg in ASSET_CONFIG.items()}

MARKET_INFO = {
    ticker: (cfg["tz"], cfg["open"][0], cfg["open"][1], cfg["close"][0], cfg["close"][1], cfg["market_name"])
    for ticker, cfg in ASSET_CONFIG.items()
}

STOCK_TICKERS = {ticker for ticker, cfg in ASSET_CONFIG.items() if cfg.get("is_stock")}

PORTFOLIO_WEIGHTS = {
    cfg["name"]: cfg["weight"] for cfg in ASSET_CONFIG.values() if cfg.get("weight")
}

PE_BENCHMARKS = {ticker: cfg["pe_benchmark"] for ticker, cfg in ASSET_CONFIG.items() if cfg.get("pe_benchmark")}
PE_BENCHMARKS["default"] = 25

PB_BENCHMARKS = {ticker: cfg["pb_benchmark"] for ticker, cfg in ASSET_CONFIG.items() if cfg.get("pb_benchmark")}
PB_BENCHMARKS["default"] = 5

CURRENCY_SYMBOLS = {ticker: cfg.get("currency_symbol", "$") for ticker, cfg in ASSET_CONFIG.items()}


# Yahoo Finance extended-hours schedule (applies to all timed markets: NYSE/NASDAQ/Forex):
# Pre-Market 04:00–09:30, Regular Session per ASSET_CONFIG, After-Hours close..+4h (until 20:00)
PREMARKET_OPEN    = (4, 0)
AFTERHOURS_LENGTH = timedelta(hours=4)  # regular close + 4h, e.g. 16:00 -> 20:00


def get_market_status(ticker):
    info = MARKET_INFO.get(ticker)
    if not info:
        return "⚪", "Unbekannt", ""
    tz_name, oh, om, ch, cm, market_name = info

    # Crypto is always open
    if tz_name is None:
        return "🟢", "Geöffnet", f"{market_name}"

    tz = zoneinfo.ZoneInfo(tz_name)
    now = datetime.now(tz)
    weekday = now.weekday()  # 0=Mon, 6=Sun

    premarket_dt  = now.replace(hour=PREMARKET_OPEN[0], minute=PREMARKET_OPEN[1], second=0, microsecond=0)
    open_dt       = now.replace(hour=oh, minute=om, second=0, microsecond=0)
    close_dt      = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
    afterhours_dt = close_dt + AFTERHOURS_LENGTH

    if weekday >= 5:  # Weekend — fully closed, next session is Monday pre-market
        days_until_monday = 7 - weekday
        next_premarket = premarket_dt + timedelta(days=days_until_monday)
        local_open = next_premarket.astimezone(zoneinfo.ZoneInfo("Europe/Zurich"))
        return "🔴", "Geschlossen", f"{market_name} · Vorbörse startet Mo {local_open.strftime('%d.%m. %H:%M')} MEZ"

    if premarket_dt <= now < open_dt:
        mins_left = int((open_dt - now).total_seconds() / 60)
        h, m = divmod(mins_left, 60)
        return "🟡", "Vorbörslich", f"{market_name} · reguläre Öffnung in {h}h {m}m"
    elif open_dt <= now <= close_dt:
        mins_left = int((close_dt - now).total_seconds() / 60)
        h, m = divmod(mins_left, 60)
        return "🟢", "Geöffnet", f"{market_name} · schließt in {h}h {m}m"
    elif close_dt < now <= afterhours_dt:
        mins_left = int((afterhours_dt - now).total_seconds() / 60)
        h, m = divmod(mins_left, 60)
        return "🟠", "Nachbörslich", f"{market_name} · Handel endet in {h}h {m}m"
    elif now < premarket_dt:
        # Before today's pre-market start
        mins_until = int((premarket_dt - now).total_seconds() / 60)
        h, m = divmod(mins_until, 60)
        local_open = premarket_dt.astimezone(zoneinfo.ZoneInfo("Europe/Zurich"))
        return "🔴", "Geschlossen", f"{market_name} · Vorbörse startet in {h}h {m}m ({local_open.strftime('%H:%M')} MEZ)"
    else:
        # After after-hours session — find next trading day's pre-market
        next_premarket = premarket_dt + timedelta(days=1)
        while next_premarket.weekday() >= 5:
            next_premarket += timedelta(days=1)
        local_open = next_premarket.astimezone(zoneinfo.ZoneInfo("Europe/Zurich"))
        return "🔴", "Geschlossen", f"{market_name} · Vorbörse startet {local_open.strftime('%d.%m. %H:%M')} MEZ"


# ── Trade Republic (LS Exchange) trading hours ───────────────────────────────
# Trade Republic routes stock/ETF/bond orders through LS Exchange (Lang & Schwarz,
# Hamburg), which runs a single continuous session Mon–Fri 07:30–23:00 CET/CEST —
# no separate pre-/after-market like NYSE. Crypto trades 24/7. Forex pairs (e.g.
# EUR/CHF) are not offered by Trade Republic, so they're excluded here.
TR_TZ    = "Europe/Berlin"
TR_OPEN  = (7, 30)
TR_CLOSE = (23, 0)
TR_NOT_TRADABLE = {"EURCHF=X"}  # brokers/products without TR support


def get_broker_status(ticker):
    """Returns (icon, status, detail) for whether Trade Republic is open for this ticker."""
    if ticker in TR_NOT_TRADABLE:
        return "⚪", "Nicht verfügbar", "Kein Trade Republic Handel für dieses Produkt"

    info = MARKET_INFO.get(ticker)
    if not info:
        return "⚪", "Unbekannt", ""
    tz_name = info[0]

    # Crypto trades 24/7 on Trade Republic too
    if tz_name is None:
        return "🟢", "Geöffnet", "Trade Republic · Krypto 24/7"

    tz = zoneinfo.ZoneInfo(TR_TZ)
    now = datetime.now(tz)
    weekday = now.weekday()

    open_dt  = now.replace(hour=TR_OPEN[0], minute=TR_OPEN[1], second=0, microsecond=0)
    close_dt = now.replace(hour=TR_CLOSE[0], minute=TR_CLOSE[1], second=0, microsecond=0)

    if weekday >= 5:  # Weekend — LS Exchange closed for TR customers
        days_until_monday = 7 - weekday
        next_open = open_dt + timedelta(days=days_until_monday)
        return "🔴", "Geschlossen", f"Trade Republic · öffnet Mo {next_open.strftime('%H:%M')} Uhr"

    if open_dt <= now <= close_dt:
        mins_left = int((close_dt - now).total_seconds() / 60)
        h, m = divmod(mins_left, 60)
        return "🟢", "Geöffnet", f"Trade Republic · schließt in {h}h {m}m"
    elif now < open_dt:
        mins_until = int((open_dt - now).total_seconds() / 60)
        h, m = divmod(mins_until, 60)
        return "🟡", "Vorbörslich", f"Trade Republic · öffnet in {h}h {m}m ({open_dt.strftime('%H:%M')} Uhr)"
    else:
        next_open = open_dt + timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        return "🔴", "Geschlossen", f"Trade Republic · öffnet {next_open.strftime('%d.%m. %H:%M')} Uhr"


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background: #1e2130;
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 10px;
      border-left: 4px solid #4a9eff;
  }
  .buy-card   { border-left-color: #00d084 !important; }
  .hold-card  { border-left-color: #ffa500 !important; }
  .sell-card  { border-left-color: #ff4444 !important; }
  .signal-badge {
      display: inline-block;
      padding: 6px 18px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 1.1rem;
      letter-spacing: 1px;
  }
  .buy-badge  { background:#00d08422; color:#00d084; border:1px solid #00d084; }
  .hold-badge { background:#ffa50022; color:#ffa500; border:1px solid #ffa500; }
  .sell-badge { background:#ff444422; color:#ff4444; border:1px solid #ff4444; }
  h1 { color: #ffffff; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      background: #1e2130;
      border-radius: 8px;
      color: #aaa;
      padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
      background: #4a9eff22;
      color: #4a9eff;
      border-bottom: 2px solid #4a9eff;
  }
</style>
""", unsafe_allow_html=True)

# ── Helper: Technical Indicators ──────────────────────────────────────────────
def compute_indicators(df):
    close = df["Close"].squeeze()

    # Moving averages
    df["SMA20"]  = close.rolling(20).mean().values
    df["SMA50"]  = close.rolling(50).mean().values
    df["SMA200"] = close.rolling(200).mean().values
    df["EMA12"]  = close.ewm(span=12).mean().values
    df["EMA26"]  = close.ewm(span=26).mean().values

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    df["MACD"]        = macd_line.values
    df["MACD_signal"] = macd_signal.values
    df["MACD_hist"]   = (macd_line - macd_signal).values

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).values

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_mid"]   = bb_mid.values
    df["BB_upper"] = bb_upper.values
    df["BB_lower"] = bb_lower.values
    df["BB_pct"]   = ((close - bb_lower) / (bb_upper - bb_lower)).values

    # ATR (Volatility)
    high = df["High"].squeeze(); low_ = df["Low"].squeeze()
    tr = pd.concat([
        high - low_,
        (high - close.shift()).abs(),
        (low_ - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean().values

    # Volume trend
    if "Volume" in df.columns:
        vol = df["Volume"].squeeze()
        vol_sma = vol.rolling(20).mean()
        df["Vol_SMA20"] = vol_sma.values
        df["Vol_ratio"] = (vol / vol_sma.replace(0, np.nan)).values
    else:
        df["Vol_ratio"] = 1.0

    return df

# ── Helper: Fundamental Scoring ───────────────────────────────────────────────
def score_fundamentals(info, ticker):
    """Returns (fund_score -30..+30, fund_reasons, fund_details)"""
    fscore  = 0
    freasons = []
    fdetails = {}

    # ─ KGV (P/E) ─────────────────────────────────────────────────────────────
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe and pe > 0:
        benchmark_pe = PE_BENCHMARKS.get(ticker, PE_BENCHMARKS["default"])
        fdetails["KGV (P/E)"] = f"{pe:.1f} (Bench: {benchmark_pe})"
        if pe < benchmark_pe * 0.7:
            fscore += 12; freasons.append(f"KGV {pe:.1f} deutlich unter Benchmark ({benchmark_pe}) → günstig bewertet")
        elif pe < benchmark_pe:
            fscore += 6;  freasons.append(f"KGV {pe:.1f} unter Benchmark ({benchmark_pe}) → fair bewertet")
        elif pe > benchmark_pe * 2:
            fscore -= 12; freasons.append(f"KGV {pe:.1f} stark überteuert (Bench: {benchmark_pe})")
        elif pe > benchmark_pe * 1.3:
            fscore -= 6;  freasons.append(f"KGV {pe:.1f} leicht überteuert (Bench: {benchmark_pe})")

    # ─ KBV (P/B) ─────────────────────────────────────────────────────────────
    pb = info.get("priceToBook")
    if pb and pb > 0:
        benchmark_pb = PB_BENCHMARKS.get(ticker, PB_BENCHMARKS["default"])
        fdetails["KBV (P/B)"] = f"{pb:.2f} (Bench: {benchmark_pb})"
        if pb < 1.0:
            fscore += 8;  freasons.append(f"KBV {pb:.2f} unter Buchwert → potentiell unterbewertet")
        elif pb < benchmark_pb:
            fscore += 4;  freasons.append(f"KBV {pb:.2f} unter Benchmark ({benchmark_pb})")
        elif pb > benchmark_pb * 2:
            fscore -= 8;  freasons.append(f"KBV {pb:.2f} deutlich über Benchmark → teuer")

    # ─ PEG-Ratio ─────────────────────────────────────────────────────────────
    peg = info.get("pegRatio")
    if peg and peg > 0:
        fdetails["PEG"] = f"{peg:.2f}"
        if peg < 1.0:
            fscore += 8;  freasons.append(f"PEG {peg:.2f} < 1 → Wachstum relativ günstig")
        elif peg < 2.0:
            fscore += 3;  freasons.append(f"PEG {peg:.2f} akzeptabel")
        else:
            fscore -= 5;  freasons.append(f"PEG {peg:.2f} > 2 → Wachstum teuer eingepreist")

    # ─ Gewinnmarge ───────────────────────────────────────────────────────────
    margin = info.get("profitMargins")
    if margin:
        fdetails["Nettomarge"] = f"{margin*100:.1f}%"
        if margin > 0.20:
            fscore += 6;  freasons.append(f"Hohe Nettomarge {margin*100:.1f}% → starkes Geschäftsmodell")
        elif margin < 0:
            fscore -= 8;  freasons.append(f"Negative Marge {margin*100:.1f}% → Verlustunternehmen")

    # ─ Umsatzwachstum ────────────────────────────────────────────────────────
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        fdetails["Umsatzwachstum"] = f"{rev_growth*100:+.1f}%"
        if rev_growth > 0.20:
            fscore += 6;  freasons.append(f"Starkes Umsatzwachstum {rev_growth*100:.1f}%")
        elif rev_growth > 0.05:
            fscore += 3;  freasons.append(f"Moderates Umsatzwachstum {rev_growth*100:.1f}%")
        elif rev_growth < 0:
            fscore -= 6;  freasons.append(f"Umsatzrückgang {rev_growth*100:.1f}%")

    # ─ Schulden/Eigenkapital ─────────────────────────────────────────────────
    de = info.get("debtToEquity")
    if de is not None:
        fdetails["Verschuldung (D/E)"] = f"{de:.1f}%"
        if de < 50:
            fscore += 4;  freasons.append(f"Niedrige Verschuldung D/E {de:.1f}%")
        elif de > 200:
            fscore -= 6;  freasons.append(f"Hohe Verschuldung D/E {de:.1f}%")

    # ─ Dividendenrendite ─────────────────────────────────────────────────────
    div = info.get("dividendYield")
    if div and div > 0:
        fdetails["Dividende"] = f"{div*100:.2f}%"
        if div > 0.03:
            fscore += 3;  freasons.append(f"Attraktive Dividendenrendite {div*100:.2f}%")

    fscore = max(-30, min(30, fscore))
    return fscore, freasons, fdetails

# ── Über-/Unterverkauft Bewertung ─────────────────────────────────────────────
def get_valuation_label(rsi, bb_pct, pct_from_high, score):
    """Returns (label, color, description)"""
    ob_signals = 0
    os_signals = 0

    if rsi > 70: ob_signals += 2
    elif rsi > 60: ob_signals += 1
    if rsi < 30: os_signals += 2
    elif rsi < 40: os_signals += 1

    if bb_pct > 0.85: ob_signals += 2
    elif bb_pct > 0.70: ob_signals += 1
    if bb_pct < 0.15: os_signals += 2
    elif bb_pct < 0.30: os_signals += 1

    if pct_from_high > -5: ob_signals += 1
    if pct_from_high < -30: os_signals += 2
    elif pct_from_high < -15: os_signals += 1

    if   ob_signals >= 4: return "🔴 Stark überkauft",    "#ff4444", "Mehrere Indikatoren zeigen extreme Überhitzung. Rücksetzer wahrscheinlich."
    elif ob_signals >= 2: return "🟠 Leicht überkauft",   "#ff8800", "Kurs an oberem Bereich. Vorsicht bei Neueinstiegen."
    elif os_signals >= 4: return "🟢 Stark überverkauft", "#00d084", "Kurs deutlich unter fairen Wert gedrückt. Rebound-Potenzial hoch."
    elif os_signals >= 2: return "🟡 Leicht überverkauft","#aacc00", "Kurs leicht unterbewertet. Einstiegsgelegenheit möglich."
    else:                 return "⚪ Neutral bewertet",   "#aaaaaa", "Kurs im fairen Bereich. Kein extremes Signal."

# ── Helper: Signal Engine ─────────────────────────────────────────────────────
def generate_signal(df, ticker, info=None):
    if df is None or len(df) < 50:
        return "HALTEN", 50, {}, [], "⚪ Neutral bewertet", "#aaaaaa", "", {}, []

    df   = compute_indicators(df.copy())
    last = df.iloc[-1]
    close_val = float(df["Close"].squeeze().iloc[-1])

    score    = 50  # neutral
    reasons  = []
    details  = {}

    # ─ RSI ───────────────────────────────────────────────────────────────────
    rsi = float(last["RSI"])
    details["RSI"] = round(rsi, 1)
    if rsi < 30:
        score += 20; reasons.append(f"RSI überverkauft ({rsi:.0f}) → bullisch")
    elif rsi < 45:
        score += 10; reasons.append(f"RSI leicht überverkauft ({rsi:.0f})")
    elif rsi > 75:
        score -= 20; reasons.append(f"RSI überkauft ({rsi:.0f}) → bearisch")
    elif rsi > 60:
        score -= 8;  reasons.append(f"RSI erhöht ({rsi:.0f})")

    # ─ MACD ──────────────────────────────────────────────────────────────────
    macd      = float(last["MACD"])
    macd_sig  = float(last["MACD_signal"])
    macd_hist = float(last["MACD_hist"])
    details["MACD"] = round(macd, 4)
    if macd > macd_sig and macd_hist > 0:
        score += 15; reasons.append("MACD über Signal → bullisches Momentum")
    elif macd < macd_sig and macd_hist < 0:
        score -= 15; reasons.append("MACD unter Signal → bearisches Momentum")

    # ─ Moving Averages ───────────────────────────────────────────────────────
    sma20  = float(last["SMA20"])  if not np.isnan(last["SMA20"])  else close_val
    sma50  = float(last["SMA50"])  if not np.isnan(last["SMA50"])  else close_val
    sma200 = float(last["SMA200"]) if not np.isnan(last["SMA200"]) else close_val
    details["SMA20"]  = round(sma20, 2)
    details["SMA50"]  = round(sma50, 2)
    details["SMA200"] = round(sma200, 2)

    if close_val > sma20 > sma50 > sma200:
        score += 15; reasons.append("Kurs über SMA20/50/200 (Golden Trend)")
    elif close_val > sma50:
        score += 8;  reasons.append("Kurs über SMA50")
    elif close_val < sma200:
        score -= 12; reasons.append("Kurs unter SMA200 (Abwärtstrend)")

    # ─ Bollinger Bands ───────────────────────────────────────────────────────
    bb_pct = float(last["BB_pct"])
    details["BB%"] = round(bb_pct * 100, 1)
    if bb_pct < 0.1:
        score += 12; reasons.append("Nahe unterem Bollinger Band → Rebound möglich")
    elif bb_pct > 0.9:
        score -= 10; reasons.append("Nahe oberem Bollinger Band → Überhitzt")

    # ─ Volume ────────────────────────────────────────────────────────────────
    vol_ratio = float(last["Vol_ratio"]) if not np.isnan(last["Vol_ratio"]) else 1.0
    details["Vol_ratio"] = round(vol_ratio, 2)
    if vol_ratio > 1.5 and close_val > sma20:
        score += 5;  reasons.append("Überdurchschnittliches Volumen mit Aufwärtsbewegung")

    # ─ 52W High/Low ──────────────────────────────────────────────────────────
    close_series = df["Close"].squeeze()
    high52 = float(close_series.rolling(252).max().iloc[-1]) if len(df) >= 252 else float(close_series.max())
    low52  = float(close_series.rolling(252).min().iloc[-1]) if len(df) >= 252 else float(close_series.min())
    pct_from_high = (close_val - high52) / high52 * 100
    details["52W_High"]      = round(high52, 2)
    details["52W_Low"]       = round(low52, 2)
    details["% v. 52W_High"] = round(pct_from_high, 1)

    # ─ Fundamentaldaten in Score einbeziehen ─────────────────────────────────
    fund_score = 0
    fund_reasons = []
    fund_details = {}
    if info:
        fund_score, fund_reasons, fund_details = score_fundamentals(info, ticker)
        score += fund_score

    # ─ Über-/Unterverkauft Label ─────────────────────────────────────────────
    val_label, val_color, val_desc = get_valuation_label(rsi, bb_pct, pct_from_high, score)

    # ─ Clamp & decide ────────────────────────────────────────────────────────
    score = max(0, min(100, score))

    if score >= 62:
        signal = "KAUFEN"
    elif score <= 40:
        signal = "VERKAUFEN"
    else:
        signal = "HALTEN"

    return signal, score, details, reasons, val_label, val_color, val_desc, fund_details, fund_reasons

# ── Data Fetch ────────────────────────────────────────────────────────────────
def fetch_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        # yfinance >= 0.2.x returns MultiIndex columns like ("Close", "AAPL")
        # flatten them to simple column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def fetch_info(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info
    except:
        return {}

# ── Portfolio Strategy ────────────────────────────────────────────────────────
def render_strategy():
    st.subheader("📋 Empfohlene Portfoliostrategie")

    st.markdown("""
    <div style='background:#1e2130;border-radius:12px;padding:20px;margin-bottom:20px;border-left:4px solid #4a9eff'>
    <b style='color:#4a9eff'>Strategieprinzip:</b> Kern-Satelliten-Ansatz<br>
    <span style='color:#aaa'>Stabiler ETF-Kern (60%) + Wachstumstitel + Rohstoff-Hedge + Krypto-Spekulation</span>
    </div>
    """, unsafe_allow_html=True)

    # Wird direkt aus ASSET_CONFIG abgeleitet: nur Assets mit gesetztem "weight"
    # tauchen im Kernportfolio auf, sortiert nach Gewichtung absteigend.
    portfolio_items = sorted(
        [(t, c) for t, c in ASSET_CONFIG.items() if c.get("weight")],
        key=lambda x: x[1]["weight"], reverse=True
    )
    strategy = {
        "Asset":          [c.get("strategy_label", c["name"]) for _, c in portfolio_items],
        "Ticker":         [t for t, _ in portfolio_items],
        "Kategorie":      [c["category"] for _, c in portfolio_items],
        "Gewichtung (%)": [round(c["weight"] * 100) for _, c in portfolio_items],
        "Risiko":         [c["risk"] for _, c in portfolio_items],
        "Ziel":           [c["goal"] for _, c in portfolio_items],
    }
    df_strat = pd.DataFrame(strategy)

    fig = px.pie(
        df_strat,
        values="Gewichtung (%)",
        names="Asset",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.45
    )
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", legend_font_size=12,
        margin=dict(t=30, b=10)
    )
    fig.update_traces(textfont_size=13)

    col1, col2 = st.columns([1.1, 1.4])
    with col1:
        st.plotly_chart(fig, use_container_width=True, key="strategy_pie")
    with col2:
        st.dataframe(
            df_strat[["Asset", "Gewichtung (%)", "Kategorie", "Risiko", "Ziel"]],
            use_container_width=True, hide_index=True
        )

    st.markdown("""
    ---
    ### 💡 Regeln für dein Portfolio

    | Regel | Detail |
    |-------|--------|
    | **Rebalancing** | Alle 3–6 Monate zurück auf Zielgewichtung |
    | **Bitcoin-Limit** | Nie über 15% – zu hohe Volatilität |
    | **Rohstoffquote** | Silber + Gold = 18% als Inflationsschutz |
    | **Spekulative Titel** | BW + NVDA max. 20% zusammen (Kurskorrektur-Risiko) |
    | **Stop-Loss Idee** | Einzeltitel bei -20% vom Einstandskurs überdenken |
    | **DCA-Strategie** | Monatlich gleiche Beträge → Durchschnittskosteneffekt |
    """)

    st.markdown("---")
    st.markdown("### 📖 Erklärungen & Berechnungsmethoden")

    with st.expander("🧮 Wie wird der Bull/Bear Score berechnet?"):
        st.markdown("""
**Startpunkt: 50 Punkte** (neutral). Jeder Indikator addiert oder subtrahiert Punkte. Endergebnis wird auf 0–100 begrenzt.

| Indikator | Bedingung | Punkte |
|-----------|-----------|--------|
| **RSI** | < 30 (überverkauft) | **+20** |
| **RSI** | 30–45 (leicht tief) | **+10** |
| **RSI** | 60–75 (leicht hoch) | **−8** |
| **RSI** | > 75 (überkauft) | **−20** |
| **MACD** | MACD > Signal & Histogramm > 0 | **+15** |
| **MACD** | MACD < Signal & Histogramm < 0 | **−15** |
| **Trend** | Kurs > SMA20 > SMA50 > SMA200 (Golden Trend) | **+15** |
| **Trend** | Kurs > SMA50 | **+8** |
| **Trend** | Kurs < SMA200 | **−12** |
| **Bollinger** | Kurs nahe unterem Band (BB% < 10%) | **+12** |
| **Bollinger** | Kurs nahe oberem Band (BB% > 90%) | **−10** |
| **Volumen** | Volumen > 1.5× Schnitt bei steigendem Kurs | **+5** |
| **Fundamental** | KGV, KBV, PEG, Marge, Wachstum … | **−30 bis +30** |

**Entscheidung:**
- 🟢 Score **≥ 62** → KAUFEN
- 🟡 Score **41–61** → HALTEN
- 🔴 Score **≤ 40** → VERKAUFEN
        """)

    with st.expander("📊 Wie wird Über-/Unterverkauft berechnet?"):
        st.markdown("""
3 unabhängige Quellen liefern Punkte für "überkauft" (OB) oder "überverkauft" (OS):

| Signal | Bedingung | Wertung |
|--------|-----------|---------|
| **RSI > 70** | Stark überkauft | +2 OB |
| **RSI 60–70** | Leicht überkauft | +1 OB |
| **RSI < 30** | Stark überverkauft | +2 OS |
| **RSI 30–40** | Leicht überverkauft | +1 OS |
| **BB% > 85%** | Nahe oberem Bollinger Band | +2 OB |
| **BB% 70–85%** | Kurs erhöht | +1 OB |
| **BB% < 15%** | Nahe unterem Bollinger Band | +2 OS |
| **BB% 15–30%** | Kurs gedrückt | +1 OS |
| **< 5% unter 52W-Hoch** | Nahe Jahreshoch | +1 OB |
| **> 30% unter 52W-Hoch** | Weit vom Jahreshoch | +2 OS |
| **15–30% unter 52W-Hoch** | Deutlich korrigiert | +1 OS |

| OB-Punkte | OS-Punkte | Urteil |
|-----------|-----------|--------|
| ≥ 4 | – | 🔴 Stark überkauft |
| ≥ 2 | – | 🟠 Leicht überkauft |
| – | ≥ 4 | 🟢 Stark überverkauft |
| – | ≥ 2 | 🟡 Leicht überverkauft |
| < 2 | < 2 | ⚪ Neutral bewertet |

> **Hinweis:** Überverkauft ≠ sofortiger Kursanstieg. Es zeigt, dass der Kurs statistisch ungewöhnlich tief ist und zur Mitte tendiert.
        """)

    with st.expander("📐 Technische Kennzahlen erklärt"):
        st.markdown("""
| Kennzahl | Berechnung | Interpretation |
|----------|-----------|----------------|
| **RSI** | Relative Strength Index über 14 Tage. Verhältnis von Kursgewinnen zu -verlusten. | < 30 = überverkauft · > 70 = überkauft |
| **MACD** | Differenz aus 12-Tage-EMA und 26-Tage-EMA. Signal = 9-Tage-EMA des MACD. | MACD > Signal = bullisch · darunter = bearisch |
| **SMA20/50/200** | Einfacher gleitender Durchschnitt über 20, 50, 200 Handelstage. | Kurs über allen drei = Golden Trend |
| **BB%** | Position des Kurses innerhalb der Bollinger Bänder (±2 Standardabweichungen). | 0% = unteres Band · 100% = oberes Band |
| **Vol_ratio** | Heutiges Volumen ÷ 20-Tage-Durchschnittsvolumen. | > 1.5 bei steigendem Kurs = Bestätigung |
| **52W High/Low** | Höchst-/Tiefstkurs der letzten 252 Handelstage. | Abstand vom Jahreshoch zeigt Erholungspotenzial |
        """)

    with st.expander("📊 Fundamentalkennzahlen erklärt"):
        st.markdown("""
| Kennzahl | Berechnung | Interpretation |
|----------|-----------|----------------|
| **KGV (P/E)** | Aktienkurs ÷ Gewinn je Aktie (letzte 12 Monate). | Unter Benchmark = günstig · über 2× Benchmark = teuer |
| **Fwd. KGV** | Kurs ÷ erwarteter Gewinn (nächste 12 Monate). | Zeigt ob Wachstum bereits eingepreist ist |
| **KBV (P/B)** | Kurs ÷ Buchwert je Aktie. | < 1 = unter Buchwert (potentiell unterbewertet) |
| **PEG** | KGV ÷ jährliches Gewinnwachstum (%). | < 1 = Wachstum günstig · > 2 = zu teuer |
| **EPS (TTM)** | Gewinn je Aktie der letzten 12 Monate. | Positiv = profitabel · negativ = Verlust |
| **Nettomarge** | Jahresüberschuss ÷ Umsatz × 100. | > 20% = starkes Geschäftsmodell |
| **Umsatzwachstum** | (Umsatz aktuell − Vorjahr) ÷ Vorjahr × 100. | > 20% = stark · negativ = Schrumpfung |
| **D/E Ratio** | Gesamtschulden ÷ Eigenkapital × 100. | < 50% = solide · > 200% = hoch verschuldet |
| **Dividende** | Jährliche Dividende ÷ Kurs × 100. | Attraktiv ab ~3% |
| **Beta** | Volatilität relativ zum S&P 500. | < 1 = defensiv · > 1 = aggressiver als Markt |
| **Fund.-Score** | Gewichtete Summe aller Fundamental-Signale (−30 bis +30). | Positiv = fundamental bullisch |
        """)

    with st.expander("🕒 Beste Handelszeit bei Trade Republic"):
        st.markdown("""
Trade Republic wickelt Aktien-/ETF-Orders über die LS Exchange ab, die börsentäglich von 07:30 bis 23:00 Uhr (MEZ/MESZ) geöffnet ist. Innerhalb dieses Fensters ist die Liquidität aber sehr unterschiedlich — je näher an den Öffnungszeiten der großen Referenzbörsen (Xetra, NYSE/NASDAQ), desto engere Spreads.

| Zeitfenster | Was passiert | Empfehlung |
|-------------|--------------|------------|
| **07:30–09:00** | LS Exchange offen, Xetra noch zu | ⚠️ Meiden – dünne Liquidität, breite Spreads |
| **09:00–17:30** | Xetra-Kernzeit | ✅ Beste Zeit für europäische Aktien/ETFs |
| **15:30–17:30** | Xetra **und** US-Börsen gleichzeitig offen (15:30 MEZ = 9:30 ET) | ✅✅ Bestes Fenster für US-Titel (NVDA, AAPL, MSFT, AMZN, BW) |
| **17:30–22:00** | Xetra zu, US-Börse noch offen | 🟡 Ok für US-Titel, Spread etwas breiter |
| **20:00–23:00** | Auch US-Börse zu (Nachbörse endet ~20 Uhr MEZ) | 🔴 Meiden – kaum Liquidität |
| **Krypto** | 24/7 | Zeitpunkt spielt keine Rolle |

**Für dieses Portfolio bedeutet das konkret:**
- **NVDA, AAPL, MSFT, AMZN, BW** (US-Titel) → am besten **15:30–17:30 Uhr MEZ**
- **SLV, GLD, URTH, SPY** (in Europa gut handelbare ETFs) → **09:00–17:30 Uhr** reicht
- **BTC-USD** → jederzeit handelbar
- Vor 9:00 und nach 17:30 Uhr (besonders nach 20 Uhr) sind Spreads spürbar breiter — das ist ein unsichtbarer Aufschlag, auch wenn Trade Republic technisch "geöffnet" hat.
        """)


# ── Single Asset Analysis ─────────────────────────────────────────────────────
def render_asset(name, ticker):
    st.subheader(f"📊 {name} ({ticker})")

    period = st.session_state.get("global_period", "1y")
    df = fetch_data(ticker, period)

    if df is None or df.empty:
        st.warning(f"Keine Daten für {ticker} verfügbar.")
        return

    # Fundamentals only for tickers flagged as stocks in ASSET_CONFIG
    info = {}
    if ticker in STOCK_TICKERS:
        info = fetch_info(ticker)

    df = compute_indicators(df.copy())
    close_series = df["Close"].squeeze()
    signal, score, details, reasons, val_label, val_color, val_desc, fund_details, fund_reasons = generate_signal(df, ticker, info)

    current_price = float(close_series.iloc[-1])
    prev_price    = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
    change_pct    = (current_price - prev_price) / prev_price * 100

    badge_class = {"KAUFEN": "buy", "HALTEN": "hold", "VERKAUFEN": "sell"}[signal]
    emoji       = {"KAUFEN": "🟢", "HALTEN": "🟡", "VERKAUFEN": "🔴"}[signal]

    # ── Yahoo Finance Link + Market Status + Trade Republic Status ───────────
    yahoo_url = f"https://finance.yahoo.com/quote/{ticker}"
    mkt_icon, mkt_status, mkt_detail = get_market_status(ticker)
    tr_icon, tr_status, tr_detail = get_broker_status(ticker)
    col_link, col_mkt, col_tr = st.columns([1, 1.6, 1.6])
    with col_link:
        st.markdown(f"🔗 [Auf Yahoo Finance öffnen]({yahoo_url})")

    def render_status_badge(icon, status, detail):
        color = {"🟢": "#00d084", "🟡": "#ffa500", "🟠": "#ff8800", "🔴": "#ff4444", "⚪": "#888"}.get(icon, "#888")
        st.markdown(
            f"<div style='padding:4px 12px;border-radius:8px;background:#1e2130;display:inline-block;"
            f"border-left:3px solid {color}'>"
            f"<span style='color:{color};font-weight:700'>{icon} {status}</span>"
            f"<span style='color:#aaa;font-size:.85rem'> &nbsp;{detail}</span></div>",
            unsafe_allow_html=True
        )

    with col_mkt:
        render_status_badge(mkt_icon, mkt_status, mkt_detail)
    with col_tr:
        render_status_badge(tr_icon, tr_status, tr_detail)


    # ── Top Row ───────────────────────────────────────────────────────────────
    col_sig, col_score, col_val, col_price, col_chg = st.columns(5)
    with col_sig:
        st.markdown(f"""
        <div class='metric-card {badge_class}-card'>
            <div style='color:#aaa;font-size:.8rem'>SIGNAL</div>
            <span class='signal-badge {badge_class}-badge'>{emoji} {signal}</span>
        </div>""", unsafe_allow_html=True)
    with col_score:
        color = "#00d084" if score >= 62 else ("#ff4444" if score <= 40 else "#ffa500")
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#aaa;font-size:.8rem'>SCORE</div>
            <div style='font-size:2rem;font-weight:700;color:{color}'>{score}<span style='font-size:1rem'>/100</span></div>
        </div>""", unsafe_allow_html=True)
    with col_val:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#aaa;font-size:.8rem'>BEWERTUNG</div>
            <div style='font-size:.95rem;font-weight:700;color:{val_color};margin-top:4px'>{val_label}</div>
        </div>""", unsafe_allow_html=True)
    with col_price:
        sym = CURRENCY_SYMBOLS.get(ticker, "$")
        price_str = f"{sym}{current_price:,.4f}" if sym == "" else f"{sym}{current_price:,.2f}"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#aaa;font-size:.8rem'>KURS</div>
            <div style='font-size:1.8rem;font-weight:700;color:#fff'>{price_str}</div>
        </div>""", unsafe_allow_html=True)
    with col_chg:
        chg_color = "#00d084" if change_pct >= 0 else "#ff4444"
        chg_arrow = "▲" if change_pct >= 0 else "▼"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#aaa;font-size:.8rem'>TAGESVERÄND.</div>
            <div style='font-size:1.8rem;font-weight:700;color:{chg_color}'>{chg_arrow} {change_pct:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Bewertungs-Banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:#1e2130;border-radius:10px;padding:12px 18px;margin:10px 0 6px 0;border-left:4px solid {val_color}'>
        <b style='color:{val_color}'>{val_label}</b> &nbsp;—&nbsp;
        <span style='color:#ccc'>{val_desc}</span>
    </div>""", unsafe_allow_html=True)

    # ── Expanders: Signalbegründungen ─────────────────────────────────────────
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if reasons:
            with st.expander("📐 Technische Signalbegründung"):
                for r in reasons:
                    icon = "🟢" if any(w in r for w in ["bullisch","überverkauft","Golden","Aufwärts","Rebound","über"]) else "🔴"
                    st.markdown(f"{icon} {r}")
    with col_exp2:
        if fund_reasons:
            with st.expander("📊 Fundamentale Signalbegründung"):
                for r in fund_reasons:
                    icon = "🟢" if any(w in r for w in ["günstig","fair","stark","attraktiv","niedrig","Wachstum"]) else "🔴"
                    st.markdown(f"{icon} {r}")

    # ── Gauge + Chart ─────────────────────────────────────────────────────────
    col_chart, col_gauge = st.columns([2.5, 1])
    with col_gauge:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Bull/Bear Score", "font": {"color": "white", "size": 13}},
            number={"font": {"color": "white", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#aaa"},
                "bar":  {"color": "#4a9eff"},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0, 40],   "color": "rgba(255,68,68,0.2)"},
                    {"range": [40, 62],  "color": "rgba(255,165,0,0.2)"},
                    {"range": [62, 100], "color": "rgba(0,208,132,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": score}
            }
        ))
        fig_g.update_layout(paper_bgcolor="#0e1117", font_color="white",
                            height=220, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True, key=f"gauge_{ticker}")

    with col_chart:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.20, 0.15], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(), close=close_series,
            increasing_line_color="#00d084", decreasing_line_color="#ff4444", name="Kurs"
        ), row=1, col=1)
        for col_n, clr, dash, lbl in [
            ("BB_upper","#4a9eff","dot","BB Oben"),
            ("BB_mid","#ffa500","dash","SMA20"),
            ("BB_lower","#4a9eff","dot","BB Unten"),
        ]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_n],
                line=dict(color=clr, width=1, dash=dash), name=lbl, opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"],  line=dict(color="#ff9900", width=1.5), name="SMA50"),  row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], line=dict(color="#cc44ff", width=1.5), name="SMA200"), row=1, col=1)
        colors_hist = ["#00d084" if v >= 0 else "#ff4444" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=colors_hist, name="MACD Hist", opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        line=dict(color="#4a9eff", width=1.5), name="MACD"),   row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#ffa500", width=1.5), name="Signal"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#cc44ff", width=2), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff4444", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00d084", opacity=0.5, row=3, col=1)
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
                          font_color="white", height=720, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.02, font_size=11),
                          margin=dict(t=20, b=10, l=10, r=10))
        fig.update_yaxes(gridcolor="#2a2d3e", zerolinecolor="#2a2d3e")
        fig.update_xaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig, use_container_width=True, key=f"price_{ticker}")

    # ── Technische Kennzahlen ─────────────────────────────────────────────────
    st.markdown("#### 📐 Technische Kennzahlen")
    kz_cols = st.columns(len(details))
    for i, (k, v) in enumerate(details.items()):
        with kz_cols[i % len(kz_cols)]:
            st.metric(k, v)

    # ── Fundamentaldaten ──────────────────────────────────────────────────────
    if info:
        st.markdown("#### 📊 Fundamentaldaten")
        fs, _, _ = score_fundamentals(info, ticker)
        raw_fund = {
            "Marktkapital.":   f"${info.get('marketCap',0)/1e9:.1f}B"        if info.get("marketCap") else "–",
            "KGV (P/E)":       f"{info.get('trailingPE',0):.1f}"             if info.get("trailingPE") else "–",
            "Fwd. KGV":        f"{info.get('forwardPE',0):.1f}"              if info.get("forwardPE") else "–",
            "KBV (P/B)":       f"{info.get('priceToBook',0):.2f}"            if info.get("priceToBook") else "–",
            "PEG":             f"{info.get('pegRatio',0):.2f}"               if info.get("pegRatio") else "–",
            "EPS (TTM)":       f"${info.get('trailingEps',0):.2f}"           if info.get("trailingEps") else "–",
            "Nettomarge":      f"{info.get('profitMargins',0)*100:.1f}%"     if info.get("profitMargins") else "–",
            "Umsatzwachstum":  f"{info.get('revenueGrowth',0)*100:+.1f}%"   if info.get("revenueGrowth") is not None else "–",
            "D/E Ratio":       f"{info.get('debtToEquity',0):.1f}%"         if info.get("debtToEquity") else "–",
            "Dividende":       f"{info.get('dividendYield',0)*100:.2f}%"     if info.get("dividendYield") else "0%",
            "Beta":            f"{info.get('beta',0):.2f}"                   if info.get("beta") else "–",
            "Fund.-Score":     f"{fs:+d}/30",
        }
        f_cols = st.columns(6)
        for i, (k, v) in enumerate(raw_fund.items()):
            with f_cols[i % 6]:
                st.metric(k, v)

# ── Portfolio Overview ────────────────────────────────────────────────────────
def render_overview():
    st.subheader("Übersicht")

    results = []
    progress = st.progress(0, text="Daten werden geladen …")

    assets_to_check = ASSETS  # gleiche Assets wie in der Tab-Navigation
    total = len(assets_to_check)

    for i, (name, ticker) in enumerate(assets_to_check.items()):
        progress.progress((i + 1) / total, text=f"Lade {name} …")
        df   = fetch_data(ticker, st.session_state.get("global_period", "1y"))
        info = fetch_info(ticker) if ticker in STOCK_TICKERS else {}
        sig, score, details, _reasons, val_label, _vc, _vd, _fd, _fr = generate_signal(df, ticker, info)

        close_series = df["Close"].squeeze() if df is not None else None
        price  = float(close_series.iloc[-1]) if close_series is not None else 0
        chg1d  = float((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100) if close_series is not None and len(close_series) > 1 else 0
        chg1m  = float((close_series.iloc[-1] - close_series.iloc[-21]) / close_series.iloc[-21] * 100) if close_series is not None and len(close_series) > 21 else 0
        chg3m  = float((close_series.iloc[-1] - close_series.iloc[-63]) / close_series.iloc[-63] * 100) if close_series is not None and len(close_series) > 63 else 0

        mkt_icon_ov, mkt_status_ov, _ = get_market_status(ticker)
        results.append({
            "Asset":         name,
            "Ticker":        ticker,
            "Börse":         f"{mkt_icon_ov} {mkt_status_ov}",
            "Kurs":          round(price, 4) if ticker in CURRENCY_SYMBOLS and CURRENCY_SYMBOLS[ticker] == "" else round(price, 2),
            "1T (%)":        round(chg1d, 2),
            "1M (%)":        round(chg1m, 2),
            "3M (%)":        round(chg3m, 2),
            "Bewertung":     val_label,
            "Signal":        sig,
            "Score":         score,
        })

    progress.empty()
    df_res = pd.DataFrame(results)

    # Color mapping for Signal
    def signal_color(val):
        colors = {"KAUFEN": "background-color:#00d08433;color:#00d084",
                  "HALTEN": "background-color:#ffa50033;color:#ffa500",
                  "VERKAUFEN": "background-color:#ff444433;color:#ff4444"}
        return colors.get(val, "")

    def chg_color(val):
        if isinstance(val, (int, float)):
            return f"color:{'#00d084' if val >= 0 else '#ff4444'}"
        return ""

    def score_color(val):
        if val >= 62:
            return "background-color:#00d08433;color:#00d084;font-weight:700"
        elif val <= 40:
            return "background-color:#ff444433;color:#ff4444;font-weight:700"
        return "background-color:#ffa50033;color:#ffa500;font-weight:700"

    def val_color_fn(val):
        if "überverkauft" in val or "Stark überverkauft" in val:
            return "color:#00d084"
        if "Leicht überverkauft" in val:
            return "color:#aacc00"
        if "Stark überkauft" in val:
            return "color:#ff4444"
        if "Leicht überkauft" in val:
            return "color:#ff8800"
        return "color:#aaaaaa"

    _map = "map" if hasattr(df_res.style, "map") else "applymap"
    styled = df_res.style
    styled = getattr(styled, _map)(signal_color, subset=["Signal"])
    styled = getattr(styled, _map)(score_color,  subset=["Score"])
    styled = getattr(styled, _map)(chg_color,    subset=["1T (%)", "1M (%)", "3M (%)"])
    styled = getattr(styled, _map)(val_color_fn,  subset=["Bewertung"])
    styled = styled.format({
        "Kurs":    lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:,.2f}",
        "1T (%)":  "{:+.2f}", "1M (%)": "{:+.2f}", "3M (%)": "{:+.2f}",
        "Score":   "{:.0f}/100",
    })

    # Höhe automatisch an Zeilenanzahl anpassen, damit keine Scrollleiste nötig ist
    table_height = 38 * (len(df_res) + 1) + 3
    st.dataframe(styled, use_container_width=True, hide_index=True, height=table_height)

    # Signal summary
    buys  = len(df_res[df_res["Signal"] == "KAUFEN"])
    holds = len(df_res[df_res["Signal"] == "HALTEN"])
    sells = len(df_res[df_res["Signal"] == "VERKAUFEN"])
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 KAUFEN",    buys)
    c2.metric("🟡 HALTEN",    holds)
    c3.metric("🔴 VERKAUFEN", sells)

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Auto-refresh alle 5 Minuten (300000 ms)
    st_autorefresh(interval=300_000, key="data_autorefresh")

    # ── Sidebar: globale Einstellungen ────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")
        period_options = ["3mo", "6mo", "1y", "2y", "5y"]
        period_labels  = ["3 Monate", "6 Monate", "1 Jahr ✅", "2 Jahre", "5 Jahre"]
        selected = st.selectbox(
            "📅 Zeitraum (global)",
            options=period_options,
            format_func=lambda x: period_labels[period_options.index(x)],
            index=2,
            key="global_period"
        )
        st.caption("Gilt für alle Assets und die Übersicht.")
        st.info("**Empfehlung:** 1 Jahr\nAlle Indikatoren inkl. SMA200 vollständig verfügbar.")

    st.markdown("# 📈 Investment Dashboard")
    st.markdown("*Technische Analyse · Kauf/Halten/Verkaufen Signale · Portfoliostrategie*")
    st.divider()

    tabs = st.tabs(["🗂️ Übersicht", "📋 Strategie"] + [f"📊 {name}" for name in ASSETS.keys()])

    with tabs[0]:
        render_overview()

    with tabs[1]:
        render_strategy()

    for i, (name, ticker) in enumerate(ASSETS.items()):
        with tabs[i + 2]:
            render_asset(name, ticker)

    st.divider()
    st.components.v1.html("""
    <p style='font-size:0.8rem;color:#888;font-family:sans-serif'>
        ⚠️ Nur zu Informationszwecken. Kein Anlageberatungsersatz.
        Letzte Aktualisierung: <span id='ts'></span>
    </p>
    <script>
        document.getElementById('ts').innerText = new Date().toLocaleString('de-DE');
    </script>
    """, height=30)

if __name__ == "__main__":
    main()