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

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# ── Asset Universe ────────────────────────────────────────────────────────────
ASSETS = {
    "Silber (SLV)":           "SLV",
    "NVIDIA":                 "NVDA",
    "MSCI World (URTH)":      "URTH",
    "Bitcoin (BTC-USD)":      "BTC-USD",
    "Apple":                  "AAPL",
    "Babcock & Wilcox":       "BW",
    "S&P 500 (SPY)":          "SPY",
    "Gold (GLD)":             "GLD",
    "Microsoft":              "MSFT",
    "Amazon":                 "AMZN",
}

PORTFOLIO_WEIGHTS = {
    "MSCI World (URTH)":  0.30,
    "Apple":              0.15,
    "NVIDIA":             0.15,
    "Bitcoin (BTC-USD)":  0.12,
    "Silber (SLV)":       0.10,
    "Gold (GLD)":         0.08,
    "Babcock & Wilcox":   0.05,
    "S&P 500 (SPY)":      0.05,
}

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
# Benchmarks per sector/asset type
PE_BENCHMARKS  = {"NVDA": 40, "AAPL": 28, "MSFT": 32, "AMZN": 45, "BW": 20, "default": 25}
PB_BENCHMARKS  = {"NVDA": 20, "AAPL": 40, "MSFT": 12, "BW": 2,    "default": 5}

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
@st.cache_data(ttl=900)
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

@st.cache_data(ttl=900)
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

    strategy = {
        "Asset":           ["MSCI World ETF", "Apple", "NVIDIA", "Bitcoin", "Silber", "Gold", "Babcock & Wilcox", "S&P 500 ETF"],
        "Ticker":          ["URTH", "AAPL", "NVDA", "BTC-USD", "SLV", "GLD", "BW", "SPY"],
        "Kategorie":       ["Core ETF", "Tech Blue-Chip", "KI-Wachstum", "Krypto", "Rohstoff", "Sicherheit", "Spezialwert", "Core ETF"],
        "Gewichtung (%)":  [30, 15, 15, 12, 10, 8, 5, 5],
        "Risiko":          ["Niedrig", "Mittel", "Hoch", "Sehr Hoch", "Mittel", "Niedrig", "Hoch", "Niedrig"],
        "Ziel":            ["Breite Diversifikation", "Stabile Rendite", "KI-Wachstum", "Asymm. Upside", "Inflationsschutz", "Sicherheit", "Nischenwachstum", "US-Markt Kern"],
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

# ── Single Asset Analysis ─────────────────────────────────────────────────────
def render_asset(name, ticker):
    st.subheader(f"📊 {name} ({ticker})")

    period = st.session_state.get("global_period", "1y")
    df = fetch_data(ticker, period)

    if df is None or df.empty:
        st.warning(f"Keine Daten für {ticker} verfügbar.")
        return

    # Fundamentals for stocks only
    info = {}
    if ticker not in ["SLV", "GLD", "URTH", "SPY", "BTC-USD"]:
        info = fetch_info(ticker)

    df = compute_indicators(df.copy())
    close_series = df["Close"].squeeze()
    signal, score, details, reasons, val_label, val_color, val_desc, fund_details, fund_reasons = generate_signal(df, ticker, info)

    current_price = float(close_series.iloc[-1])
    prev_price    = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
    change_pct    = (current_price - prev_price) / prev_price * 100

    badge_class = {"KAUFEN": "buy", "HALTEN": "hold", "VERKAUFEN": "sell"}[signal]
    emoji       = {"KAUFEN": "🟢", "HALTEN": "🟡", "VERKAUFEN": "🔴"}[signal]

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
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:#aaa;font-size:.8rem'>KURS</div>
            <div style='font-size:1.8rem;font-weight:700;color:#fff'>${current_price:,.2f}</div>
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
                            row_heights=[0.55, 0.25, 0.2], vertical_spacing=0.04)
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
                          font_color="white", height=520, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.02, font_size=11),
                          margin=dict(t=20, b=10, l=10, r=10))
        fig.update_yaxes(gridcolor="#2a2d3e", zerolinecolor="#2a2d3e")
        fig.update_xaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig, use_container_width=True, key=f"price_{ticker}")

    # ── Technische Kennzahlen ─────────────────────────────────────────────────
    st.markdown("#### 📐 Technische Kennzahlen")
    with st.expander("ℹ️ Wie werden die technischen Kennzahlen berechnet?"):
        st.markdown("""
| Kennzahl | Berechnung | Interpretation |
|----------|-----------|----------------|
| **RSI** | Relative Strength Index über 14 Tage. Verhältnis von durchschnittlichen Kursgewinnen zu -verlusten. | < 30 = überverkauft (Kaufsignal) · > 70 = überkauft (Verkaufssignal) |
| **MACD** | Differenz aus 12-Tage-EMA und 26-Tage-EMA. Signal-Linie = 9-Tage-EMA des MACD. | MACD > Signal = bullisch · MACD < Signal = bearisch |
| **SMA20/50/200** | Einfacher gleitender Durchschnitt über 20, 50 bzw. 200 Handelstage. | Kurs über allen drei = starker Aufwärtstrend (Golden Trend) |
| **BB%** | Position des Kurses innerhalb der Bollinger Bänder (±2 Standardabweichungen vom SMA20). | 0% = unteres Band (überverkauft) · 100% = oberes Band (überkauft) |
| **Vol_ratio** | Heutiges Volumen ÷ 20-Tage-Durchschnittsvolumen. | > 1.5 bei steigendem Kurs = starkes Kaufsignal (Bestätigung) |
| **52W High/Low** | Höchst- und Tiefstkurs der letzten 252 Handelstage (~1 Jahr). | % vom 52W-Hoch zeigt wie weit der Kurs vom Jahreshoch entfernt ist |
| **Bull/Bear Score** | Gewichtete Summe aller technischen + fundamentalen Signale (0–100). | ≥ 62 = Kaufen · 41–61 = Halten · ≤ 40 = Verkaufen |
        """)

    with st.expander("🧮 Wie genau wird der Bull/Bear Score berechnet?"):
        st.markdown("""
**Startpunkt: 50 Punkte** (neutral). Jeder Indikator addiert oder subtrahiert Punkte:

| Indikator | Bedingung | Punkte |
|-----------|-----------|--------|
| **RSI** | < 30 (überverkauft) | **+20** |
| **RSI** | 30–45 (leicht tief) | **+10** |
| **RSI** | 60–75 (leicht hoch) | **−8** |
| **RSI** | > 75 (überkauft) | **−20** |
| **MACD** | MACD > Signal & Histogramm > 0 | **+15** |
| **MACD** | MACD < Signal & Histogramm < 0 | **−15** |
| **Trend** | Kurs > SMA20 > SMA50 > SMA200 | **+15** |
| **Trend** | Kurs > SMA50 | **+8** |
| **Trend** | Kurs < SMA200 | **−12** |
| **Bollinger** | Kurs nahe unterem Band (BB% < 10%) | **+12** |
| **Bollinger** | Kurs nahe oberem Band (BB% > 90%) | **−10** |
| **Volumen** | Volumen > 1.5× Schnitt bei steigendem Kurs | **+5** |
| **Fundamental** | KGV, KBV, PEG, Marge, Wachstum … | **−30 bis +30** |

**Endergebnis wird auf 0–100 begrenzt:**
- 🟢 **≥ 62** → KAUFEN
- 🟡 **41–61** → HALTEN  
- 🔴 **≤ 40** → VERKAUFEN
        """)

    with st.expander("📊 Wie wird Über-/Unterverkauft berechnet?"):
        st.markdown("""
Die Bewertung kombiniert **3 unabhängige Signalquellen** zu einem Gesamturteil.
Jede Quelle kann 1 oder 2 Punkte für "überkauft" (OB) oder "überverkauft" (OS) liefern:

| Signal | Bedingung | Wertung |
|--------|-----------|---------|
| **RSI > 70** | Stark überkauft | +2 OB-Punkte |
| **RSI 60–70** | Leicht überkauft | +1 OB-Punkt |
| **RSI < 30** | Stark überverkauft | +2 OS-Punkte |
| **RSI 30–40** | Leicht überverkauft | +1 OS-Punkt |
| **BB% > 85%** | Kurs nahe oberem Bollinger Band | +2 OB-Punkte |
| **BB% 70–85%** | Kurs erhöht | +1 OB-Punkt |
| **BB% < 15%** | Kurs nahe unterem Bollinger Band | +2 OS-Punkte |
| **BB% 15–30%** | Kurs gedrückt | +1 OS-Punkt |
| **< 5% unter 52W-Hoch** | Nahe Jahreshoch | +1 OB-Punkt |
| **> 30% unter 52W-Hoch** | Weit vom Jahreshoch | +2 OS-Punkte |
| **15–30% unter 52W-Hoch** | Deutlich korrigiert | +1 OS-Punkt |

**Auswertung der Gesamtpunkte:**

| OB-Punkte | OS-Punkte | Urteil |
|-----------|-----------|--------|
| ≥ 4 | – | 🔴 Stark überkauft |
| ≥ 2 | – | 🟠 Leicht überkauft |
| – | ≥ 4 | 🟢 Stark überverkauft |
| – | ≥ 2 | 🟡 Leicht überverkauft |
| < 2 | < 2 | ⚪ Neutral bewertet |

> **Hinweis:** Überverkauft bedeutet nicht zwingend sofortiger Kursanstieg – es zeigt, dass der Kurs im Vergleich zu seiner jüngsten Entwicklung ungewöhnlich tief ist und statistisch zur Mitte tendiert.
        """)
    kz_cols = st.columns(len(details))
    for i, (k, v) in enumerate(details.items()):
        with kz_cols[i % len(kz_cols)]:
            st.metric(k, v)

    # ── Fundamentaldaten ──────────────────────────────────────────────────────
    if info:
        st.markdown("#### 📊 Fundamentaldaten")
        with st.expander("ℹ️ Wie werden die Fundamentaldaten berechnet?"):
            st.markdown("""
| Kennzahl | Berechnung | Interpretation |
|----------|-----------|----------------|
| **KGV (P/E)** | Aktienkurs ÷ Gewinn je Aktie (letzte 12 Monate). | < Sektorbenchmark = günstig · > 2× Benchmark = teuer |
| **Fwd. KGV** | Kurs ÷ erwarteter Gewinn je Aktie (nächste 12 Monate). | Zeigt ob Wachstum bereits eingepreist ist |
| **KBV (P/B)** | Kurs ÷ Buchwert je Aktie (Eigenkapital ÷ Aktienanzahl). | < 1 = unter Buchwert (potentiell unterbewertet) |
| **PEG** | KGV ÷ jährliches Gewinnwachstum (%). | < 1 = Wachstum günstig · > 2 = Wachstum teuer eingepreist |
| **EPS (TTM)** | Gewinn je Aktie der letzten 12 Monate (Trailing Twelve Months). | Positiv = profitabel · negativ = Verlustunternehmen |
| **Nettomarge** | Jahresüberschuss ÷ Umsatz × 100. | > 20% = sehr starkes Geschäftsmodell |
| **Umsatzwachstum** | (Umsatz aktuell − Umsatz Vorjahr) ÷ Umsatz Vorjahr × 100. | > 20% = starkes Wachstum · negativ = Schrumpfung |
| **D/E Ratio** | Gesamtschulden ÷ Eigenkapital × 100. | < 50% = solide · > 200% = hoch verschuldet |
| **Dividende** | Jährliche Dividende je Aktie ÷ Aktienkurs × 100. | Attraktiv ab ~3% – nur bei stabilen Unternehmen relevant |
| **Beta** | Kursvolatilität relativ zum Gesamtmarkt (S&P 500). | < 1 = defensiv · > 1 = volatiler als Markt · > 2 = sehr spekulativ |
| **Fund.-Score** | Gewichtete Summe aller Fundamental-Signale (−30 bis +30). | Wird zum technischen Score addiert · positiv = fundamental bullisch |
            """)
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
    st.subheader("🗂️ Portfolio Übersicht – Alle Signale")

    results = []
    progress = st.progress(0, text="Daten werden geladen …")

    assets_to_check = {k: v for k, v in ASSETS.items() if k in PORTFOLIO_WEIGHTS}
    total = len(assets_to_check)

    for i, (name, ticker) in enumerate(assets_to_check.items()):
        progress.progress((i + 1) / total, text=f"Lade {name} …")
        df   = fetch_data(ticker, st.session_state.get("global_period", "1y"))
        sig, score, details, _reasons, val_label, _vc, _vd, _fd, _fr = generate_signal(df, ticker)

        close_series = df["Close"].squeeze() if df is not None else None
        price  = float(close_series.iloc[-1]) if close_series is not None else 0
        chg1d  = float((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100) if close_series is not None and len(close_series) > 1 else 0
        chg1m  = float((close_series.iloc[-1] - close_series.iloc[-21]) / close_series.iloc[-21] * 100) if close_series is not None and len(close_series) > 21 else 0
        chg3m  = float((close_series.iloc[-1] - close_series.iloc[-63]) / close_series.iloc[-63] * 100) if close_series is not None and len(close_series) > 63 else 0

        results.append({
            "Asset":       name,
            "Ticker":      ticker,
            "Gewicht (%)": int(PORTFOLIO_WEIGHTS[name] * 100),
            "Kurs ($)":    round(price, 2),
            "1T (%)":      round(chg1d, 2),
            "1M (%)":      round(chg1m, 2),
            "3M (%)":      round(chg3m, 2),
            "RSI":         details.get("RSI", "–"),
            "Score":       score,
            "Signal":      sig,
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

    _map = "map" if hasattr(df_res.style, "map") else "applymap"
    styled = df_res.style
    styled = getattr(styled, _map)(signal_color, subset=["Signal"])
    styled = getattr(styled, _map)(chg_color,    subset=["1T (%)", "1M (%)", "3M (%)"])
    styled = styled.format({"Kurs ($)": "{:.2f}", "1T (%)": "{:+.2f}", "1M (%)": "{:+.2f}", "3M (%)": "{:+.2f}"})

    st.dataframe(styled, use_container_width=True, hide_index=True)

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
    st.caption(f"⚠️ Nur zu Informationszwecken. Kein Anlageberatungsersatz. Letzte Aktualisierung: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

if __name__ == "__main__":
    main()