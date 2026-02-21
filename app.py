import streamlit as st
import requests
import pandas as pd
import feedparser
import time
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, UTC
from google import genai

# --- KONFIGURATION & INITIALISIERUNG ---
st.set_page_config(page_title="Gold Intelligence Terminal", layout="wide", page_icon="🤖")

# --- KEY ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Streamlit Secrets
#GEMINI_API_KEY = "" # lokal
client = genai.Client(api_key=GEMINI_API_KEY)

# --- FUNKTIONEN ---

def get_exact_model_name():
    """Findet den exakten Namen für Gemini 3 Flash in deinem Account."""
    try:
        for m in client.models.list():
            # Suche nach gemini-3 und flash im Namen
            if "gemini-3" in m.name.lower() and "flash" in m.name.lower():
                return m.name
        # Fallback auf Standard, falls Liste scheitert
        return "gemini-2.0-flash" 
    except Exception:
        return "gemini-1.5-flash"

# --- FUNKTION: YAHOO SCRAPER ---
def scrape_yahoo_gold_page():
    url = "https://finance.yahoo.com/quote/GC=F/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {"error": f"Yahoo blockiert (Status {response.status_code})", "price": "N/A", "news": []}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice", "data-symbol": "GC=F"})
        current_price = price_tag.text if price_tag else "Nicht gefunden"

        news_list = []
        for h3 in soup.find_all('h3'):
            title = h3.get_text(strip=True)
            if len(title) > 20: 
                news_list.append(title)
        
        return {"timestamp": datetime.now().strftime('%d.%m. %H:%M'), "price": current_price, "news": news_list}
    except Exception as e:
        return {"error": str(e), "price": "Fehler", "news": []}

# --- FUNKTION: FINANZEN.CH RSS ---
def get_finanzen_ch_news_data():
    rss_url = "https://www.finanzen.ch/rss/news"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(rss_url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        news_items = []
        for entry in feed.entries:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                date_str = time.strftime('%d.%m.%y | %H:%M', entry.published_parsed)
            else:
                date_str = "Heute"
            
            soup = BeautifulSoup(entry.get('description', ''), 'html.parser')
            clean_desc = soup.get_text(separator=" ").strip()
            
            news_items.append({
                "date": date_str,
                "title": entry.get('title', 'Kein Titel').upper(),
                "desc": clean_desc,
                "link": entry.get('link', '#')
            })
        return news_items
    except:
        return []
    
    
# --- FUNKTION: TRADINGVIEW KALENDER ---
def get_gold_calendar():
    url = "https://economic-calendar.tradingview.com/events"
    today = datetime.now(UTC)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/"
    }
    params = {
        "from": today.strftime('%Y-%m-%dT00:00:00.000Z'),
        "to": (today + timedelta(days=7)).strftime('%Y-%m-%dT23:59:59.000Z'),
        "countries": "US,CN,EU,DE,IN,GB,JP",
        "importance": "1" 
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        events = response.json().get('result', [])
        return [{
            "Zeit (UTC)": datetime.fromisoformat(e['date'].replace('Z', '+00:00')).strftime('%d.%m. %H:%M'),
            "Land": e.get('country', '??'),
            "Event": e.get('title', 'Kein Titel'),
            "Actual": e.get('actual', '-'),
            "Forecast": e.get('forecast', '-'),
            "Previous": e.get('previous', '-')
        } for e in events]
    except:
        return []

# --- course data ---
def get_course_data():
    # Ticker abrufen
    tickers = ["GC=F", "DX-Y.NYB", "^VIX"]
    # Wir nehmen 5 Tage mit 1h Intervall
    data = yf.download(tickers, period="5d", interval="1h", group_by='ticker')

    df = pd.DataFrame()

    # Daten extrahieren und Lücken füllen
    # ffill() füllt NaN-Werte mit dem letzten bekannten Kurs auf
    df['Gold'] = data['GC=F']['Close'].ffill()
    df['USD_Index'] = data['DX-Y.NYB']['Close'].ffill()
    df['VIX'] = data['^VIX']['Close'].ffill()

    # Wichtige KI-Features berechnen
    df['Gold_Ret_%'] = df['Gold'].pct_change() * 100
    df['VIX_Change'] = df['VIX'].diff()

    # Nur Zeilen behalten, wo wir für alle drei Ticker Daten haben
    df.dropna(inplace=True)
    df_rounded = df.round(3)
    return df_rounded


# --- UI ---
st.title("🤖 Gold AI Intelligence Terminal")

if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    with st.spinner("Sammle Daten..."):

        news_yahoo = scrape_yahoo_gold_page()
        news_finanzen_ch= get_finanzen_ch_news_data()
        calendar_output = get_gold_calendar()
        course_data = get_course_data()

    with st.spinner("KI-Analyse..."):
        # Dynamische Modellfindung
        model_name = get_exact_model_name()

        prompt = f"""
        Rolle: Handle als erfahrener Finanzanalyst mit Spezialisierung auf Edelmetalle und Makroökonomie.

        Aufgabe: Erstelle eine fundierte Goldpreis-Prognose für die Zeiträume 1 Tag, 1 Woche und 3 Wochen.

        Datenbasis:
        1. Yahoo News: {news_yahoo}
        2. Finanzen.ch News: {news_finanzen_ch}
        3. Wirtschaftskalender: {calendar_output}
        4. Aktuelle Kursdaten (Gold, USD_Index, VIX): {course_data}

        Analyse-Schritte:
        Schritt 1: Analysiere die News-Quellen (1 & 2). Klassifiziere jede Meldung als [BULLISCH], [BÄRISCH] oder [NEUTRAL]. Vergib einen Relevanz-Score von 1-10.
        Schritt 2: Werte den Wirtschaftskalender (3) aus. Welche Ereignisse (z.B. Fed-Sitzung, Inflation) sind "High-Impact" für Gold?
        Schritt 3: Kombiniere die News-Stimmung mit den Kursdaten (4). Achte auf Divergenzen (z.B. steigender VIX bei fallendem Gold).

        Ausgabeformat:
        - Kurze Zusammenfassung der aktuellen Marktstimmung.
        - Eine Tabelle der wichtigsten Einflussfaktoren mit Sentiment und Gewichtung.
        - Prognose-Fazit:
        * 1 Tag: [Richtung + Begründung]
        * 1 Woche: [Richtung + Begründung]
        * 3 Wochen: [Richtung + Begründung]
        - Disclaimer: Weise darauf hin, dass dies keine Anlageberatung ist.
        """

        # Gemini request
        if True:
            try:
                response = client.models.generate_content(model=model_name.replace("models/", ""), contents=prompt)
                
                # Anzeige der Analyse
                st.success("Analyse abgeschlossen!")
                st.markdown("### Gemini KI Markt-Einschätzung")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"KI Fehler: {e}")

        st.divider()
        st.write("### Rohdaten") 
        st.write("Yahoo:") 
        st.write(news_yahoo) 
        st.write("Finanzen.ch:") 
        st.write(news_finanzen_ch) 
        st.write("Wirtschaftskalender:") 
        st.write(calendar_output) 
        st.write("Kursdaten:") 
        st.write(course_data) 

    st.divider()
    

else:
    st.info("Klicke auf den Button oben, um die Datenquellen abzufragen und die Gemini-Analyse zu starten.")