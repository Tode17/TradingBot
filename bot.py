import os
import json
import time
import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import anthropic
import yfinance as yf

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY   = os.getenv("CLAUDE_API_KEY")

claude_client  = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
WATCHLIST_FILE = "watchlist.txt"
ALERTED_FILE   = "alerted.json"

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        print(f"[Telegram] Message envoye")
    except Exception as e:
        print(f"[Telegram] Erreur: {e}")

# ============================================================
# WATCHLIST
# ============================================================
def load_watchlist() -> list:
    try:
        with open(WATCHLIST_FILE, "r") as f:
            tickers = [
                line.strip().upper()
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]
        print(f"[Watchlist] {len(tickers)} tickers charges")
        return tickers
    except:
        print(f"[Watchlist] Fichier introuvable !")
        send_telegram("⚠️ Fichier watchlist.txt introuvable !")
        return []

# ============================================================
# ALERTES
# ============================================================
def load_alerted() -> dict:
    try:
        with open(ALERTED_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_alerted(alerted: dict):
    with open(ALERTED_FILE, "w") as f:
        json.dump(alerted, f)

def reset_alerted():
    with open(ALERTED_FILE, "w") as f:
        json.dump({}, f)
    print("[Alertes] Remise a zero")

# ============================================================
# DONNÉES YAHOO FINANCE — PRIX
# ============================================================
def get_bars(symbol: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df     = ticker.history(period="3mo", interval="1d")
        if df is None or len(df) < 20:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 20 else None
    except:
        return None

# ============================================================
# DONNÉES YAHOO FINANCE — FONDAMENTAUX
# ============================================================
def get_fundamentals(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        earnings_date = "Inconnue"
        earnings_risk = False
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                ed = cal.iloc[0, 0] if hasattr(cal, 'iloc') else None
                if ed:
                    jours = (pd.Timestamp(ed) - pd.Timestamp.now()).days
                    earnings_date = pd.Timestamp(ed).strftime("%d/%m/%Y")
                    if jours <= 5:
                        earnings_risk = True
        except:
            pass

        inst_ownership  = info.get("institutionOwnership", None)
        inst_pct        = f"{round(inst_ownership * 100, 1)}%" if inst_ownership else "N/A"
        market_cap      = info.get("marketCap", 0)
        market_cap_str  = f"${round(market_cap/1e6)}M" if market_cap else "N/A"
        short_float     = info.get("shortPercentOfFloat", None)
        short_float_str = f"{round(short_float * 100, 1)}%" if short_float else "N/A"
        revenue_growth  = info.get("revenueGrowth", None)
        rev_str         = f"+{round(revenue_growth * 100, 1)}%" if revenue_growth else "N/A"
        earnings_growth = info.get("earningsGrowth", None)
        earn_str        = f"+{round(earnings_growth * 100, 1)}%" if earnings_growth else "N/A"
        sector          = info.get("sector", "N/A")
        industry        = info.get("industry", "N/A")

        news_items = []
        try:
            news = ticker.news
            if news:
                news_items = [n.get("title", "") for n in news[:3]]
        except:
            pass

        return {
            "earnings_date":   earnings_date,
            "earnings_risk":   earnings_risk,
            "inst_ownership":  inst_pct,
            "market_cap":      market_cap_str,
            "short_float":     short_float_str,
            "revenue_growth":  rev_str,
            "earnings_growth": earn_str,
            "sector":          sector,
            "industry":        industry,
            "news_items":      news_items
        }
    except:
        return {
            "earnings_date":   "N/A",
            "earnings_risk":   False,
            "inst_ownership":  "N/A",
            "market_cap":      "N/A",
            "short_float":     "N/A",
            "revenue_growth":  "N/A",
            "earnings_growth": "N/A",
            "sector":          "N/A",
            "industry":        "N/A",
            "news_items":      []
        }

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================
def calculate_indicators(df: pd.DataFrame) -> dict:
    close  = df["close"]
    volume = df["volume"]

    sma20      = close.rolling(20).mean().iloc[-1]
    sma50      = close.rolling(min(50, len(close))).mean().iloc[-1]
    sma20_prev = close.rolling(20).mean().iloc[-2]
    sma50_prev = close.rolling(min(50, len(close))).mean().iloc[-2]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = (100 - (100 / (1 + rs))).iloc[-1]

    avg_volume     = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    rel_volume     = current_volume / avg_volume if avg_volume > 0 else 0

    vol_5j       = volume.tail(5).mean()
    vol_compress = vol_5j / avg_volume if avg_volume > 0 else 1

    perf_month  = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100 if len(close) >= 20 else 0
    returns     = close.pct_change().dropna()
    monthly_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(20) * 100 if len(returns) >= 20 else 0
    adr         = ((df["high"] - df["low"]) / df["close"]).rolling(20).mean().iloc[-1] * 100

    current_price       = close.iloc[-1]
    last_10             = df.tail(10)
    range_high          = last_10["high"].max()
    range_low           = last_10["low"].min()
    consolidation_range = ((range_high - range_low) / range_low) * 100
    is_consolidating    = consolidation_range < 8
    distance_range_high = ((range_high - current_price) / range_high) * 100

    # Variation du jour
    open_price  = df["open"].iloc[-1]
    day_change  = ((current_price - open_price) / open_price) * 100

    jours_conso = 0
    for i in range(len(df) - 1, -1, -1):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if h <= range_high * 1.01 and l >= range_low * 0.99:
            jours_conso += 1
        else:
            break

    breakout = (
        current_price >= range_high * 0.995 and
        rel_volume >= 1.5
    )

    return {
        "price":               round(current_price, 2),
        "day_change":          round(day_change, 2),
        "sma20":               round(sma20, 2),
        "sma50":               round(sma50, 2),
        "sma20_trending_up":   bool(sma20 > sma20_prev),
        "sma50_trending_up":   bool(sma50 > sma50_prev),
        "rsi":                 round(rsi, 1),
        "rel_volume":          round(rel_volume, 2),
        "vol_compress":        round(vol_compress, 2),
        "perf_month":          round(perf_month, 1),
        "monthly_vol":         round(monthly_vol, 1),
        "adr":                 round(adr, 1),
        "consolidation_range": round(consolidation_range, 1),
        "is_consolidating":    bool(is_consolidating),
        "distance_range_high": round(max(distance_range_high, 0), 2),
        "jours_conso":         jours_conso,
        "breakout":            bool(breakout),
        "range_high":          round(range_high, 2),
        "range_low":           round(range_low, 2)
    }

# ============================================================
# SCORE PROXIMITE BREAKOUT
# ============================================================
def calculate_breakout_score(ind: dict) -> tuple[int, list]:
    score   = 0
    details = []

    dist = ind["distance_range_high"]
    if dist <= 1.0:
        score += 3
        details.append(f"📍 Prix a {dist}% du range high — IMMINENT")
    elif dist <= 2.5:
        score += 2
        details.append(f"📍 Prix a {dist}% du range high — TRES PROCHE")
    elif dist <= 5.0:
        score += 1
        details.append(f"📍 Prix a {dist}% du range high — PROCHE")
    else:
        details.append(f"📍 Prix a {dist}% du range high — LOIN")

    compress = ind["vol_compress"]
    if compress <= 0.6:
        score += 2
        details.append(f"📉 Volume compresse a {int(compress*100)}% — FORTE TENSION")
    elif compress <= 0.8:
        score += 1
        details.append(f"📉 Volume compresse a {int(compress*100)}% — TENSION")
    else:
        details.append(f"📊 Volume normal ({int(compress*100)}%)")

    jours = ind["jours_conso"]
    if 5 <= jours <= 10:
        score += 2
        details.append(f"📅 Consolidation ideale : {jours} jours")
    elif 3 <= jours < 5:
        score += 1
        details.append(f"📅 Consolidation courte : {jours} jours")
    elif jours > 10:
        score += 1
        details.append(f"📅 Consolidation longue : {jours} jours")
    else:
        details.append(f"📅 Consolidation trop courte : {jours} jours")

    conso = ind["consolidation_range"]
    if conso <= 4:
        score += 2
        details.append(f"📐 Range tres serre : {conso}% — PARFAIT")
    elif conso <= 6:
        score += 1
        details.append(f"📐 Range serre : {conso}%")
    else:
        details.append(f"📐 Range large : {conso}%")

    if ind["sma20_trending_up"] and ind["sma50_trending_up"]:
        score += 1
        details.append(f"📈 SMA20 & SMA50 pointent vers le haut")
    else:
        details.append(f"📉 SMA pas encore alignees")

    return score, details

def score_label(score: int) -> str:
    if score >= 9:
        return "🔥 BREAKOUT IMMINENT"
    elif score >= 7:
        return "⚡ TRES PROCHE"
    elif score >= 5:
        return "👀 A SURVEILLER"
    else:
        return "⏳ PAS ENCORE PRET"

def score_bar(score: int) -> str:
    filled = "█" * score
    empty  = "░" * (10 - score)
    return f"{filled}{empty} {score}/10"

# ============================================================
# ANALYSE CLAUDE
# ============================================================
def analyse_claude(symbol: str, fund: dict) -> str:
    prompt = f"""Tu es un expert en Swing Trading sur Small Caps US. Analyse l'action [{symbol}]. Ne me donne pas de conseils financiers, donne-moi un diagnostic factuel.

Donnees reelles Yahoo Finance :
- Secteur : {fund['sector']} | Industrie : {fund['industry']}
- Market Cap : {fund['market_cap']}
- Croissance revenus : {fund['revenue_growth']}
- Croissance benefices : {fund['earnings_growth']}
- Ownership institutionnel : {fund['inst_ownership']}
- Short Float : {fund['short_float']}
- Prochain Earnings : {fund['earnings_date']}
- News recentes : {' | '.join(fund['news_items']) if fund['news_items'] else 'Aucune'}

Format STRICT :
SCORE DE CONFIANCE : /10
INSTITUTIONS : [Achat / Vente / Neutre]
RISQUE CALENDRIER : [Date + niveau de risque]
CATALYSEUR : [News specifique ou Technique pur]
VERDICT : [GO / ATTENDRE / REJET] car [Raison courte]"""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erreur Claude: {e}"

# ============================================================
# RESUME DE FIN DE JOURNEE — 16h30
# ============================================================
def resume_fin_journee():
    now     = datetime.now()
    tickers = load_watchlist()
    if not tickers:
        return

    print(f"\n[{now.strftime('%H:%M')}] Resume fin de journee en cours...")

    resultats = []

    for symbol in tickers:
        df = get_bars(symbol)
        if df is None:
            continue
        try:
            ind            = calculate_indicators(df)
            score, details = calculate_breakout_score(ind)
            resultats.append({
                "symbol":  symbol,
                "score":   score,
                "ind":     ind,
            })
        except:
            continue
        time.sleep(0.5)

    # Trie par score decroissant
    resultats.sort(key=lambda x: x["score"], reverse=True)

    # ── EN-TETE ──
    msg  = f"📊 *RAPPORT DE FIN DE JOURNEE*\n"
    msg += f"📅 {now.strftime('%A %d/%m/%Y')} — 16h30\n"
    msg += f"{'─'*32}\n\n"

    # ── ALERTES DU JOUR ──
    alerted   = load_alerted()
    today     = now.strftime("%Y-%m-%d")
    alertes_j = [s for s, d in alerted.items() if d == today]

    if alertes_j:
        msg += f"🚨 *BREAKOUTS DU JOUR :*\n"
        for s in alertes_j:
            msg += f"  ✅ {s}\n"
        msg += "\n"
    else:
        msg += f"😴 *Aucun breakout aujourd'hui*\n\n"

    # ── CLASSEMENT DU JOUR ──
    msg += f"*🏆 CLASSEMENT ACTUEL :*\n"
    msg += f"{'─'*32}\n"

    numeros = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

    imminents  = []
    surveiller = []
    loin       = []

    for i, r in enumerate(resultats):
        s   = r["symbol"]
        ind = r["ind"]
        num = numeros[i] if i < len(numeros) else f"{i+1}."

        # Variation du jour
        chg       = ind["day_change"]
        chg_emoji = "🟢" if chg >= 0 else "🔴"
        chg_str   = f"+{chg}%" if chg >= 0 else f"{chg}%"

        line = (
            f"{num} *{s}*\n"
            f"     {score_bar(r['score'])} {score_label(r['score'])}\n"
            f"     💲{ind['price']} {chg_emoji} {chg_str} aujourd'hui\n"
            f"     🎯 A {ind['distance_range_high']}% du breakout │ {ind['jours_conso']}j conso\n\n"
        )

        if r["score"] >= 7:
            imminents.append(line)
        elif r["score"] >= 5:
            surveiller.append(line)
        else:
            loin.append(line)

    if imminents:
        msg += f"🔥 *BREAKOUT IMMINENT / TRES PROCHE :*\n"
        for l in imminents:
            msg += l

    if surveiller:
        msg += f"👀 *A SURVEILLER :*\n"
        for l in surveiller:
            msg += l

    if loin:
        msg += f"⏳ *PAS ENCORE PRETS :*\n"
        for l in loin:
            msg += l

    # ── CONSEIL DU SOIR ──
    msg += f"{'─'*32}\n"
    if imminents:
        nb = len(imminents)
        msg += (
            f"⚡ *{nb} action(s) tres proche(s) du breakout !*\n"
            f"_Surveille bien l'ouverture demain matin._\n"
            f"_Confirme toujours sur le 30min avant d'entrer._ 🚀"
        )
    else:
        msg += (
            f"😌 *Aucune action imminente ce soir.*\n"
            f"_Le marche consolide, c'est bon signe._\n"
            f"_La tension monte pour les prochains jours._ 💪"
        )

    send_telegram(msg)
    print(f"[Resume] Envoye ! {len(resultats)} actions analysees.")

# ============================================================
# MODE DIMANCHE
# ============================================================
def mode_dimanche():
    now     = datetime.now()
    tickers = load_watchlist()
    if not tickers:
        return

    print(f"\n{'='*55}")
    print(f"MODE DIMANCHE — Scoring de {len(tickers)} actions")
    print(f"{'='*55}")

    send_telegram(
        f"🔍 *ANALYSE DOMINICALE*\n"
        f"{'─'*30}\n"
        f"Scoring de tes *{len(tickers)} actions* Finviz\n"
        f"Classement par proximite de breakout\n\n"
        f"⏳ _Patiente quelques minutes..._"
    )

    reset_alerted()
    resultats = []
    erreurs   = []

    for i, symbol in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {symbol}...")

        df   = get_bars(symbol)
        fund = get_fundamentals(symbol)

        if df is None:
            erreurs.append(symbol)
            print(f"  -> Donnees insuffisantes")
            continue

        try:
            ind = calculate_indicators(df)
        except Exception as e:
            erreurs.append(symbol)
            print(f"  -> Erreur: {e}")
            continue

        score, details = calculate_breakout_score(ind)
        analyse        = analyse_claude(symbol, fund)

        resultats.append({
            "symbol":   symbol,
            "score":    score,
            "details":  details,
            "ind":      ind,
            "fund":     fund,
            "analyse":  analyse,
            "breakout": ind["breakout"]
        })

        print(f"  -> Score: {score}/10 — {score_label(score)}")
        time.sleep(1)

    resultats.sort(key=lambda x: x["score"], reverse=True)

    # Classement rapide
    numeros    = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    classement  = f"🏆 *CLASSEMENT BREAKOUT*\n"
    classement += f"📅 {now.strftime('%d/%m/%Y')} — {len(resultats)} actions\n"
    classement += f"{'─'*30}\n\n"

    for i, r in enumerate(resultats):
        num = numeros[i] if i < len(numeros) else f"{i+1}."
        brk = " 🚨" if r["breakout"] else ""
        classement += (
            f"{num} *{r['symbol']}*{brk}\n"
            f"     {score_bar(r['score'])} {score_label(r['score'])}\n"
            f"     💲{r['ind']['price']} │ "
            f"🎯 {r['ind']['distance_range_high']}% du top │ "
            f"🕐 {r['ind']['jours_conso']}j\n\n"
        )

    send_telegram(classement)
    time.sleep(2)

    # Fiche complete de chaque action
    for r in resultats:
        s    = r["symbol"]
        ind  = r["ind"]
        fund = r["fund"]
        det  = "\n".join(r["details"])

        sma20_arrow = "↑" if ind["sma20_trending_up"] else "↓"
        sma50_arrow = "↑" if ind["sma50_trending_up"] else "↓"
        earn_risk   = " ⚠️ RISQUE" if fund["earnings_risk"] else " ✅"

        news_str = "\n".join([f"  • {n[:70]}..." for n in fund["news_items"]]) if fund["news_items"] else "  • Aucune news recente"

        msg = (
            f"{'─'*32}\n"
            f"📊 *{s}* — {score_label(r['score'])}\n"
            f"{'─'*32}\n\n"
            f"*🎯 SCORE BREAKOUT*\n"
            f"`{score_bar(r['score'])}`\n"
            f"{det}\n\n"
            f"*📈 TECHNIQUE*\n"
            f"💲 Prix : *${ind['price']}*\n"
            f"⚡ RSI : {ind['rsi']} │ Vol : {ind['rel_volume']}x\n"
            f"📏 SMA20 {sma20_arrow} ${ind['sma20']} │ SMA50 {sma50_arrow} ${ind['sma50']}\n"
            f"📦 Range : ${ind['range_low']} → ${ind['range_high']}\n"
            f"⏳ Conso : {ind['consolidation_range']}% sur {ind['jours_conso']} jours\n"
            f"🔥 Perf mois : +{ind['perf_month']}% │ ADR : {ind['adr']}%\n\n"
            f"*💼 FONDAMENTAUX*\n"
            f"🏭 {fund['sector']} — {fund['industry']}\n"
            f"💰 Market Cap : {fund['market_cap']}\n"
            f"🏦 Institutions : {fund['inst_ownership']}\n"
            f"📉 Short Float : {fund['short_float']}\n"
            f"📊 Rev : {fund['revenue_growth']} │ EPS : {fund['earnings_growth']}\n"
            f"📅 Earnings : {fund['earnings_date']}{earn_risk}\n\n"
            f"*📰 NEWS*\n"
            f"{news_str}\n\n"
            f"*🤖 ANALYSE IA*\n"
            f"{r['analyse']}\n\n"
            f"_✅ Verifie la RS Line sur Finviz avant d'entrer !_"
        )
        send_telegram(msg)
        time.sleep(3)

    top = resultats[:3]
    send_telegram(
        f"✅ *ANALYSE TERMINEE !*\n"
        f"{'─'*30}\n\n"
        f"🏅 *Top 3 de la semaine :*\n"
        f"🥇 {top[0]['symbol'] if len(top) > 0 else '—'} — {score_label(top[0]['score']) if len(top) > 0 else ''}\n"
        f"🥈 {top[1]['symbol'] if len(top) > 1 else '—'}\n"
        f"🥉 {top[2]['symbol'] if len(top) > 2 else '—'}\n\n"
        f"🤖 _Surveillance automatique lun-ven_\n"
        f"_Rapport quotidien a 16h30 chaque jour_ 📊\n"
        f"_Alerte immediate au breakout_ 🚨"
    )

    print(f"\nMode dimanche termine ! {len(resultats)} actions scorees.")

# ============================================================
# MODE SEMAINE — Surveillance breakout
# ============================================================
def mode_surveillance():
    now     = datetime.now()
    tickers = load_watchlist()
    if not tickers:
        return

    alerted     = load_alerted()
    alerts_sent = 0

    print(f"[{now.strftime('%H:%M:%S')}] Surveillance de {len(tickers)} actions...")

    for symbol in tickers:
        today = now.strftime("%Y-%m-%d")
        if alerted.get(symbol) == today:
            continue

        df = get_bars(symbol)
        if df is None:
            continue

        try:
            ind = calculate_indicators(df)
        except:
            continue

        if ind["breakout"]:
            print(f"  BREAKOUT : {symbol} !")
            fund    = get_fundamentals(symbol)
            analyse = analyse_claude(symbol, fund)

            sma20_arrow = "↑" if ind["sma20_trending_up"] else "↓"
            sma50_arrow = "↑" if ind["sma50_trending_up"] else "↓"
            earn_risk   = " ⚠️ RISQUE ELEVE" if fund["earnings_risk"] else " ✅ OK"
            news_str    = "\n".join([f"  • {n[:70]}..." for n in fund["news_items"]]) if fund["news_items"] else "  • Aucune news recente"

            chg     = ind["day_change"]
            chg_str = f"+{chg}%" if chg >= 0 else f"{chg}%"

            message = (
                f"🚨🚨 *BREAKOUT CONFIRME* 🚨🚨\n"
                f"{'─'*32}\n"
                f"📊 *{symbol}*\n"
                f"🕐 {now.strftime('%d/%m/%Y a %H:%M')}\n"
                f"{'─'*32}\n\n"
                f"*📈 TECHNIQUE*\n"
                f"💲 Prix : *${ind['price']}* ({chg_str} aujourd'hui)\n"
                f"⚡ RSI : {ind['rsi']}\n"
                f"📦 Volume : *{ind['rel_volume']}x* la normale 🔥\n"
                f"📏 SMA20 {sma20_arrow} │ SMA50 {sma50_arrow}\n"
                f"💥 Cassure du range : *${ind['range_high']}*\n"
                f"📐 Perf mois : +{ind['perf_month']}%\n\n"
                f"*💼 FONDAMENTAUX*\n"
                f"🏦 Institutions : {fund['inst_ownership']}\n"
                f"📅 Earnings : {fund['earnings_date']}{earn_risk}\n\n"
                f"*📰 NEWS*\n"
                f"{news_str}\n\n"
                f"*🤖 ANALYSE IA*\n"
                f"{analyse}\n\n"
                f"{'─'*32}\n"
                f"_⚡ Confirme l'entree sur le graphique 30min !_"
            )
            send_telegram(message)

            alerted[symbol] = today
            save_alerted(alerted)
            alerts_sent += 1
            time.sleep(2)

        time.sleep(0.5)

    if alerts_sent == 0:
        print(f"  Aucun breakout detecte.")
    else:
        print(f"  {alerts_sent} breakout(s) envoye(s) !")

# ============================================================
# LANCEMENT
# ============================================================
def main():
    now     = datetime.now()
    weekday = now.weekday()

    print("TradingBot demarre !")
    print(f"Jour : {now.strftime('%A %d/%m/%Y %H:%M')}")

    if weekday == 6:
        print("Mode : DIMANCHE")
        mode_dimanche()

    elif weekday < 5:
        print("Mode : SEMAINE — Surveillance + Resume 16h30")

        if weekday == 0:
            reset_alerted()

        send_telegram(
            f"🤖 *TRADINGBOT ACTIF*\n"
            f"{'─'*30}\n"
            f"📋 {len(load_watchlist())} actions surveillees\n"
            f"🔍 Scan toutes les 30 minutes\n"
            f"📊 Rapport quotidien a 16h30\n"
            f"🚨 Alerte immediate au breakout !\n"
            f"{'─'*30}\n"
            f"_Bonne journee, je surveille pour toi_ 💪"
        )

        # Scan immediat
        mode_surveillance()

        # Scan toutes les 30 minutes
        schedule.every(30).minutes.do(mode_surveillance)

        # Resume quotidien a 16h30 (heure du serveur = UTC)
        # Railway est en UTC donc 16h30 EST = 21h30 UTC
        schedule.every().day.at("21:30").do(resume_fin_journee)

        while True:
            schedule.run_pending()
            time.sleep(30)

    else:
        print("Mode : SAMEDI — Bot en pause")

if __name__ == "__main__":
    main()
