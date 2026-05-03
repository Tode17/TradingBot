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

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY    = os.getenv("CLAUDE_API_KEY")

ALPACA_BASE_URL   = "https://data.alpaca.markets/v2"
ALPACA_HEADERS    = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

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
        send_telegram("Fichier watchlist.txt introuvable ! Cree-le et relance le bot.")
        return []

# ============================================================
# ALERTES DEJA ENVOYEES
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
# DONNÉES BARS
# ============================================================
def get_bars(symbol: str, limit: int = 60):
    url    = f"{ALPACA_BASE_URL}/stocks/{symbol}/bars"
    params = {"timeframe": "1Day", "limit": limit, "feed": "iex"}
    try:
        r    = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
        data = r.json()
        bars = data.get("bars", [])
        if not bars or len(bars) < 20:
            return None
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df
    except:
        return None

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

    # Compression volume 5 derniers jours
    vol_5j       = volume.tail(5).mean()
    vol_compress = vol_5j / avg_volume if avg_volume > 0 else 1

    perf_month  = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100 if len(close) >= 20 else 0
    returns     = close.pct_change().dropna()
    monthly_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(20) * 100 if len(returns) >= 20 else 0
    adr         = ((df["high"] - df["low"]) / df["close"]).rolling(20).mean().iloc[-1] * 100

    current_price = close.iloc[-1]

    # Range 10 jours
    last_10             = df.tail(10)
    range_high          = last_10["high"].max()
    range_low           = last_10["low"].min()
    consolidation_range = ((range_high - range_low) / range_low) * 100
    is_consolidating    = consolidation_range < 8

    # Distance au range high
    distance_range_high = ((range_high - current_price) / range_high) * 100

    # Jours consecutifs dans le range
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
        "price": round(current_price, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma20_trending_up": bool(sma20 > sma20_prev),
        "sma50_trending_up": bool(sma50 > sma50_prev),
        "rsi": round(rsi, 1),
        "rel_volume": round(rel_volume, 2),
        "vol_compress": round(vol_compress, 2),
        "perf_month": round(perf_month, 1),
        "monthly_vol": round(monthly_vol, 1),
        "adr": round(adr, 1),
        "consolidation_range": round(consolidation_range, 1),
        "is_consolidating": bool(is_consolidating),
        "distance_range_high": round(max(distance_range_high, 0), 2),
        "jours_conso": jours_conso,
        "breakout": bool(breakout),
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2)
    }

# ============================================================
# SCORE PROXIMITE BREAKOUT (0 a 10)
# ============================================================
def calculate_breakout_score(ind: dict) -> tuple[int, list]:
    score   = 0
    details = []

    # 1. Distance au range high (max 3 pts)
    dist = ind["distance_range_high"]
    if dist <= 1.0:
        score += 3
        details.append(f"Prix a {dist}% du range high (IMMINENT)")
    elif dist <= 2.5:
        score += 2
        details.append(f"Prix a {dist}% du range high (TRES PROCHE)")
    elif dist <= 5.0:
        score += 1
        details.append(f"Prix a {dist}% du range high (PROCHE)")
    else:
        details.append(f"Prix a {dist}% du range high (LOIN)")

    # 2. Compression du volume (max 2 pts)
    compress = ind["vol_compress"]
    if compress <= 0.6:
        score += 2
        details.append(f"Volume compresse a {int(compress*100)}% (FORTE TENSION)")
    elif compress <= 0.8:
        score += 1
        details.append(f"Volume compresse a {int(compress*100)}% (TENSION)")
    else:
        details.append(f"Volume normal ({int(compress*100)}%)")

    # 3. Jours de consolidation (max 2 pts)
    jours = ind["jours_conso"]
    if 5 <= jours <= 10:
        score += 2
        details.append(f"Consolidation ideale : {jours} jours")
    elif 3 <= jours < 5:
        score += 1
        details.append(f"Consolidation courte : {jours} jours")
    elif jours > 10:
        score += 1
        details.append(f"Consolidation longue : {jours} jours")
    else:
        details.append(f"Consolidation trop courte : {jours} jours")

    # 4. Resserrement du range (max 2 pts)
    conso = ind["consolidation_range"]
    if conso <= 4:
        score += 2
        details.append(f"Range tres serre : {conso}% (PARFAIT)")
    elif conso <= 6:
        score += 1
        details.append(f"Range serre : {conso}%")
    else:
        details.append(f"Range large : {conso}% (pas encore en conso)")

    # 5. SMA alignees (max 1 pt)
    if ind["sma20_trending_up"] and ind["sma50_trending_up"]:
        score += 1
        details.append("SMA20 & SMA50 pointent vers le haut")
    else:
        details.append("SMA pas encore idealement alignees")

    return score, details

# ============================================================
# LABEL DU SCORE
# ============================================================
def score_label(score: int) -> str:
    if score >= 9:
        return "BREAKOUT IMMINENT"
    elif score >= 7:
        return "TRES PROCHE"
    elif score >= 5:
        return "A SURVEILLER"
    else:
        return "PAS ENCORE PRET"

# ============================================================
# ANALYSE CLAUDE
# ============================================================
def analyse_claude(symbol: str) -> str:
    prompt = f"""Tu es un expert en Swing Trading sur Small Caps US. Analyse l'action [{symbol}] selon ma strategie stricte. Ne me donne pas de conseils financiers, donne-moi un diagnostic factuel.

1. Transactions Institutionnelles : Le solde des transactions sur les 3 derniers mois est-il positif ou negatif ?
2. Calendrier Macro & Earnings : Quelle est la date du prochain rapport de resultats ? (Si < 5 jours, signale un risque eleve). Y a-t-il une annonce FED ou CPI dans les 48h ?
3. Catalyseur de News : Y a-t-il une news specifique dans les 48h (contrat, FDA, partenariat) ou est-ce purement technique ?

Format STRICT :
SCORE DE CONFIANCE : /10
INSTITUTIONS : [Achat/Vente/Neutre]
RISQUE CALENDRIER : [Date Earnings / News Fed]
CATALYSEUR : [News ou Technique pur]
VERDICT : [GO / ATTENDRE / REJET] car [Raison courte]"""

    try:
        message = claude_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erreur Claude: {e}"

# ============================================================
# MODE DIMANCHE — Score et classe TOUTES les actions
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
        f"*Analyse dominicale en cours...*\n"
        f"Scoring de tes {len(tickers)} actions Finviz.\n"
        f"Classement complet dans quelques minutes !"
    )

    reset_alerted()

    resultats  = []
    deja_break = []
    erreurs    = []

    for i, symbol in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {symbol}...")

        df = get_bars(symbol)
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

        # Deja en breakout — on note mais on score quand meme
        if ind["breakout"]:
            deja_break.append(symbol)
            print(f"  -> Deja en breakout !")

        # Score TOUTES les actions sans exception
        score, details = calculate_breakout_score(ind)
        analyse        = analyse_claude(symbol)

        resultats.append({
            "symbol":   symbol,
            "score":    score,
            "details":  details,
            "ind":      ind,
            "analyse":  analyse,
            "breakout": ind["breakout"]
        })

        print(f"  -> Score: {score}/10 — {score_label(score)}")
        time.sleep(1)

    # Trie par score decroissant
    resultats.sort(key=lambda x: x["score"], reverse=True)

    # ---- RAPPORT TELEGRAM ----

    # 1. Classement rapide
    classement = f"*CLASSEMENT BREAKOUT — {now.strftime('%d/%m/%Y')}*\n"
    classement += f"_{len(resultats)} actions analysees_\n\n"

    numeros = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

    for i, r in enumerate(resultats):
        num   = numeros[i] if i < len(numeros) else f"{i+1}."
        label = score_label(r["score"])
        fire  = "🔥" if r["score"] >= 7 else ("⚡" if r["score"] >= 5 else "👀")
        brk   = " ⚠️ DEJA EN BREAKOUT" if r["breakout"] else ""

        classement += (
            f"{num} *{r['symbol']}* — {r['score']}/10 {fire} {label}{brk}\n"
            f"     ${r['ind']['price']} | "
            f"A {r['ind']['distance_range_high']}% du breakout | "
            f"{r['ind']['jours_conso']}j conso\n"
        )

    send_telegram(classement)
    time.sleep(2)

    # 2. Fiche complete de chaque action
    for r in resultats:
        s   = r["symbol"]
        ind = r["ind"]
        det = "\n".join([f"  • {d}" for d in r["details"]])
        brk = "⚠️ DEJA EN BREAKOUT — surveille quand meme\n\n" if r["breakout"] else ""

        msg = (
            f"*{s}* — Score {r['score']}/10 — {score_label(r['score'])}\n"
            f"{brk}"
            f"\n*Proximite breakout :*\n{det}\n\n"
            f"*Technique :*\n"
            f"Prix: ${ind['price']} | RSI: {ind['rsi']}\n"
            f"Range: ${ind['range_low']} → ${ind['range_high']}\n"
            f"Consolidation: {ind['consolidation_range']}% sur {ind['jours_conso']}j\n"
            f"SMA20: {'↑' if ind['sma20_trending_up'] else '↓'} | "
            f"SMA50: {'↑' if ind['sma50_trending_up'] else '↓'}\n"
            f"Perf mois: +{ind['perf_month']}% | ADR: {ind['adr']}% | Vol: {ind['rel_volume']}x\n\n"
            f"*Analyse IA :*\n{r['analyse']}\n\n"
            f"_RS Line a verifier sur Finviz !_"
        )
        send_telegram(msg)
        time.sleep(3)

    # 3. Message de fin
    top3 = [r["symbol"] for r in resultats[:3]]
    send_telegram(
        f"*Analyse terminee !*\n\n"
        f"Top 3 a surveiller cette semaine :\n"
        f"*{' | '.join(top3)}*\n\n"
        f"Le bot surveille automatiquement du lundi au vendredi.\n"
        f"Tu recevras une alerte des qu'un breakout est confirme !"
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
            analyse = analyse_claude(symbol)

            message = (
                f"\U0001F6A8 *BREAKOUT CONFIRME : {symbol}*\n"
                f"\u23F0 {now.strftime('%d/%m/%Y %H:%M')}\n\n"
                f"\U0001F4CA *TECHNIQUE*\n"
                f"\U0001F4B0 Prix: ${ind['price']}\n"
                f"\u26A1 RSI: {ind['rsi']}\n"
                f"\U0001F4E6 Volume: {ind['rel_volume']}x la normale\n"
                f"\U0001F4C8 SMA20 {'↑' if ind['sma20_trending_up'] else '↓'} | "
                f"SMA50 {'↑' if ind['sma50_trending_up'] else '↓'}\n"
                f"\U0001F525 Perf mois: +{ind['perf_month']}%\n"
                f"\U0001F513 Cassure: ${ind['range_high']} explose !\n"
                f"Volume: {ind['rel_volume']}x\n\n"
                f"\U0001F9E0 *ANALYSE IA*\n"
                f"{analyse}\n\n"
                f"_Confirme l'entree sur le 30min !_"
            )
            send_telegram(message)

            alerted[symbol] = today
            save_alerted(alerted)
            alerts_sent += 1
            time.sleep(2)

        time.sleep(0.3)

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
        print("Mode : DIMANCHE — Scoring complet de toutes les actions")
        mode_dimanche()

    elif weekday < 5:
        print("Mode : SEMAINE — Surveillance breakout toutes les 30min")

        if weekday == 0:
            reset_alerted()

        send_telegram(
            f"*TradingBot actif !*\n"
            f"Surveillance de ta watchlist en cours.\n"
            f"Alerte immediate des qu'un breakout est confirme !"
        )

        mode_surveillance()
        schedule.every(30).minutes.do(mode_surveillance)

        while True:
            schedule.run_pending()
            time.sleep(30)

    else:
        print("Mode : SAMEDI — Bot en pause")
        print("Mets a jour watchlist.txt depuis Finviz et relance dimanche !")

if __name__ == "__main__":
    main()
