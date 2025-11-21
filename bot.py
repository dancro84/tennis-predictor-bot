import os
import re
import math
import joblib
import pandas as pd
from datetime import datetime
from statistics import mean

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ================================
# CONFIGURAZIONE
# ================================
TOKEN = os.getenv("BOT_TOKEN")  # Consigliato: impostare su Railway Environment
DATA_DIR = "."

MATCHES_FILE = os.path.join(DATA_DIR, "matches.csv")
PLAYERS_FILE = os.path.join(DATA_DIR, "player_stats_filled.csv")
MODEL_PATH = os.path.join(DATA_DIR, "rf_under21_model.joblib")
LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")


# ================================
# CARICAMENTO DATI
# ================================
def load_matches():
    if not os.path.exists(MATCHES_FILE):
        return pd.DataFrame()

    df = pd.read_csv(MATCHES_FILE)

    # parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # compute total games if missing
    if "total_games" not in df.columns:
        def parse_games(score):
            if pd.isna(score):
                return None
            sets = re.findall(r"(\d+)-(\d+)", str(score))
            return sum(int(a) + int(b) for a, b in sets)

        df["total_games"] = df["result"].apply(parse_games)

    # under label
    df["under21_label"] = df["total_games"].apply(
        lambda x: 1 if (not pd.isna(x) and x <= 21) else 0
    )

    return df


def load_players():
    if not os.path.exists(PLAYERS_FILE):
        return pd.DataFrame()
    return pd.read_csv(PLAYERS_FILE)


matches_df = load_matches()
players_df = load_players()

players_index = {}
if not players_df.empty:
    players_index = {
        row["player"]: row for _, row in players_df.iterrows()
    }


# ================================
# H2H / FORMA
# ================================
def h2h_stats(p1, p2, surface=None):
    df = matches_df
    cond = (
        ((df["player1"] == p1) & (df["player2"] == p2))
        | ((df["player1"] == p2) & (df["player2"] == p1))
    )
    sub = df[cond]
    wins_p1 = (sub["winner"] == p1).sum()
    wins_p2 = (sub["winner"] == p2).sum()
    total = len(sub)

    # superficie
    wins_p1_surf = wins_p2_surf = 0
    if surface:
        ss = sub[sub["surface"].astype(str).str.lower().str.contains(surface.lower(), na=False)]
        wins_p1_surf = (ss["winner"] == p1).sum()
        wins_p2_surf = (ss["winner"] == p2).sum()

    return {
        "wins_p1": int(wins_p1),
        "wins_p2": int(wins_p2),
        "total": int(total),
        "wins_p1_surface": int(wins_p1_surf),
        "wins_p2_surface": int(wins_p2_surf),
    }


def last_n_matches(player, n=5):
    df = matches_df
    sub = df[(df["player1"] == player) | (df["player2"] == player)].sort_values("date", ascending=False)
    return sub.head(n)


def player_form_score(player, n=5):
    last = last_n_matches(player, n)
    if last.empty:
        return 0.5
    wins = (last["winner"] == player).sum()
    return wins / len(last)


# ================================
# FEATURE ESTRAZIONE
# ================================
def compute_features(p1, p2, extra):
    feat = {}

    # ranks
    r1 = players_index.get(p1, {}).get("rank")
    r2 = players_index.get(p2, {}).get("rank")

    if r1 and r2 and not pd.isna(r1) and not pd.isna(r2):
        feat["rank_diff"] = float(r2) - float(r1)
    else:
        feat["rank_diff"] = 0.0

    # under storici
    u1 = players_index.get(p1, {}).get("under21_pct", 45) / 100
    u2 = players_index.get(p2, {}).get("under21_pct", 45) / 100
    feat["avg_under"] = (u1 + u2) / 2

    # forma
    feat["form_p1"] = player_form_score(p1)
    feat["form_p2"] = player_form_score(p2)

    # head to head
    surf = extra.get("surface")
    h = h2h_stats(p1, p2, surf)
    feat.update(h)

    # bookmaker → implied probability
    if extra.get("book_odds_under"):
        try:
            feat["book_implied_under"] = 1 / float(extra["book_odds_under"])
        except:
            feat["book_implied_under"] = 0.5
    else:
        feat["book_implied_under"] = 0.5

    # handedness
    h1 = players_index.get(p1, {}).get("hand", "")
    h2 = players_index.get(p2, {}).get("hand", "")
    feat["hand_adj"] = 0.0
    if h1.startswith("R") and h2.startswith("L"):
        feat["hand_adj"] = -0.04
    if h1.startswith("L") and h2.startswith("R"):
        feat["hand_adj"] = 0.04

    return feat


# ================================
# MODELLO ML
# ================================
def train_model():
    df = matches_df
    rows = []

    for _, row in df.iterrows():
        p1 = row["player1"]
        p2 = row["player2"]

        extra = {"surface": row.get("surface"), "book_odds_under": row.get("player1_odd")}
        feat = compute_features(p1, p2, extra)

        rows.append({
            "rank_diff": feat["rank_diff"],
            "avg_under": feat["avg_under"],
            "form_diff": feat["form_p1"] - feat["form_p2"],
            "h2h_total": feat["total"],
            "h2h_diff": feat["wins_p1"] - feat["wins_p2"],
            "book_implied_under": feat["book_implied_under"],
            "hand_adj": feat["hand_adj"],
            "label": row["under21_label"],
        })

    df_feat = pd.DataFrame(rows).dropna()

    if df_feat.empty:
        return None, "Dati insufficienti."

    X = df_feat.drop("label", axis=1)
    y = df_feat["label"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=8)
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds)

    joblib.dump(model, MODEL_PATH)
    return model, f"Modello RF aggiornato — ACC={acc:.3f}, F1={f1:.3f}"


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            return None
    return None


rf_model = load_model()


# ================================
# PREDIZIONE
# ================================
def rule_predict(p1, p2, extra):
    f = compute_features(p1, p2, extra)
    prob = (
        0.45 * f["avg_under"]
        + 0.2 * f["book_implied_under"]
        + 0.1 * (f["form_p1"] - f["form_p2"])
        + 0.05 * (f["wins_p1"] - f["wins_p2"])
        + f["hand_adj"]
    )
    return max(0.01, min(0.99, prob)), f


def ml_predict(p1, p2, extra):
    if rf_model is None:
        return None, None

    f = compute_features(p1, p2, extra)

    X = pd.DataFrame([{
        "rank_diff": f["rank_diff"],
        "avg_under": f["avg_under"],
        "form_diff": f["form_p1"] - f["form_p2"],
        "h2h_total": f["total"],
        "h2h_diff": f["wins_p1"] - f["wins_p2"],
        "book_implied_under": f["book_implied_under"],
        "hand_adj": f["hand_adj"],
    }])

    prob = rf_model.predict_proba(X)[0][1]
    return float(prob), f


# ================================
# TELEGRAM HANDLERS
# ================================
HELP_TEXT = (
    "🎾 *TennisPredictor*\n"
    "Scrivi: `GiocatoreA, GiocatoreB, superficie, quota_under`\n"
    "Esempio: `Sinner, Alcaraz, hard, 1.85`\n\n"
    "Comandi:\n"
    "/start — Avvio bot\n"
    "/help — Aiuto\n"
    "/retrain — Allena il modello\n"
)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎾 Bot attivo!\nScrivi /help per le istruzioni.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def retrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Addestramento del modello in corso...")
    model, info = train_model()
    global rf_model
    rf_model = model
    await update.message.reply_text("✅ " + info)


async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # normalizzazione input
    text = text.replace(" vs ", ",").replace(" - ", ",")
    parts = [p.strip() for p in text.split(",")]

    if len(parts) < 2:
        await update.message.reply_text("Formato non valido. Usa: A, B, superficie, quota")
        return

    p1, p2 = parts[0], parts[1]
    extra = {}

    if len(parts) >= 3:
        extra["surface"] = parts[2]
    if len(parts) >= 4:
        extra["book_odds_under"] = parts[3]

    # matching nomi
    def find_name(name):
        for k in players_index:
            if name.lower() in k.lower():
                return k
        return None

    p1 = find_name(p1)
    p2 = find_name(p2)

    if not p1 or not p2:
        await update.message.reply_text("❌ Giocatori non trovati nel database.")
        return

    ml_prob, _ = ml_predict(p1, p2, extra)
    rule_prob, _ = rule_predict(p1, p2, extra)

    if ml_prob:
        prob = 0.7 * ml_prob + 0.3 * rule_prob
        src = "ML + Regole"
    else:
        prob = rule_prob
        src = "Regole"

    suggestion = "UNDER 21.5" if prob > 0.55 else ("OVER 21.5" if prob < 0.45 else "NO BET")

    pct = round(prob * 100, 1)

    # inline buttons
    keyboard = [
        [InlineKeyboardButton("Dettagli", callback_data=f"D|{p1}|{p2}|{extra.get('surface','')}")],
        [InlineKeyboardButton("Spiegazione", callback_data=f"E|{p1}|{p2}|{extra.get('surface','')}")]
    ]

    await update.message.reply_text(
        f"🎯 *{p1} vs {p2}*\n"
        f"Probabilità UNDER: *{pct}%*\n"
        f"Suggerimento: *{suggestion}*\n"
        f"Modello: {src}",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    typ, p1, p2, surf = q.data.split("|")
    extra = {"surface": surf}

    feat = compute_features(p1, p2, extra)
    h = h2h_stats(p1, p2, surf)

    if typ == "D":
        text = (
            f"📊 *Dettagli Match*\n"
            f"H2H: {h['wins_p1']} - {h['wins_p2']} (totale {h['total']})\n"
            f"Ultime 5: {feat['form_p1']:.2f} vs {feat['form_p2']:.2f}\n"
            f"Under storico medio: {feat['avg_under']:.2f}\n"
            f"Book implied: {feat['book_implied_under']:.2f}\n"
            f"Rank diff: {feat['rank_diff']:.1f}\n"
            f"Hand effect: {feat['hand_adj']:.2f}"
        )
        await q.edit_message_text(text, parse_mode="Markdown")

    if typ == "E":
        reasons = []
        if feat["avg_under"] > 0.5:
            reasons.append("Storico UNDER favorevole")
        if feat["form_p1"] > feat["form_p2"]:
            reasons.append(f"{p1} in forma migliore")
        if h["total"] > 0:
            reasons.append(f"H2H: {h['wins_p1']} - {h['wins_p2']}")
        if not reasons:
            reasons.append("Dati limitati → valutazione standard.")

        await q.edit_message_text("• " + "\n• ".join(reasons))


# ================================
# AVVIO BOT
# ================================
if __name__ == "__main__":
    print("▶️ Avvio bot...")
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("retrain", retrain_cmd))
    app.add_handler(CallbackQueryHandler(callback_query_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict_handler))

    app.run_polling()
