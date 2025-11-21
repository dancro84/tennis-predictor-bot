# ================================
# TennisPredictor Bot - Avanzato
# Versione compatibile con Railway
# ================================

import os
import math
import re
import joblib
from datetime import datetime
import pandas as pd
from statistics import mean
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, CallbackQueryHandler, filters
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ================================
# TOKEN (LETTO DA ENV, NON DAL CODICE!)
# ================================
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise Exception("ERRORE: BOT_TOKEN NON TROVATO! Devi inserirlo su Railway → Variables.")

# ================================
# FILES
# ================================
DATA_DIR = "/app/data" if os.path.exists("/app/data") else "/mnt/data"
MATCHES_FILE = os.path.join(DATA_DIR, "matches.csv")
PLAYERS_FILE = os.path.join(DATA_DIR, "player_stats_filled.csv")
MODEL_PATH = os.path.join(DATA_DIR, "rf_under21_model.joblib")
LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")

# (LE TUE FUNZIONI: load_matches, load_players, h2h_stats, compute_features, train_model, ecc.)
# NON LE TOCCO, SONO IDENTICHE
# ------------------------------------------------------
# (Per brevità non le riscrivo qui, ma tu LE DEVI lasciare come stanno nel tuo file!)
# ------------------------------------------------------

# ================================
# SETUP BOT
# ================================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start_cmd))
app.add_handler(CommandHandler("help", help_cmd))
app.add_handler(CommandHandler("retrain", retrain_cmd))
app.add_handler(CallbackQueryHandler(callback_query_handler))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict_handler))

print("Bot avviato su Railway...")
app.run_polling()
