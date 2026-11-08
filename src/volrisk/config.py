"""Paramètres centraux du projet : chemins, tickers, dates et constantes."""

from pathlib import Path

# --- Chemins -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# --- Données ---------------------------------------------------------------
TICKERS = {
    "sp500": "^GSPC",
    "vix": "^VIX",
}
START = "2015-01-01"
END = "2026-11-01"  # yfinance exclut la date de fin : on obtient le 31/10/2026

# --- Paramètres d'analyse --------------------------------------------------
WINDOWS = [21, 63]               # fenêtres glissantes : 1 mois et 1 trimestre de bourse
CONFIDENCE_LEVELS = [0.95, 0.99] # niveaux de VaR et d'Expected Shortfall
TRADING_DAYS = 252               # nombre de séances par an, pour annualiser