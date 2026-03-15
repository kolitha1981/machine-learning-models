import csv
import math
import os
import random

# ---------------------------------------------------------------------------
# seq-to-seq-data-csv.py
# ---------------------------------------------------------------------------
# PURPOSE
#   Generate ~100,000 sliding-window stock-price records and write them to
#   two headerless CSV files (80 % train / 20 % validation).
#
# OUTPUT FILES
#   seq-to-seq-train.csv   — 80 % of records, used for training
#   seq-to-seq-val.csv     — 20 % of records, used for validation
#
# CSV FORMAT (35 columns, NO header row, all floats)
#   Columns  0-29  — source sequence: 30 consecutive daily closing prices
#   Columns 30-34  — target sequence: the 5 trading-day closing prices that
#                    immediately follow the 30-day source window
#
#   Example row:
#     182.34, 183.01, 181.75, ...(27 more)..., 190.22, 191.05, 189.80, 188.50, 192.10
#     \___________ 30 source prices ___________/ \________ 5 target prices ________/
#
# !! IMPORTANT — CSV IS FOR EXPLORATION / INSPECTION ONLY !!
#   The SageMaker Seq2Seq container does NOT accept CSV as a training input.
#   Its container exclusively reads RecordIO-wrapped protobuf (.rec) files.
#   Use seq-to-seq-data-recordio.py to generate the files required for actual
#   SageMaker Seq2Seq training.
#   This CSV is useful for:
#     • Inspecting the raw data in Excel, pandas, or a text editor
#     • Sanity-checking the GBM price simulation visually
#     • Feeding alternative algorithms (DeepAR, XGBoost baseline) that do
#       accept CSV via SageMaker
#
# DEPENDENCIES
#   Standard library only — no pip installs required.
#
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Sliding-window dimensions — must match the Seq2Seq hyperparameters
# max_seq_len_source and max_seq_len_target in seq-to-seq.py.
SOURCE_LEN = 30      # Number of historical trading days fed as encoder input
TARGET_LEN = 5       # Number of future trading days the decoder must predict
WINDOW_LEN = SOURCE_LEN + TARGET_LEN   # Total days per record = 35

# Target total number of sliding-window records across all tickers.
# With 5 tickers and ~820 trading days of history each, we get
# 5 × (820 - 35 + 1) = 5 × 786 = 3,930 natural windows.
# To reach ~100,000 records we generate enough days per ticker so that
# the total window count across all tickers meets or exceeds this target.
TARGET_RECORDS = 100_000

# Five synthetic tickers with realistic starting prices, annualised drift
# (μ) and annualised volatility (σ) modelled on real large-cap behaviour.
#
#   Ticker  Start $   Ann. drift   Ann. volatility
#   ------  --------  -----------  ----------------
#   AAPL      182.00     0.12           0.28   (moderately volatile tech)
#   MSFT      375.00     0.14           0.24   (steady compounder)
#   AMZN      178.00     0.16           0.32   (higher-growth, higher vol)
#   GOOGL     140.00     0.13           0.26   (large-cap search/cloud)
#   TSLA      240.00     0.10           0.55   (very high volatility EV)
#
TICKERS = {
    "AAPL":  {"start": 182.00, "drift": 0.12, "volatility": 0.28},
    "MSFT":  {"start": 375.00, "drift": 0.14, "volatility": 0.24},
    "AMZN":  {"start": 178.00, "drift": 0.16, "volatility": 0.32},
    "GOOGL": {"start": 140.00, "drift": 0.13, "volatility": 0.26},
    "TSLA":  {"start": 240.00, "drift": 0.10, "volatility": 0.55},
}

# Number of simulated trading days per ticker.
# Rather than one long GBM run (which drifts to unrealistic price ranges over
# 80+ simulated years), we simulate a realistic 3-year window per ticker
# (≈ 756 trading days) and repeat in multiple independent runs.
#
# Windows per single 756-day run  = 756 - WINDOW_LEN + 1 = 722
# Runs needed per ticker          = ceil(20,000 / 722)    = 28
# Total windows per ticker        = 28 × 722              = 20,216  ≥ 20,000 ✓
TRADING_DAYS_PER_RUN  = 756    # ≈ 3 calendar years of trading days
WINDOWS_PER_RUN       = TRADING_DAYS_PER_RUN - WINDOW_LEN + 1   # 722
WINDOWS_PER_TICKER    = TARGET_RECORDS // len(TICKERS)           # 20,000
RUNS_PER_TICKER       = -(-WINDOWS_PER_TICKER // WINDOWS_PER_RUN)  # ceil → 28

# Output file paths (same directory as this script).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(SCRIPT_DIR, "seq-to-seq-train.csv")
VAL_FILE   = os.path.join(SCRIPT_DIR, "seq-to-seq-val.csv")

# Reproducible random seed — same prices generated on every run.
RANDOM_SEED = 42

# Train / validation split ratio.
TRAIN_RATIO = 0.80

# ---------------------------------------------------------------------------
# STEP 1 — Geometric Brownian Motion (GBM) price simulator
# ---------------------------------------------------------------------------
# GBM is the standard model for equity price simulation used in quantitative
# finance (Black-Scholes). Each daily return is drawn from a log-normal
# distribution parameterised by the annualised drift (μ) and volatility (σ).
#
# Daily parameters derived from annual values:
#   daily_drift = (μ - 0.5 × σ²) / 252        (Itô correction term)
#   daily_vol   = σ / sqrt(252)
#
# Price update rule:
#   P(t+1) = P(t) × exp(daily_drift + daily_vol × Z)
#   where Z ~ N(0,1) is a standard normal random variate.
#
# The Itô correction (−0.5 × σ²) ensures that the *expected* price grows at
# the true drift μ rather than at μ + 0.5σ² (which would be the naïve result
# of a straight log-normal without the correction term).

def simulate_gbm_prices(
    start_price: float,
    annual_drift: float,
    annual_vol: float,
    num_days: int,
    rng: random.Random,
) -> list:
    """
    Simulate `num_days` daily closing prices using Geometric Brownian Motion.

    Parameters
    ----------
    start_price  : Initial closing price on day 0 (USD).
    annual_drift : Annualised expected return (e.g. 0.12 = 12 % per year).
    annual_vol   : Annualised volatility / standard deviation of returns
                   (e.g. 0.28 = 28 % per year).
    num_days     : Total number of daily prices to generate (including day 0).
    rng          : A seeded random.Random instance for reproducibility.

    Returns
    -------
    List of `num_days` float closing prices starting at `start_price`.
    """
    # Convert annual parameters to daily equivalents.
    # 252 = standard number of trading days in a calendar year.
    daily_drift = (annual_drift - 0.5 * annual_vol ** 2) / 252
    daily_vol   = annual_vol / math.sqrt(252)

    prices = [start_price]
    for _ in range(num_days - 1):
        # Draw a standard normal random variate using the Box-Muller transform
        # via Python's built-in random.gauss(mu=0, sigma=1).
        z = rng.gauss(0.0, 1.0)
        # Apply the GBM update rule: P(t+1) = P(t) * exp(drift + vol * Z)
        next_price = prices[-1] * math.exp(daily_drift + daily_vol * z)
        prices.append(round(next_price, 4))  # 4 d.p. matches real tick data
    return prices


# ---------------------------------------------------------------------------
# STEP 2 — Generate sliding-window records from all tickers
# ---------------------------------------------------------------------------
# For each ticker we run RUNS_PER_TICKER independent 3-year GBM simulations.
# Each run produces WINDOWS_PER_RUN sliding windows.  We stop each ticker
# once we reach exactly WINDOWS_PER_TICKER windows (20,000).
#
# Each position i within a run produces one record:
#   source = prices[i : i + SOURCE_LEN]               (30 prices — encoder input)
#   target = prices[i + SOURCE_LEN : i + WINDOW_LEN]  ( 5 prices — decoder target)
# The record is the concatenation [source | target] — a flat list of 35 floats.

print("=" * 65)
print("  seq-to-seq-data-csv.py — Stock Price Dataset Generator (CSV)")
print("=" * 65)
print(f"\nConfiguration:")
print(f"  Source window length : {SOURCE_LEN} trading days")
print(f"  Target window length : {TARGET_LEN} trading days")
print(f"  Tickers              : {', '.join(TICKERS.keys())}")
print(f"  Days per run         : {TRADING_DAYS_PER_RUN}  (≈ 3 years)")
print(f"  Runs per ticker      : {RUNS_PER_TICKER}")
print(f"  Windows per run      : {WINDOWS_PER_RUN}")
print(f"  Target records       : {TARGET_RECORDS:,}")
print(f"  Random seed          : {RANDOM_SEED}")

rng     = random.Random(RANDOM_SEED)
records = []   # Will hold all 35-float rows across all tickers

print(f"\nSimulating GBM prices and extracting sliding windows ...")
for ticker, params in TICKERS.items():
    ticker_windows = 0

    for run_idx in range(RUNS_PER_TICKER):
        # Each run starts from the ticker's nominal start price — prices stay
        # in a realistic range.  Deterministic per-run seed for reproducibility.
        run_rng = random.Random(RANDOM_SEED + hash(ticker) + run_idx)

        prices = simulate_gbm_prices(
            start_price  = params["start"],
            annual_drift = params["drift"],
            annual_vol   = params["volatility"],
            num_days     = TRADING_DAYS_PER_RUN,
            rng          = run_rng,
        )

        for i in range(len(prices) - WINDOW_LEN + 1):
            if ticker_windows >= WINDOWS_PER_TICKER:
                break
            source = prices[i : i + SOURCE_LEN]              # 30 prices
            target = prices[i + SOURCE_LEN : i + WINDOW_LEN] #  5 prices
            records.append(source + target)                   # 35 floats
            ticker_windows += 1

        if ticker_windows >= WINDOWS_PER_TICKER:
            break

    print(f"  {ticker:<6}  start=${params['start']:>7.2f}  "
          f"drift={params['drift']:.0%}  vol={params['volatility']:.0%}  "
          f"runs={run_idx+1}  windows={ticker_windows:,}")

print(f"\n  Total records generated : {len(records):,}")


# ---------------------------------------------------------------------------
# STEP 3 — Shuffle and split 80 % train / 20 % validation
# ---------------------------------------------------------------------------
# Shuffle so that windows from different tickers and different time periods
# are interleaved — prevents the model from seeing sequential price runs
# from a single ticker in every mini-batch.

rng.shuffle(records)

split_index  = int(len(records) * TRAIN_RATIO)
train_records = records[:split_index]
val_records   = records[split_index:]

print(f"\nSplit (80 / 20):")
print(f"  Training records   : {len(train_records):,}")
print(f"  Validation records : {len(val_records):,}")


# ---------------------------------------------------------------------------
# STEP 4 — Write CSV files
# ---------------------------------------------------------------------------
# Format rules (consistent with the project's XGBoost and SageMaker CSV
# conventions established in xg-boost-regression.py):
#
#   • NO header row — all rows are purely numeric floating-point values.
#   • 35 columns per row, separated by commas.
#   • Column layout:
#       Columns  0-29  → source closing prices (encoder input)
#       Columns 30-34  → target closing prices (decoder ground truth)
#   • Values are rounded to 4 decimal places to match real tick-data
#     precision without bloating the file size.

def write_csv(path: str, rows: list) -> None:
    """Write a list of float lists to a headerless CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            # Round each value to 4 decimal places before writing.
            writer.writerow([f"{v:.4f}" for v in row])

print(f"\nWriting CSV files ...")
write_csv(TRAIN_FILE, train_records)
print(f"  Training   → {TRAIN_FILE}")

write_csv(VAL_FILE, val_records)
print(f"  Validation → {VAL_FILE}")


# ---------------------------------------------------------------------------
# STEP 5 — Print final summary
# ---------------------------------------------------------------------------

train_size_mb = os.path.getsize(TRAIN_FILE) / (1024 * 1024)
val_size_mb   = os.path.getsize(VAL_FILE)   / (1024 * 1024)

print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Total records         : {len(records):,}")
print(f"  Training records      : {len(train_records):,}  (80 %)")
print(f"  Validation records    : {len(val_records):,}  (20 %)")
print(f"  Columns per row       : {WINDOW_LEN}  "
      f"({SOURCE_LEN} source + {TARGET_LEN} target)")
print(f"  Training file size    : {train_size_mb:.1f} MB  →  {TRAIN_FILE}")
print(f"  Validation file size  : {val_size_mb:.1f} MB  →  {VAL_FILE}")
print(f"\n  !! REMINDER — CSV IS FOR EXPLORATION / INSPECTION ONLY !!")
print(f"  The SageMaker Seq2Seq container requires RecordIO-protobuf (.rec).")
print(f"  Run seq-to-seq-data-recordio.py to produce the training files.")
print(f"{'=' * 65}\n")

