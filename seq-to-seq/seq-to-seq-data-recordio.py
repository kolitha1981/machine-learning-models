import io
import json
import math
import os
import random
import struct

# ---------------------------------------------------------------------------
# seq-to-seq-data-recordio.py
# ---------------------------------------------------------------------------
# PURPOSE
#   Generate ~100,000 sliding-window stock-price records and write them to
#   two RecordIO-wrapped protobuf binary files (80 % train / 20 % validation).
#   This is the ONLY format accepted by the SageMaker Seq2Seq container.
#
# OUTPUT FILES
#   seq-to-seq-train.rec   — 80 % of records  (upload to S3 "train" channel)
#   seq-to-seq-val.rec     — 20 % of records  (upload to S3 "validation" channel)
#   seq-to-seq-vocab.json  — token-ID → midpoint dollar price mapping
#                            (upload to S3 "vocab" channel)
#
# HOW THE FORMAT WORKS — THREE CONVERSION LAYERS
# ─────────────────────────────────────────────────────────────────────────
#  Layer 1 │ Float prices  ──► integer token IDs
#           │   Continuous closing prices are discretised into NUM_BINS
#           │   equal-width buckets.  Two special tokens are reserved:
#           │     0 = <pad>  (padding — not written in these files)
#           │     1 = <eos>  (end-of-sequence sentinel)
#           │     2 … NUM_BINS+1 = price bin IDs
#           │   vocab_size passed to the Seq2Seq estimator MUST equal
#           │   NUM_BINS + 2  (e.g. 1002 when NUM_BINS = 1000).
#           │
#  Layer 2 │ Token IDs  ──► protobuf Record message
#           │   Each window is packed into a sagemaker Record protobuf:
#           │     record.features["source_ids"].int32_tensor.values  ← 30 IDs
#           │     record.features["target_ids"].int32_tensor.values  ←  5 IDs
#           │   Imported from sagemaker.amazon.record_pb2 (ships with the
#           │   SageMaker Python SDK — no custom .proto file needed).
#           │
#  Layer 3 │ Protobuf blob  ──► RecordIO binary frame
#           │   Every serialised protobuf is prefixed with a fixed 8-byte
#           │   RecordIO header using Python's struct module:
#           │     Bytes 0-3  : magic number  0xCED7230A  (little-endian uint32)
#           │     Bytes 4-7  : payload length in bytes   (little-endian uint32)
#           │     Bytes 8-N  : raw serialised protobuf bytes
#           │     Bytes N+1… : zero-padding to align next frame to 4-byte boundary
#           │   All frames are concatenated into a single binary .rec file.
# ─────────────────────────────────────────────────────────────────────────
#
# DEPENDENCIES (install once before running this script)
#   pip install sagemaker protobuf
#
#   sagemaker  — provides sagemaker.amazon.record_pb2.Record (the protobuf
#                message class) and is also needed for seq-to-seq.py.
#   protobuf   — Google's Protocol Buffers runtime; required by record_pb2.
#
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Imports — sagemaker.amazon.record_pb2 is the key addition vs. the CSV script
# ---------------------------------------------------------------------------

# Record is the SageMaker protobuf message class.
# It ships inside the sagemaker Python package at:
#   sagemaker/amazon/record_pb2.py
# Fields used here:
#   record.features[key].int32_tensor.values  — repeated int32 sequence field
# No custom .proto schema file is needed — the compiled class is included
# with the sagemaker SDK.
try:
    from sagemaker.amazon.record_pb2 import Record
except ImportError:
    raise ImportError(
        "\n\nCould not import sagemaker.amazon.record_pb2.\n"
        "Install the required packages with:\n"
        "    pip install sagemaker protobuf\n"
    )


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Sliding-window dimensions.
# These MUST match the Seq2Seq hyperparameters in seq-to-seq.py:
#   max_seq_len_source = SOURCE_LEN  (30)
#   max_seq_len_target = TARGET_LEN  (5)
SOURCE_LEN = 30      # Encoder input  : 30 historical trading-day closing prices
TARGET_LEN = 5       # Decoder target :  5 future   trading-day closing prices
WINDOW_LEN = SOURCE_LEN + TARGET_LEN   # Total days spanned per record = 35

# Target total records.  See CSV script for the per-ticker maths.
TARGET_RECORDS = 100_000

# Number of equal-width price bins for discretisation (Layer 1).
# Larger NUM_BINS → finer price granularity, larger vocab, more memory.
# Smaller NUM_BINS → coarser predictions but faster training.
# Recommended range: 500 – 2000.
# !! vocab_size in seq-to-seq.py MUST be set to NUM_BINS + 2 !!
NUM_BINS = 1000

# Special token IDs (reserved — do not overlap with price bin IDs).
PAD_ID = 0   # Padding token — used if a sequence is shorter than SOURCE_LEN
EOS_ID = 1   # End-of-sequence sentinel appended after the last real token
# Price bin token IDs run from BIN_OFFSET to BIN_OFFSET + NUM_BINS - 1.
BIN_OFFSET = 2   # First price-bin token ID (= 2, after <pad> and <eos>)

# VOCAB_SIZE = NUM_BINS + BIN_OFFSET = 1000 + 2 = 1002.
# This value must be set as the "vocab_size" hyperparameter in seq-to-seq.py.
VOCAB_SIZE = NUM_BINS + BIN_OFFSET

# RecordIO magic number — identifies each frame as a valid RecordIO record.
# The SageMaker Seq2Seq container validates this header before reading the
# protobuf payload.  Value is fixed by the MXNet/SageMaker RecordIO spec.
RECORDIO_MAGIC = 0xCED7230A

# Five synthetic tickers — same parameters as seq-to-seq-data-csv.py so both
# scripts produce records from the same underlying price universe.
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
# (≈ 756 trading days) and then repeat in multiple independent runs until we
# accumulate enough windows.
#
# Windows per single 756-day run  = 756 - WINDOW_LEN + 1 = 722
# Runs needed per ticker          = ceil(20,000 / 722)    = 28
# Total windows per ticker        = 28 × 722              = 20,216  ≥ 20,000 ✓
#
# Each run uses a freshly re-seeded RNG so the 28 runs are statistically
# independent (different random shock sequences) — not just a copy-paste of
# the same 3-year path.
TRADING_DAYS_PER_RUN  = 756    # ≈ 3 calendar years of trading days
WINDOWS_PER_RUN       = TRADING_DAYS_PER_RUN - WINDOW_LEN + 1   # 722
WINDOWS_PER_TICKER    = TARGET_RECORDS // len(TICKERS)           # 20,000
RUNS_PER_TICKER       = -(-WINDOWS_PER_TICKER // WINDOWS_PER_RUN)  # ceil div → 28

# Output file paths (same directory as this script).
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE  = os.path.join(SCRIPT_DIR, "seq-to-seq-train.rec")
VAL_FILE    = os.path.join(SCRIPT_DIR, "seq-to-seq-val.rec")
VOCAB_FILE  = os.path.join(SCRIPT_DIR, "seq-to-seq-vocab.json")

# Reproducible random seed.
RANDOM_SEED = 42

# Train / validation split ratio.
TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------------
# STEP 1 — Geometric Brownian Motion (GBM) price simulator
# ---------------------------------------------------------------------------
# Identical logic to seq-to-seq-data-csv.py.  See that script for a full
# explanation of the GBM model and the Itô correction term.

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
    annual_drift : Annualised expected return  (e.g. 0.12 = 12 % p.a.).
    annual_vol   : Annualised volatility        (e.g. 0.28 = 28 % p.a.).
    num_days     : Total number of daily prices to generate.
    rng          : Seeded random.Random instance for reproducibility.

    Returns
    -------
    List of `num_days` float closing prices starting at `start_price`.
    """
    daily_drift = (annual_drift - 0.5 * annual_vol ** 2) / 252
    daily_vol   = annual_vol / math.sqrt(252)

    prices = [start_price]
    for _ in range(num_days - 1):
        z          = rng.gauss(0.0, 1.0)
        next_price = prices[-1] * math.exp(daily_drift + daily_vol * z)
        prices.append(round(next_price, 4))
    return prices


# ---------------------------------------------------------------------------
# STEP 2 — Generate sliding-window records from all tickers
# ---------------------------------------------------------------------------
# Each record is stored as a tuple (source_prices, target_prices) where both
# are Python lists of floats.  The discretisation to token IDs happens later
# in STEP 3 once we know the global price min/max across all tickers.

print("=" * 65)
print(" seq-to-seq-data-recordio.py — Stock Price Dataset Generator")
print("                              (RecordIO-wrapped Protobuf)")
print("=" * 65)
print(f"\nConfiguration:")
print(f"  Source window length : {SOURCE_LEN} trading days")
print(f"  Target window length : {TARGET_LEN} trading days")
print(f"  Price bins (vocab)   : {NUM_BINS}  →  vocab_size = {VOCAB_SIZE}")
print(f"  Tickers              : {', '.join(TICKERS.keys())}")
print(f"  Days per run         : {TRADING_DAYS_PER_RUN}  (≈ 3 years)")
print(f"  Runs per ticker      : {RUNS_PER_TICKER}")
print(f"  Windows per run      : {WINDOWS_PER_RUN}")
print(f"  Target records       : {TARGET_RECORDS:,}")
print(f"  Random seed          : {RANDOM_SEED}")

rng        = random.Random(RANDOM_SEED)
# raw_records: list of (source_prices[30], target_prices[5]) float tuples.
raw_records = []
all_prices  = []   # Flat list of every price generated — used for bin bounds

print(f"\nSimulating GBM prices and extracting sliding windows ...")
for ticker, params in TICKERS.items():
    ticker_windows = 0

    for run_idx in range(RUNS_PER_TICKER):
        # Each run starts fresh from the ticker's nominal start price so
        # prices always stay in a realistic range (no 80-year drift).
        # We re-seed with a deterministic per-run seed so results are
        # reproducible while each run is statistically independent.
        run_rng = random.Random(RANDOM_SEED + hash(ticker) + run_idx)

        prices = simulate_gbm_prices(
            start_price  = params["start"],
            annual_drift = params["drift"],
            annual_vol   = params["volatility"],
            num_days     = TRADING_DAYS_PER_RUN,
            rng          = run_rng,
        )
        all_prices.extend(prices)   # Accumulate for global min/max

        for i in range(len(prices) - WINDOW_LEN + 1):
            if ticker_windows >= WINDOWS_PER_TICKER:
                break   # Stop once we have exactly WINDOWS_PER_TICKER windows
            source = prices[i : i + SOURCE_LEN]
            target = prices[i + SOURCE_LEN : i + WINDOW_LEN]
            raw_records.append((source, target))
            ticker_windows += 1

        if ticker_windows >= WINDOWS_PER_TICKER:
            break

    print(f"  {ticker:<6}  start=${params['start']:>7.2f}  "
          f"drift={params['drift']:.0%}  vol={params['volatility']:.0%}  "
          f"runs={run_idx+1}  windows={ticker_windows:,}")

print(f"\n  Total records generated : {len(raw_records):,}")


# ---------------------------------------------------------------------------
# STEP 3 — LAYER 1: Discretise float prices → integer token IDs
# ---------------------------------------------------------------------------
# Why discretisation?
#   The Seq2Seq algorithm is fundamentally a sequence-to-sequence model that
#   was designed for NLP (natural language).  Its encoder and decoder both
#   work over a discrete vocabulary of integer token IDs — not continuous
#   floats.  We therefore map every closing price to the ID of the bin it
#   falls into.
#
# How the bins are constructed:
#   1. Find the global min and max price across ALL tickers and ALL days.
#   2. Divide the [min, max] range into NUM_BINS equal-width intervals.
#   3. Each interval (bin) is assigned a token ID starting at BIN_OFFSET (2).
#      BIN_OFFSET = 2 because IDs 0 (<pad>) and 1 (<eos>) are reserved.
#
# Bin assignment formula for a price p:
#   bin_index = floor((p - price_min) / bin_width)     [0-based]
#   token_id  = min(bin_index, NUM_BINS - 1) + BIN_OFFSET
#
#   The min() clamp ensures that the maximum price (p == price_max) falls
#   into the last bin (NUM_BINS - 1) rather than overflowing to index NUM_BINS.
#
# Vocab JSON:
#   For each token ID we store its midpoint dollar value so that predictions
#   (which return token IDs) can be decoded back to approximate dollar prices.
#   midpoint = price_min + (bin_index + 0.5) × bin_width

price_min = min(all_prices)
price_max = max(all_prices)
bin_width = (price_max - price_min) / NUM_BINS

print(f"\nDiscretising prices into {NUM_BINS} bins ...")
print(f"  Global price range : ${price_min:.4f}  –  ${price_max:.4f}")
print(f"  Bin width          : ${bin_width:.4f}")
print(f"  Token ID range     : {BIN_OFFSET}  –  {BIN_OFFSET + NUM_BINS - 1}")
print(f"  Special tokens     : 0=<pad>  1=<eos>")
print(f"  vocab_size         : {VOCAB_SIZE}  "
      f"← set this as the Seq2Seq 'vocab_size' hyperparameter in seq-to-seq.py")


def price_to_token(price: float) -> int:
    """
    Map a single float closing price to its integer token ID.

    The bin index is computed as:
        bin_index = floor((price - price_min) / bin_width)
    clamped to [0, NUM_BINS - 1] to handle the exact maximum price, then
    shifted by BIN_OFFSET to avoid collision with special tokens 0 and 1.

    Parameters
    ----------
    price : Closing price in USD.

    Returns
    -------
    Integer token ID in the range [BIN_OFFSET, BIN_OFFSET + NUM_BINS - 1].
    """
    bin_index = int((price - price_min) / bin_width)
    bin_index = min(bin_index, NUM_BINS - 1)   # Clamp for price_max
    return bin_index + BIN_OFFSET


# Build and write the vocabulary mapping:
#   token_id (str) → midpoint dollar price (float, 4 d.p.)
vocab = {}
for i in range(NUM_BINS):
    token_id     = i + BIN_OFFSET
    midpoint_usd = price_min + (i + 0.5) * bin_width
    vocab[str(token_id)] = round(midpoint_usd, 4)

# Also document the two special tokens in the vocab file for completeness.
vocab["0"] = "<pad>"
vocab["1"] = "<eos>"

with open(VOCAB_FILE, "w") as f:
    json.dump(vocab, f, indent=2)
print(f"  Vocabulary written → {VOCAB_FILE}")

# Convert all raw (float) records to (token_id) records.
# token_records: list of (source_ids[30], target_ids[5]) int tuples.
token_records = []
for source_prices, target_prices in raw_records:
    source_ids = [price_to_token(p) for p in source_prices]
    target_ids = [price_to_token(p) for p in target_prices]
    token_records.append((source_ids, target_ids))

print(f"  Discretisation complete — {len(token_records):,} records tokenised.")


# ---------------------------------------------------------------------------
# STEP 4 — Shuffle and split 80 % train / 20 % validation
# ---------------------------------------------------------------------------

rng.shuffle(token_records)

split_index   = int(len(token_records) * TRAIN_RATIO)
train_records = token_records[:split_index]
val_records   = token_records[split_index:]

print(f"\nSplit (80 / 20):")
print(f"  Training records   : {len(train_records):,}")
print(f"  Validation records : {len(val_records):,}")


# ---------------------------------------------------------------------------
# STEP 5 — LAYER 2: Pack token IDs into protobuf Record messages
# ---------------------------------------------------------------------------
# The sagemaker.amazon.record_pb2.Record class is a compiled protobuf message
# that is included with the SageMaker Python SDK.  No custom .proto file is
# needed.
#
# Record structure used here:
#
#   record.features["source_ids"].int32_tensor.values
#       A repeated int32 field.  We extend() it with the 30 source token IDs.
#       The Seq2Seq container reads this as the encoder input sequence.
#
#   record.features["target_ids"].int32_tensor.values
#       A repeated int32 field.  We extend() it with the 5 target token IDs.
#       The Seq2Seq container reads this as the decoder target sequence.
#
# The Record is then serialised to raw bytes with record.SerializeToString(),
# which is the standard protobuf wire-format serialisation method.

def build_record(source_ids: list, target_ids: list) -> bytes:
    """
    Serialise one sliding-window record into a protobuf Record message.

    Parameters
    ----------
    source_ids : List of 30 integer token IDs (encoder input).
    target_ids : List of  5 integer token IDs (decoder target).

    Returns
    -------
    Raw protobuf bytes (not yet wrapped in a RecordIO frame).
    """
    record = Record()

    # Populate the source (encoder input) sequence.
    # int32_tensor.values is a protobuf repeated int32 field.
    # extend() appends all items from the list in one call — more efficient
    # than calling append() in a loop for large sequences.
    record.features["source_ids"].int32_tensor.values.extend(source_ids)

    # Populate the target (decoder ground truth) sequence.
    record.features["target_ids"].int32_tensor.values.extend(target_ids)

    # SerializeToString() converts the in-memory protobuf object to the
    # compact binary wire format defined by Google's Protocol Buffers spec.
    # This byte string is what gets framed by the RecordIO header in STEP 6.
    return record.SerializeToString()


# ---------------------------------------------------------------------------
# STEP 6 — LAYER 3: Wrap each protobuf blob in a RecordIO binary frame
# ---------------------------------------------------------------------------
# What is RecordIO?
#   RecordIO is a simple binary container format originally from Apache MXNet.
#   SageMaker adopted it as the standard binary data format for its built-in
#   algorithms (Seq2Seq, KNN, Linear Learner, etc.).
#
# Frame layout (8-byte header + payload + padding):
#   ┌────────────────────────────────────────────────────────────┐
#   │ Bytes 0–3   │ Magic number : 0xCED7230A  (uint32, LE)     │
#   │ Bytes 4–7   │ Payload len  : len(proto_bytes) (uint32, LE)│
#   │ Bytes 8–N   │ Payload      : raw protobuf bytes           │
#   │ Bytes N+1–? │ Zero-padding to align next frame to 4 bytes │
#   └────────────────────────────────────────────────────────────┘
#   LE = little-endian byte order (MXNet / SageMaker spec).
#
# Why struct.pack?
#   Python's struct module packs Python integers into fixed-width binary
#   representations.  "<II" means:
#     <  = little-endian byte order  (REQUIRED by MXNet RecordIO spec)
#     I  = unsigned 32-bit integer (4 bytes) — the magic number
#     I  = unsigned 32-bit integer (4 bytes) — the raw payload length
#
# All frames are written sequentially into a single .rec binary file.
# The Seq2Seq container reads the file start-to-end, using each magic-number
# header to find the start and length of each protobuf record.

def write_recordio_file(path: str, records: list) -> int:
    """
    Serialise a list of (source_ids, target_ids) tuples to a RecordIO file.

    Each record goes through two transformations:
      1. build_record()  → protobuf bytes  (Layer 2)
      2. RecordIO frame  → header + bytes  (Layer 3)

    Parameters
    ----------
    path    : Destination file path for the .rec binary file.
    records : List of (source_ids[30], target_ids[5]) int-list tuples.

    Returns
    -------
    Total number of bytes written to the file.
    """
    total_bytes = 0
    with open(path, "wb") as f:
        for i, (source_ids, target_ids) in enumerate(records):
            # Layer 2 — convert token IDs to a serialised protobuf blob.
            proto_bytes = build_record(source_ids, target_ids)
            length = len(proto_bytes)

            # Layer 3 — build the 8-byte RecordIO header.
            #   struct.pack("<II", magic, length):
            #     "<"  = little-endian  (MXNet / SageMaker RecordIO spec)
            #     "I"  = unsigned 32-bit int
            #   First  I = RECORDIO_MAGIC  (0xCED7230A) — frame start marker
            #   Second I = len(proto_bytes) — raw (unpadded) payload length
            header = struct.pack("<II", RECORDIO_MAGIC, length)

            # Write header immediately followed by the protobuf payload.
            f.write(header)
            f.write(proto_bytes)

            # Pad to 4-byte boundary so the next frame starts aligned.
            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.write(b"\x00" * pad_len)

            total_bytes += 8 + length + pad_len

            # Progress indicator every 10,000 records.
            if (i + 1) % 10_000 == 0:
                print(f"    Written {i + 1:>7,} / {len(records):,} records ...")

    return total_bytes


# ---------------------------------------------------------------------------
# STEP 7 — Write train and validation .rec files
# ---------------------------------------------------------------------------

print(f"\nWriting RecordIO-protobuf files ...")

print(f"  Training file ...")
train_bytes = write_recordio_file(TRAIN_FILE, train_records)
train_mb    = train_bytes / (1024 * 1024)
print(f"  Training   → {TRAIN_FILE}  ({train_mb:.1f} MB)")

print(f"  Validation file ...")
val_bytes = write_recordio_file(VAL_FILE, val_records)
val_mb    = val_bytes / (1024 * 1024)
print(f"  Validation → {VAL_FILE}  ({val_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# STEP 8 — Full format validation of both generated .rec files
# ---------------------------------------------------------------------------
# Scans EVERY frame in both files and checks:
#   ✔ Magic number matches 0xCED7230A (little-endian)
#   ✔ Payload length field is non-zero and consistent with file size
#   ✔ Payload bytes parse as a valid protobuf Record
#   ✔ source_ids length == SOURCE_LEN (30)
#   ✔ target_ids length == TARGET_LEN (5)
#   ✔ All token IDs are within [BIN_OFFSET, BIN_OFFSET + NUM_BINS - 1]
#   ✔ Total frame count matches expected record count
#   ✔ File ends cleanly (no trailing garbage bytes)

TOKEN_MIN = BIN_OFFSET                  # 2
TOKEN_MAX = BIN_OFFSET + NUM_BINS - 1   # 1001


def validate_recordio_file(path: str, expected_records: int, label: str) -> bool:
    """
    Validate every RecordIO frame in *path*.

    Parameters
    ----------
    path             : Path to the .rec file to validate.
    expected_records : Number of frames expected in the file.
    label            : Human-readable label for print output ("train" / "val").

    Returns
    -------
    True if all checks pass, False if any check fails.
    """
    print(f"\n  Validating {label} file : {path}")
    errors   = []
    warnings = []
    frame_idx = 0

    file_size = os.path.getsize(path)
    print(f"    File size        : {file_size / 1024 / 1024:.2f} MB  ({file_size:,} bytes)")

    with open(path, "rb") as f:
        while True:
            # ── Header ──────────────────────────────────────────────────
            hdr = f.read(8)
            if len(hdr) == 0:
                break                       # clean EOF
            if len(hdr) < 8:
                errors.append(f"Frame {frame_idx}: incomplete header — only {len(hdr)} bytes before EOF")
                break

            magic, length = struct.unpack("<II", hdr)

            # Check 1 — magic number
            if magic != RECORDIO_MAGIC:
                errors.append(
                    f"Frame {frame_idx}: bad magic {magic:#010x} "
                    f"(expected {RECORDIO_MAGIC:#010x}) — "
                    f"file may have been written with big-endian byte order"
                )
                break   # cannot continue; frame boundaries are now unknown

            # Check 2 — non-zero length
            if length == 0:
                errors.append(f"Frame {frame_idx}: payload length is 0")
                break

            # ── Payload ─────────────────────────────────────────────────
            payload = f.read(length)
            if len(payload) < length:
                errors.append(
                    f"Frame {frame_idx}: truncated payload — "
                    f"got {len(payload)} bytes, expected {length}"
                )
                break

            # Skip 4-byte boundary padding
            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.read(pad_len)

            # Check 3 — protobuf parse
            try:
                rec = Record()
                rec.ParseFromString(payload)
            except Exception as exc:
                errors.append(f"Frame {frame_idx}: protobuf ParseFromString failed — {exc}")
                frame_idx += 1
                continue

            src_ids = list(rec.features["source_ids"].int32_tensor.values)
            tgt_ids = list(rec.features["target_ids"].int32_tensor.values)

            # Check 4 — source length
            if len(src_ids) != SOURCE_LEN:
                errors.append(
                    f"Frame {frame_idx}: source_ids length {len(src_ids)} ≠ {SOURCE_LEN}"
                )

            # Check 5 — target length
            if len(tgt_ids) != TARGET_LEN:
                errors.append(
                    f"Frame {frame_idx}: target_ids length {len(tgt_ids)} ≠ {TARGET_LEN}"
                )

            # Check 6 — token ID range
            bad_src = [t for t in src_ids if not (TOKEN_MIN <= t <= TOKEN_MAX)]
            bad_tgt = [t for t in tgt_ids if not (TOKEN_MIN <= t <= TOKEN_MAX)]
            if bad_src:
                errors.append(
                    f"Frame {frame_idx}: source token(s) out of range "
                    f"[{TOKEN_MIN},{TOKEN_MAX}]: {bad_src[:5]}"
                )
            if bad_tgt:
                errors.append(
                    f"Frame {frame_idx}: target token(s) out of range "
                    f"[{TOKEN_MIN},{TOKEN_MAX}]: {bad_tgt[:5]}"
                )

            frame_idx += 1

            # Print first record as a sample
            if frame_idx == 1:
                def _tid_to_price(tid):
                    return round(price_min + (tid - BIN_OFFSET + 0.5) * bin_width, 2)
                print(f"    Sample record 1  :")
                print(f"      source_ids  (first 5) : {src_ids[:5]} …")
                print(f"      target_ids            : {tgt_ids}")
                print(f"      source prices (first 5): "
                      f"{[_tid_to_price(t) for t in src_ids[:5]]} …")
                print(f"      target prices         : "
                      f"{[_tid_to_price(t) for t in tgt_ids]}")

            # Stop collecting errors after 10 to avoid flooding output
            if len(errors) >= 10:
                warnings.append("Stopped after 10 errors — fix and re-run to see more.")
                break

    # ── Frame count ─────────────────────────────────────────────────────────
    if frame_idx != expected_records:
        errors.append(
            f"Frame count mismatch: found {frame_idx:,}, expected {expected_records:,}"
        )

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"    Frames scanned   : {frame_idx:,}")
    if errors:
        print(f"    Status           : ✗ FAILED ({len(errors)} error(s))")
        for err in errors:
            print(f"      ✗ {err}")
        for wrn in warnings:
            print(f"      ⚠ {wrn}")
        return False
    else:
        print(f"    Status           : ✓ ALL CHECKS PASSED")
        return True


# Run validation on both files
print(f"\n{'=' * 65}")
print(f"  STEP 8 — Format Validation")
print(f"{'=' * 65}")

train_ok = validate_recordio_file(TRAIN_FILE, len(train_records), "train")
val_ok   = validate_recordio_file(VAL_FILE,   len(val_records),   "validation")

all_ok = train_ok and val_ok
print(f"\n  Overall validation : {'✓ PASSED' if all_ok else '✗ FAILED — do NOT upload to S3'}")


# ---------------------------------------------------------------------------
# STEP 9 — Final summary
# ---------------------------------------------------------------------------

vocab_size_reminder = f"vocab_size = {VOCAB_SIZE}  (= NUM_BINS {NUM_BINS} + 2 special tokens)"

print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Total records         : {len(token_records):,}")
print(f"  Training records      : {len(train_records):,}  (80 %)")
print(f"  Validation records    : {len(val_records):,}  (20 %)")
print(f"  Source sequence len   : {SOURCE_LEN} tokens  (= max_seq_len_source)")
print(f"  Target sequence len   : {TARGET_LEN} tokens  (= max_seq_len_target)")
print(f"  Price bins            : {NUM_BINS}")
print(f"  Training file         : {train_mb:.1f} MB  →  {TRAIN_FILE}")
print(f"  Validation file       : {val_mb:.1f} MB  →  {VAL_FILE}")
print(f"  Vocabulary file       :          →  {VOCAB_FILE}")
print(f"  Validation result     : {'✓ PASSED — safe to upload to S3' if all_ok else '✗ FAILED — fix errors before uploading'}")
if all_ok:
    print(f"\n  !! IMPORTANT — seq-to-seq.py hyperparameter reminder !!")
    print(f"  Set  {vocab_size_reminder}")
    print(f"  Set  max_seq_len_source = {SOURCE_LEN}")
    print(f"  Set  max_seq_len_target = {TARGET_LEN}")
    print(f"\n  Upload channels to S3:")
    print(f"    train      → {TRAIN_FILE}")
    print(f"    validation → {VAL_FILE}")
    print(f"    vocab      → {VOCAB_FILE}")
print(f"{'=' * 65}\n")

