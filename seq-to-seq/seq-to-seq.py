import json
import math
import os
import random
import struct

import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve as get_image_uri
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import JSONSerializer
# ---------------------------------------------------------------------------
# SageMaker Python SDK v2.x imports
# ---------------------------------------------------------------------------
# sagemaker.session.Session        — wraps boto3; provides upload_data() to
#                                    push local files to S3 and acts as the
#                                    shared context for all SDK objects
#                                    (Estimator, Predictor, etc.).
# sagemaker.estimator.Estimator    — high-level training-job launcher; manages
#                                    CreateTrainingJob, CloudWatch log streaming,
#                                    and exposes model_data after training.
# sagemaker.image_uris.retrieve    — resolves the correct ECR image URI for the
#                                    Seq2Seq built-in algorithm and region
#                                    without hard-coding AWS account IDs.
# sagemaker.inputs.TrainingInput   — wraps an S3 URI as a named data channel
#                                    passed to Estimator.fit().
# sagemaker.predictor.Predictor    — thin wrapper around invoke_endpoint() that
#                                    handles serialisation and deserialisation.
# sagemaker.serializers.JSONSerializer     — encodes the inference request
#                                    payload as JSON before posting.
# sagemaker.deserializers.JSONDeserializer — decodes the JSON response body
#                                    returned by the Seq2Seq endpoint.
from sagemaker.session import Session

# sagemaker.amazon.record_pb2.Record — the SageMaker protobuf message class
# used to build and read RecordIO-wrapped protobuf records.  Needed here to
# verify the round-trip of the data files before training starts.
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

# S3 bucket where all data and model artefacts will be stored.
# Must already exist in the same region as REGION before running this script.
S3_BUCKET = "seq2seq-stock-price-bucket-123456"
# S3 key prefix to organise all files under a single namespace in the bucket.
S3_PREFIX = "seq2seq-stock-price"
# AWS region for all SageMaker and S3 API calls.
REGION = "us-east-2"
# IAM role that SageMaker assumes to access S3 and pull the ECR image.
# Minimum required policy: AmazonSageMakerFullAccess.
ROLE_ARN = "arn:aws:iam::148469447057:role/service-role/AmazonSageMaker-ExecutionRole-20260315T102297"
# EC2 instance type for the training job.
# The SageMaker Seq2Seq algorithm ONLY supports GPU instances for training —
# CPU instances (ml.m4, ml.m5, ml.c5 etc.) will raise a ValidationException.
#
# GPU options (cheapest → most powerful):
#   ml.p2.xlarge   — 1× K80 GPU, 12 GiB GPU RAM, 61 GiB system RAM  ← used here
#   ml.p3.2xlarge  — 1× V100 GPU, 16 GiB GPU RAM, 61 GiB system RAM
#   ml.p3.8xlarge  — 4× V100 GPU, 64 GiB GPU RAM
#   ml.p3.16xlarge — 8× V100 GPU, 128 GiB GPU RAM
TRAINING_INSTANCE_TYPE = "ml.g5.2xlarge"
# EC2 instance type for the real-time inference endpoint.
# Inference does NOT require GPU — ml.m4.xlarge (4 vCPU, 16 GiB) is sufficient
# and significantly cheaper than keeping a GPU instance running.
INFERENCE_INSTANCE_TYPE = "ml.g5.2xlarge"

# ---------------------------------------------------------------------------
# Sliding-window and vocabulary parameters.
# These MUST match the values used in seq-to-seq-data-recordio.py exactly.
# If you change NUM_BINS there, update VOCAB_SIZE here too.
# ---------------------------------------------------------------------------
# Number of historical closing-price days in the encoder input sequence.
SOURCE_LEN = 30
# Number of future closing-price days the decoder must predict.
TARGET_LEN = 5
# Total window length (source + target).
WINDOW_LEN = SOURCE_LEN + TARGET_LEN
# Number of equal-width price bins used for discretisation.
# Must match NUM_BINS in seq-to-seq-data-recordio.py.
NUM_BINS = 1000
# Special token IDs — must match seq-to-seq-data-recordio.py.
PAD_ID = 0    # <pad>
EOS_ID = 1    # <eos>
BIN_OFFSET = 2    # First price-bin token ID
# vocab_size = NUM_BINS + 2 special tokens (PAD + EOS).
# !! This must be passed as a hyperparameter to the Seq2Seq estimator !!
VOCAB_SIZE = NUM_BINS + BIN_OFFSET   # 1002
# RecordIO frame magic number (fixed by MXNet/SageMaker spec).
RECORDIO_MAGIC = 0xCED7230A
# Valid token ID range — used by validate_recordio_file() for range checks.
TOKEN_MIN = BIN_OFFSET                  # 2    — first price-bin token
TOKEN_MAX = BIN_OFFSET + NUM_BINS - 1   # 1001 — last  price-bin token
# ---------------------------------------------------------------------------
# Single company configuration — Apple Inc. (AAPL)
# ---------------------------------------------------------------------------
TICKER = "AAPL"    # Stock ticker symbol
START_PRICE = 182.00    # Initial closing price on day 0 (USD)
DRIFT = 0.12      # Annualised expected return (12 % p.a.)
VOLATILITY = 0.28      # Annualised volatility (28 % p.a.)
TARGET_RECORDS = 100_000   # Total sliding windows to generate (80k train + 20k val)
TRADING_DAYS_PER_RUN = 756        # Days simulated per GBM run — approx 3 years of trading (252 days/yr × 3)
WINDOWS_PER_RUN = TRADING_DAYS_PER_RUN - WINDOW_LEN + 1  # Windows extractable from one run: 756 - 35 + 1 = 722
RUNS_NEEDED = -(-TARGET_RECORDS // WINDOWS_PER_RUN)   # GBM runs required to hit 100k: ceil(100,000 / 722) = 139
RANDOM_SEED = 42         # Fixed seed so GBM simulations are reproducible across runs
TRAIN_RATIO = 0.80       # 80% of windows → training set (80,000), remaining 20% → validation (20,000)
# Local file paths for the RecordIO data files and vocab (same directory).
TRAIN_FILE = "seq-to-seq-train.rec"
VAL_FILE = "seq-to-seq-val.rec"
VOCAB_FILE = "seq-to-seq-vocab.json"

# ===========================================================================
# DATA GENERATION FUNCTIONS
# (Imported from seq-to-seq-data-recordio.py — kept here so seq-to-seq.py is
#  fully self-contained and can regenerate data if the .rec files are absent.)
# ===========================================================================
# ---------------------------------------------------------------------------
# GBM price simulator
# ---------------------------------------------------------------------------
# Geometric Brownian Motion — the standard equity price model (Black-Scholes).
# Each day's return is drawn from a log-normal distribution parameterised by
# annualised drift (μ) and volatility (σ):
#
#   daily_drift = (μ - 0.5σ²) / 252      ← Itô correction avoids upward bias
#   daily_vol   = σ / sqrt(252)
#   P(t+1)      = P(t) × exp(daily_drift + daily_vol × Z),  Z ~ N(0,1)

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
    daily_vol = annual_vol / math.sqrt(252)
    prices = [start_price]
    for _ in range(num_days - 1):
        z = rng.gauss(0.0, 1.0)
        next_price = prices[-1] * math.exp(daily_drift + daily_vol * z)
        prices.append(round(next_price, 4))
    return prices


# ---------------------------------------------------------------------------
# Sliding-window record generator
# ---------------------------------------------------------------------------
def generate_raw_records(rng_seed: int) -> tuple:
    """
    Simulate GBM price series for AAPL and extract 35-day sliding windows
    (30 source + 5 target) across RUNS_NEEDED independent runs until
    TARGET_RECORDS windows are produced.

    Returns
    -------
    raw_records : list of (source_prices[30], target_prices[5]) float tuples.
    all_prices  : flat list of every simulated price (used for bin bounds).
    """
    raw_records = []
    all_prices = []
    total_windows = 0
    run_idx = 0   # initialised here so the final print is always safe

    for run_idx in range(RUNS_NEEDED):
        if total_windows >= TARGET_RECORDS:
            break
        # Each run uses a unique seed derived from the base seed + run index
        # so every 3-year simulation is an independent GBM path.
        run_rng = random.Random(rng_seed + run_idx)
        prices = simulate_gbm_prices(
            start_price=START_PRICE,
            annual_drift=DRIFT,
            annual_vol=VOLATILITY,
            num_days=TRADING_DAYS_PER_RUN,
            rng=run_rng,
        )
        all_prices.extend(prices)

        for i in range(len(prices) - WINDOW_LEN + 1):
            if total_windows >= TARGET_RECORDS:
                break
            raw_records.append((
                prices[i: i + SOURCE_LEN],
                prices[i + SOURCE_LEN: i + WINDOW_LEN],
            ))
            total_windows += 1

        if (run_idx + 1) % 20 == 0 or run_idx == RUNS_NEEDED - 1:
            print(f"    Run {run_idx + 1:>3}/{RUNS_NEEDED}  windows so far: {total_windows:,}")

    print(f"\n  {TICKER}  start=${START_PRICE:.2f}  drift={DRIFT:.0%}  "
          f"vol={VOLATILITY:.0%}  runs={run_idx + 1}  windows={total_windows:,}")

    return raw_records, all_prices


# ---------------------------------------------------------------------------
# Price → token ID discretisation
# ---------------------------------------------------------------------------

def build_discretiser(all_prices: list) -> tuple:
    """
    Compute global bin bounds from all simulated prices and return a
    (price_to_token, vocab, price_min, bin_width) tuple.
    Token ID layout:
        0          → <pad>  (padding)
        1          → <eos>  (end-of-sequence)
        2 … 1001   → price bin 0 … 999  (when NUM_BINS = 1000)
    """
    price_min = min(all_prices)
    price_max = max(all_prices)
    bin_width = (price_max - price_min) / NUM_BINS

    def price_to_token(price: float) -> int:
        """Map a float closing price to its integer token ID."""
        bin_index = int((price - price_min) / bin_width)
        bin_index = min(bin_index, NUM_BINS - 1)   # clamp for price_max edge
        return bin_index + BIN_OFFSET
    # Vocabulary: token_id (str) → midpoint dollar value
    vocab = {str(PAD_ID): "<pad>", str(EOS_ID): "<eos>"}
    for i in range(NUM_BINS):
        vocab[str(i + BIN_OFFSET)] = round(price_min + (i + 0.5) * bin_width, 4)

    return price_to_token, vocab, price_min, bin_width


# ---------------------------------------------------------------------------
# Protobuf Record builder  (Layer 2)
# ---------------------------------------------------------------------------

def build_proto_record(source_ids: list, target_ids: list) -> bytes:
    """
    Pack source and target token-ID sequences into a SageMaker protobuf Record
    and return the serialised wire-format bytes.

    The Record fields populated are:
        record.features["source_ids"].int32_tensor.values  ← SOURCE_LEN ints
        record.features["target_ids"].int32_tensor.values  ← TARGET_LEN  ints

    These field names are exactly what the SageMaker Seq2Seq container reads
    from its train / validation RecordIO channels.
    """
    record = Record()
    record.features["source_ids"].int32_tensor.values.extend(source_ids)
    record.features["target_ids"].int32_tensor.values.extend(target_ids)
    return record.SerializeToString()


# ---------------------------------------------------------------------------
# RecordIO frame writer  (Layer 3)
# ---------------------------------------------------------------------------

def write_recordio_file(path: str, token_records: list) -> int:
    """
    Write a list of (source_ids, target_ids) tuples to a binary RecordIO file.

    Each record is serialised through two layers:
      Layer 2 : build_proto_record()  → raw protobuf bytes
      Layer 3 : 8-byte RecordIO header (magic + length) prepended to each blob

    RecordIO frame layout (MXNet / SageMaker spec — LITTLE-ENDIAN)
    ─────────────────────
      Bytes 0–3 : 0xCED7230A  magic number   (little-endian uint32)
      Bytes 4–7 : raw payload length in bytes (little-endian uint32)
      Bytes 8–N : serialised protobuf payload
      Bytes N+1…: zero-padding to align the NEXT frame to a 4-byte boundary

    Parameters
    ----------
    path         : Output .rec file path.
    token_records: List of (source_ids[SOURCE_LEN], target_ids[TARGET_LEN]).

    Returns
    -------
    Total bytes written to the file.
    """
    total_bytes = 0
    with open(path, "wb") as f:
        for i, (source_ids, target_ids) in enumerate(token_records):
            proto_bytes = build_proto_record(source_ids, target_ids)
            length = len(proto_bytes)
            # Header: magic + raw payload length — LITTLE-ENDIAN (MXNet spec)
            header = struct.pack("<II", RECORDIO_MAGIC, length)
            f.write(header)
            f.write(proto_bytes)
            # Pad to 4-byte boundary so the next frame starts aligned
            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.write(b"\x00" * pad_len)
            total_bytes += 8 + length + pad_len
            if (i + 1) % 10_000 == 0:
                print(f"      Written {i + 1:>7,} / {len(token_records):,} frames ...")
    return total_bytes


# ---------------------------------------------------------------------------
# RecordIO frame reader (used for verification and inference decoding)
# ---------------------------------------------------------------------------

def token_to_price_factory(price_min: float, bin_width: float):
    """Return a closure that decodes a token ID back to its midpoint price."""
    def token_to_price(token_id: int) -> float:
        bin_index = token_id - BIN_OFFSET
        return round(price_min + (bin_index + 0.5) * bin_width, 4)
    return token_to_price


# ---------------------------------------------------------------------------
# RecordIO format validator
# ---------------------------------------------------------------------------
# Mirrors the validate_recordio_file() in seq-to-seq-data-recordio.py.
# Called after regenerating files (else block) to confirm the output is
# correct before uploading to S3.
# ---------------------------------------------------------------------------

def validate_recordio_file(path: str, expected_records: int, label: str) -> bool:
    """
    Scan every RecordIO frame in *path* and run 8 format checks.

    Checks
    ------
    1. Magic number == 0xCED7230A  (little-endian)
    2. Payload length > 0
    3. Payload not truncated  (len(payload) == length)
    4. Payload parses as a valid protobuf Record
    5. source_ids length == SOURCE_LEN  (30)
    6. target_ids length == TARGET_LEN  (5)
    7. All token IDs within [TOKEN_MIN, TOKEN_MAX]  ([2, 1001])
    8. Total frame count == expected_records

    Returns
    -------
    True if all checks pass, False otherwise.
    """
    print(f"\n  Validating {label} file : {path}")
    errors   = []
    warnings = []
    frame_idx = 0

    file_size = os.path.getsize(path)
    print(f"    File size        : {file_size / 1024 / 1024:.2f} MB  ({file_size:,} bytes)")

    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) == 0:
                break                           # clean EOF
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
                break

            # Check 2 — non-zero length
            if length == 0:
                errors.append(f"Frame {frame_idx}: payload length is 0")
                break

            payload = f.read(length)

            # Check 3 — no truncation
            if len(payload) < length:
                errors.append(
                    f"Frame {frame_idx}: truncated payload — "
                    f"got {len(payload)} bytes, expected {length}"
                )
                break

            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.read(pad_len)

            # Check 4 — protobuf parseable
            try:
                rec = Record()
                rec.ParseFromString(payload)
            except Exception as exc:
                errors.append(f"Frame {frame_idx}: protobuf ParseFromString failed — {exc}")
                frame_idx += 1
                continue

            src_ids = list(rec.features["source_ids"].int32_tensor.values)
            tgt_ids = list(rec.features["target_ids"].int32_tensor.values)

            # Check 5 — source sequence length
            if len(src_ids) != SOURCE_LEN:
                errors.append(f"Frame {frame_idx}: source_ids length {len(src_ids)} ≠ {SOURCE_LEN}")

            # Check 6 — target sequence length
            if len(tgt_ids) != TARGET_LEN:
                errors.append(f"Frame {frame_idx}: target_ids length {len(tgt_ids)} ≠ {TARGET_LEN}")

            # Check 7 — token ID range
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

            if len(errors) >= 10:
                warnings.append("Stopped after 10 errors — fix and re-run to see more.")
                break

    # Check 8 — total frame count
    if frame_idx != expected_records:
        errors.append(
            f"Frame count mismatch: found {frame_idx:,}, expected {expected_records:,}"
        )

    print(f"    Frames scanned   : {frame_idx:,}")
    if errors:
        print(f"    Status           : ✗ FAILED ({len(errors)} error(s))")
        for err in errors:
            print(f"      ✗ {err}")
        for wrn in warnings:
            print(f"      ⚠ {wrn}")
        return False

    print(f"    Status           : ✓ ALL CHECKS PASSED")
    return True


# ===========================================================================
# MAIN SCRIPT — SageMaker Seq2Seq training and inference
# ===========================================================================

print("=" * 70)
print("  seq-to-seq.py — Stock Price Prediction with SageMaker Seq2Seq")
print("=" * 70)
print(f"\n  S3 bucket      : {S3_BUCKET}")
print(f"  S3 prefix      : {S3_PREFIX}")
print(f"  Region         : {REGION}")
print(f"  Train instance : {TRAINING_INSTANCE_TYPE}")
print(f"  Infer instance : {INFERENCE_INSTANCE_TYPE}")
print(f"  vocab_size     : {VOCAB_SIZE}  (NUM_BINS={NUM_BINS} + 2 special tokens)")
print(f"  source_len     : {SOURCE_LEN} days  |  target_len : {TARGET_LEN} days")

# ---------------------------------------------------------------------------
# STEP 1 — Generate (or reuse) the RecordIO data files
# ---------------------------------------------------------------------------
# If the .rec files already exist on disk (produced by a previous run of
# seq-to-seq-data-recordio.py), they are reused directly — no regeneration.
# If they are absent, this script regenerates them inline using the same
# data-generation functions defined above.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1 — Preparing RecordIO data files")
print("=" * 70)
files_exist = (
    os.path.exists(TRAIN_FILE) and
    os.path.exists(VAL_FILE) and
    os.path.exists(VOCAB_FILE)
)

# ---------------------------------------------------------------------------
# Pre-validation: if the files are on disk, quickly confirm the first frame
# is readable with the correct little-endian magic and valid protobuf bytes.
# This runs BEFORE the if/else so that files_exist can be set to False here
# and the else branch below handles both "files absent" and "files corrupt"
# without needing a second condition.
# ---------------------------------------------------------------------------
if files_exist:
    try:
        with open(VAL_FILE, "rb") as f:
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ValueError("VAL_FILE too short to contain a RecordIO header")
            magic, length = struct.unpack("<II", hdr)
            if magic != RECORDIO_MAGIC:
                raise ValueError(
                    f"Bad RecordIO magic: {magic:#010x} (expected {RECORDIO_MAGIC:#010x}). "
                    f"File may have been written with big-endian byte order."
                )
            payload = f.read(length)
            if len(payload) < length:
                raise ValueError(f"Truncated payload: got {len(payload)} bytes, expected {length}")
            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.read(pad_len)
        probe = Record()
        probe.ParseFromString(payload)
        if not probe.features["source_ids"].int32_tensor.values:
            raise ValueError("First record has an empty source sequence — file may be corrupt")
    except Exception as e:
        print(f"\n  !! Existing .rec files are invalid or incompatible: {e}")
        print(f"  !! Deleting and regenerating from scratch ...\n")
        for bad_file in (TRAIN_FILE, VAL_FILE, VOCAB_FILE):
            if os.path.exists(bad_file):
                os.remove(bad_file)
        files_exist = False

if files_exist:
    # ---- Reuse existing files -------------------------------------------
    print(f"  Existing files found — reusing:")
    print(f"    {TRAIN_FILE}  ({os.path.getsize(TRAIN_FILE) / 1024 / 1024:.1f} MB)")
    print(f"    {VAL_FILE}    ({os.path.getsize(VAL_FILE) / 1024 / 1024:.1f} MB)")
    print(f"    {VOCAB_FILE}")

    # Lazily reconstruct price_min and bin_width from the vocab file.
    target_keys = {str(BIN_OFFSET), str(BIN_OFFSET + 1)}
    lazy_vals = {}
    with open(VOCAB_FILE, "r") as f:
        buf = ""
        decoder = json.JSONDecoder()
        for chunk in iter(lambda: f.read(256), ""):
            buf += chunk
            while True:
                buf = buf.lstrip(" \t\n\r{,")
                if not buf or buf.startswith("}"):
                    break
                try:
                    key, idx = decoder.raw_decode(buf)
                    rest = buf[idx:].lstrip(" \t\n\r")
                    if not rest.startswith(":"):
                        break
                    rest = rest[1:].lstrip(" \t\n\r")
                    val, vidx = decoder.raw_decode(rest)
                    buf = rest[vidx:]
                    if key in target_keys:
                        lazy_vals[key] = float(val)
                    if len(lazy_vals) == len(target_keys):
                        break
                except json.JSONDecodeError:
                    break
            if len(lazy_vals) == len(target_keys):
                break

    mid_0 = lazy_vals[str(BIN_OFFSET)]
    mid_1 = lazy_vals[str(BIN_OFFSET + 1)]
    bin_width = mid_1 - mid_0
    price_min = mid_0 - 0.5 * bin_width

    # Read the first frame from the validation file for the sample record.
    with open(VAL_FILE, "rb") as f:
        hdr = f.read(8)
        magic, length = struct.unpack("<II", hdr)
        payload = f.read(length)
        pad_len = (4 - length % 4) % 4
        if pad_len:
            f.read(pad_len)

    sample_record = Record()
    sample_record.ParseFromString(payload)
    sample_source_ids = list(sample_record.features["source_ids"].int32_tensor.values)
    sample_target_ids = list(sample_record.features["target_ids"].int32_tensor.values)

    print(f"  Sample source IDs (first 5): {sample_source_ids[:5]} ...")
    print(f"  Sample target IDs          : {sample_target_ids}")

else:
    # ---- Regenerate data files ------------------------------------------
    print("  .rec files not found — regenerating from GBM simulation ...")
    print(f"  Ticker       : {TICKER}  (start=${START_PRICE:.2f}  drift={DRIFT:.0%}  vol={VOLATILITY:.0%})")
    print(f"  Days/run     : {TRADING_DAYS_PER_RUN}  (≈ 3 years)")
    print(f"  Runs needed  : {RUNS_NEEDED}")
    print(f"  Target total : {TARGET_RECORDS:,} records")

    print("\n  Simulating GBM prices ...")
    raw_records, all_prices = generate_raw_records(RANDOM_SEED)
    print(f"\n  Total windows generated : {len(raw_records):,}")

    price_to_token, vocab, price_min, bin_width = build_discretiser(all_prices)
    print(f"  Price range : ${min(all_prices):.4f} – ${max(all_prices):.4f}")
    print(f"  Bin width   : ${bin_width:.4f}")

    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  Vocabulary written → {VOCAB_FILE}")

    token_records = [
        ([price_to_token(p) for p in src], [price_to_token(p) for p in tgt])
        for src, tgt in raw_records
    ]

    random.Random(RANDOM_SEED).shuffle(token_records)
    split = int(len(token_records) * TRAIN_RATIO)
    train_recs = token_records[:split]
    val_recs   = token_records[split:]
    print(f"  Train : {len(train_recs):,}  |  Val : {len(val_recs):,}")

    sample_source_ids = val_recs[0][0]
    sample_target_ids = val_recs[0][1]

    print(f"\n  Writing {TRAIN_FILE} ...")
    t_bytes = write_recordio_file(TRAIN_FILE, train_recs)
    print(f"    → {t_bytes / 1024 / 1024:.1f} MB")

    print(f"\n  Writing {VAL_FILE} ...")
    v_bytes = write_recordio_file(VAL_FILE, val_recs)
    print(f"    → {v_bytes / 1024 / 1024:.1f} MB")

    # ---- Validate the freshly generated files before uploading to S3 ----
    print(f"\n{'=' * 70}")
    print(f"  Format Validation — newly generated files")
    print(f"{'=' * 70}")
    train_ok = validate_recordio_file(TRAIN_FILE, len(train_recs), "train")
    val_ok   = validate_recordio_file(VAL_FILE,   len(val_recs),   "validation")
    if not (train_ok and val_ok):
        raise RuntimeError(
            "Generated .rec files failed format validation — "
            "do NOT upload to S3. Check errors above."
        )
    print(f"\n  Validation : ✓ PASSED — files are safe to upload to S3")

print(f"\n  Data files ready.")

# Build the shared token → price decoder.
token_to_price = token_to_price_factory(price_min, bin_width)


# ---------------------------------------------------------------------------
# STEP 2 — Create a SageMaker Session and upload data to S3
# ---------------------------------------------------------------------------
# Session wraps the boto3 session and is the shared context passed to every
# SageMaker SDK object.  upload_data() copies a local file to S3 and returns
# the full s3:// URI.
#
# The SageMaker Seq2Seq container reads from two named data channels:
#   "train"      — binary RecordIO protobuf file (.rec) with training windows
#                  + vocab.src.json and vocab.trg.json in the SAME prefix
#   "validation" — binary RecordIO protobuf file (.rec) with validation windows
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Uploading data files to S3")
print("=" * 70)

boto3_session = boto3.Session(region_name=REGION)
sm_session = Session(boto_session=boto3_session)

print("  Uploading train.rec ...")
s3_train_uri = sm_session.upload_data(
    path=TRAIN_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/train",
)
print(f"    → {s3_train_uri}")

print("  Uploading val.rec ...")
s3_val_uri = sm_session.upload_data(
    path=VAL_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/validation",
)
print(f"    → {s3_val_uri}")

# The Seq2Seq container expects vocab.src.json and vocab.trg.json to be
# present in a dedicated S3 prefix (input/vocab/) which is passed to the
# training job as a separate "vocab" named channel in estimator.fit().
# The container mounts that channel at /opt/ml/input/data/vocab/ and reads
# the two files from there.  Since source and target use the same
# discretisation, we upload the single vocab file twice under both names.
#
# We use boto3 upload_file() directly so the S3 keys have the exact names
# the container expects — no intermediate rename/copy step required.
print("  Uploading vocab files into the vocab channel prefix (input/vocab/) ...")
s3_client = boto3.client("s3", region_name=REGION)

s3_client.upload_file(
    Filename=VOCAB_FILE,
    Bucket=S3_BUCKET,
    Key=f"{S3_PREFIX}/input/vocab/vocab.src.json",
)
print(f"    → s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab/vocab.src.json")

s3_client.upload_file(
    Filename=VOCAB_FILE,
    Bucket=S3_BUCKET,
    Key=f"{S3_PREFIX}/input/vocab/vocab.trg.json",
)
print(f"    → s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab/vocab.trg.json")


# ---------------------------------------------------------------------------
# STEP 3 — Retrieve the Seq2Seq ECR container image URI
# ---------------------------------------------------------------------------
# get_image_uri() resolves the correct ECR registry account and image tag for
# the named built-in algorithm in the given region, avoiding hard-coded IDs.
# The SageMaker Seq2Seq algorithm uses the "seq2seq" framework name.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — Resolving Seq2Seq container image URI")
print("=" * 70)

seq2seq_image_uri = get_image_uri(
    region=REGION,
    framework="seq2seq",
    version="1",         # Latest stable Seq2Seq container
)
print(f"  Seq2Seq image URI : {seq2seq_image_uri}")


# ---------------------------------------------------------------------------
# STEP 4 — Configure the Seq2Seq Estimator with hyperparameters
# ---------------------------------------------------------------------------
# Hyperparameter names below are taken EXACTLY from the official AWS docs:
# https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq-hyperparameters.html
#
# Only the names listed on that page are accepted by the container.
# Any other name causes:
#   ValidationError: Additional hyperparameters are not allowed (... unexpected)
#
# num_embed_source (int, default=512)
#   Embedding size for source tokens (encoder embedding dimension).
#
# num_embed_target (int, default=512)
#   Embedding size for target tokens (decoder embedding dimension).
#
# encoder_type (str, default="rnn")
#   Encoder architecture. "rnn" = Bahdanau attention-based LSTM.
#   "cnn" = Gehring et al. convolutional encoder.
#
# decoder_type (str, default="rnn")
#   Decoder architecture. "rnn" or "cnn".
#
# num_layers_encoder (int, default=1)
#   Number of layers for the encoder rnn or cnn.
#
# num_layers_decoder (int, default=1)
#   Number of layers for the decoder rnn or cnn.
#
# rnn_num_hidden (int, default=1024)
#   Number of rnn hidden units for BOTH encoder and decoder.
#   MUST be a positive EVEN integer (bidirectional LSTM requirement).
#
# rnn_cell_type (str, default="lstm")
#   Specific rnn cell type. "lstm" or "gru".
#
# rnn_attention_type (str, default="mlp")
#   Attention model. One of: dot, fixed, mlp, bilinear.
#
# rnn_decoder_state_init (str, default="last")
#   How to initialise decoder state from encoder. One of: last, avg, zero.
#
# rnn_residual_connections (bool, default=false)
#   Add residual connections to stacked rnn layers (needs num_layers > 1).
#
# max_seq_len_source (int, default=100)
#   Maximum source sequence length. Sequences longer are truncated.
#   Set to SOURCE_LEN = 30.
#
# max_seq_len_target (int, default=100)
#   Maximum target sequence length. Sequences longer are truncated.
#   Set to TARGET_LEN = 5.
#
# max_num_epochs (int, default=none)
#   Maximum number of epochs. Training stops when this is reached
#   even if validation is still improving.
#
# optimizer_type (str, default="adam")
#   Optimiser. One of: adam, sgd, rmsprop.
#
# learning_rate (float, default=0.0003)
#   Initial learning rate.
#
# weight_init_type (str, default="xavier")
#   Weight initialisation. Either "uniform" or "xavier".
#
# optimized_metric (str, default="perplexity")
#   Metric for early stopping. One of: perplexity, accuracy, bleu.
#
# checkpoint_frequency_num_batches (int, default=1000)
#   Checkpoint and evaluate every N batches.
#
# checkpoint_threshold (int, default=3)
#   Stop training if optimized_metric does not improve for this many
#   consecutive checkpoints.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — Configuring Seq2Seq Estimator and hyperparameters")
print("=" * 70)

hyperparameters = {
    # ── Sequence lengths (must match data generator) ──────────────────────
    "max_seq_len_source": SOURCE_LEN,       # 30 — encoder input length
    "max_seq_len_target": TARGET_LEN,       # 5  — decoder output length

    # ── Embeddings ────────────────────────────────────────────────────────
    "num_embed_source": 512,                # encoder token embedding size
    "num_embed_target": 512,                # decoder token embedding size

    # ── Encoder / decoder architecture ────────────────────────────────────
    "encoder_type": "rnn",                  # rnn (LSTM/GRU) or cnn
    "decoder_type": "rnn",                  # rnn (LSTM/GRU) or cnn
    "num_layers_encoder": 1,                # stacked rnn layers in encoder
    "num_layers_decoder": 1,                # stacked rnn layers in decoder

    # ── RNN-specific settings ─────────────────────────────────────────────
    "rnn_num_hidden": 512,                  # hidden units — MUST be even (biLSTM)
    "rnn_cell_type": "lstm",               # lstm or gru
    "rnn_attention_type": "dot",           # dot | fixed | mlp | bilinear
    "rnn_decoder_state_init": "last",      # how encoder state seeds decoder

    # ── Training settings ─────────────────────────────────────────────────
    "max_num_epochs": 10,                   # stop after 10 full passes
    "learning_rate": 0.0003,               # Adam initial learning rate
    "optimizer_type": "adam",              # adam | sgd | rmsprop
    "weight_init_type": "xavier",          # uniform or xavier

    # ── Early stopping ────────────────────────────────────────────────────
    "optimized_metric": "bleu",            # perplexity | accuracy | bleu
    "checkpoint_frequency_num_batches": 1000,  # evaluate every 1000 batches
    "checkpoint_threshold": 3,             # stop if no improvement for 3 checkpoints
}

for k, v in hyperparameters.items():
    print(f"  {k:<24} : {v}")

# Estimator manages the SageMaker training job lifecycle.
estimator = Estimator(
    image_uri=seq2seq_image_uri,
    role=ROLE_ARN,
    instance_count=1,
    instance_type=TRAINING_INSTANCE_TYPE,
    volume_size=30,               # 30 GiB EBS — ample for the .rec files
    output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output",
    max_run=7200,             # 2-hour hard wall-clock limit
    hyperparameters=hyperparameters,
    sagemaker_session=sm_session,
)


# ---------------------------------------------------------------------------
# STEP 5 — Define the three training data channels and launch the job
# ---------------------------------------------------------------------------
# The Seq2Seq container reads from three S3 channels:
#
#   "train"
#       content_type = "application/x-recordio-protobuf"
#       Binary RecordIO file with 80,000 (source[30], target[5]) windows.
#       Mounted inside the container at /opt/ml/input/data/train/.
#
#   "validation"
#       content_type = "application/x-recordio-protobuf"
#       20,000 held-out windows.  The container evaluates BLEU score on this
#       channel at the end of each epoch and prints it to CloudWatch logs.
#       Mounted at /opt/ml/input/data/validation/.
#
#   "vocab"
#       content_type = "application/json"
#       Contains vocab.src.json and vocab.trg.json in a dedicated S3 prefix
#       (input/vocab/).  Providing this as a named channel means the vocab
#       files are mounted at /opt/ml/input/data/vocab/ and the container
#       picks them up from there — keeping them cleanly separate from the
#       binary .rec training files.
#       NOTE: YES — the vocab channel MUST be passed to fit(); without it
#       the container cannot locate the vocab files and will raise:
#         ClientError: Vocab files are not present in the input directory
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Launching Seq2Seq training job")
print("         (this may take 20–40 minutes on ml.g5.2xlarge GPU)")
print("=" * 70)

train_input = TrainingInput(
    s3_data=f"s3://{S3_BUCKET}/{S3_PREFIX}/input/train",
    content_type="application/x-recordio-protobuf",
)

validation_input = TrainingInput(
    s3_data=s3_val_uri,
    content_type="application/x-recordio-protobuf",
)

# Dedicated vocab channel — points to the input/vocab/ prefix where
# vocab.src.json and vocab.trg.json were uploaded in STEP 2.
vocab_input = TrainingInput(
    s3_data=f"s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab",
    content_type="application/json",
)

print("  Data channels:")
print(f"    train      → s3://{S3_BUCKET}/{S3_PREFIX}/input/train")
print(f"    validation → {s3_val_uri}")
print(f"    vocab      → s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab")
print("\n  Starting training job ...")
print("  Monitor progress in the SageMaker console or CloudWatch Logs.\n")

estimator.fit(
    inputs={
        "train":      train_input,
        "validation": validation_input,
        "vocab":      vocab_input,   # required — container reads vocab.src/trg.json from this channel
    },
    wait=True,    # Block until the job reaches Completed or Failed
    logs=True,    # Stream CloudWatch training logs to the console
)

print(f"\n  Training complete.")
print(f"  Model artefact : {estimator.model_data}")


# ---------------------------------------------------------------------------
# STEP 6 — Deploy a real-time inference endpoint
# ---------------------------------------------------------------------------
# deploy() creates a SageMaker Endpoint Configuration and Endpoint backed by
# the trained model artefact.  The endpoint accepts a JSON payload containing
# a 30-token source sequence and returns the 5 most-likely next-token IDs.
#
# Payload format sent to the endpoint:
#   {
#     "instances": [
#       { "source": [token_id_1, token_id_2, ..., token_id_30] }
#     ]
#   }
#
# Response format returned by the endpoint:
#   {
#     "predictions": [
#       { "score": [...], "target": [tok1, tok2, tok3, tok4, tok5] }
#     ]
#   }
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 — Deploying Seq2Seq inference endpoint")
print("=" * 70)

endpoint_name = f"seq2seq-stock-price"

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type=INFERENCE_INSTANCE_TYPE,
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)
print(f"  Endpoint '{endpoint_name}' is InService.")


# ---------------------------------------------------------------------------
# STEP 7 — Predict next-week (5-day) stock prices using sample records
# ---------------------------------------------------------------------------
# We use the sample source sequence saved from the validation split.
# The source IDs represent 30 historical daily closing prices.
# The endpoint returns 5 predicted token IDs → decoded back to dollar prices.
#
# RMSE is computed by comparing the decoded predicted prices against the
# ground-truth target prices decoded from the ground-truth target token IDs.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 — Running next-week stock price predictions")
print("=" * 70)

# Collect all validation records for batch evaluation.
# Read all frames from the validation .rec file.
print("  Reading validation records from disk for evaluation ...")
val_source_list = []
val_target_list = []

with open(VAL_FILE, "rb") as f:
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        magic, length = struct.unpack("<II", hdr)
        if magic != RECORDIO_MAGIC:
            break
        payload = f.read(length)
        if len(payload) < length:
            break
        # Skip 4-byte boundary padding
        pad_len = (4 - length % 4) % 4
        if pad_len:
            f.read(pad_len)
        rec = Record()
        rec.ParseFromString(payload)
        val_source_list.append(list(rec.features["source_ids"].int32_tensor.values))
        val_target_list.append(list(rec.features["target_ids"].int32_tensor.values))

print(f"  Loaded {len(val_source_list):,} validation records.")

# Run predictions in batches.
# The Seq2Seq endpoint accepts one "instance" per request in this SDK version.
# We score the first 100 records to illustrate accuracy without excessive cost.
EVAL_SAMPLE = min(100, len(val_source_list))
print(f"  Evaluating first {EVAL_SAMPLE} records ...")
print()

predicted_prices_all = []   # List of lists — 5 predicted prices per record
actual_prices_all = []   # List of lists — 5 ground-truth prices per record
squared_errors = []

print(f"  {'Rec':>4}  {'Actual prices (5 days)':>42}  {'Predicted prices (5 days)':>42}  {'RMSE':>8}")
print(f"  {'-' * 4}  {'-' * 42}  {'-' * 42}  {'-' * 8}")

for idx in range(EVAL_SAMPLE):
    source_ids = val_source_list[idx]
    target_ids = val_target_list[idx]

    # Build the JSON payload for the endpoint.
    # "source" must be a list of integer token IDs.
    payload = {"instances": [{"source": source_ids}]}
    response = predictor.predict(payload)

    # The container returns:
    #   {"predictions": [{"score": [...], "target": [id1, id2, id3, id4, id5]}]}
    # "target" contains the greedy-decoded output token IDs.
    pred_ids = response["predictions"][0]["target"]

    # Decode token IDs → midpoint dollar prices.
    pred_prices = [token_to_price(tid) for tid in pred_ids]
    actual_prices = [token_to_price(tid) for tid in target_ids]

    predicted_prices_all.append(pred_prices)
    actual_prices_all.append(actual_prices)

    # Per-record RMSE across the 5 predicted days.
    record_mse = sum((p - a) ** 2 for p, a in zip(pred_prices, actual_prices)) / TARGET_LEN
    record_rmse = math.sqrt(record_mse)
    squared_errors.append(record_mse)

    actual_str = "  ".join(f"${p:>7.2f}" for p in actual_prices)
    pred_str = "  ".join(f"${p:>7.2f}" for p in pred_prices)
    print(f"  {idx + 1:>4}  {actual_str}  {pred_str}  ${record_rmse:>7.2f}")

# Overall RMSE across all evaluated records and all 5 target days.
overall_rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

print()
print(f"  Overall RMSE across {EVAL_SAMPLE} records × {TARGET_LEN} days : ${overall_rmse:.4f}")
print()
print("  Interpretation:")
print(f"    Each predicted price is the midpoint of its token bin (bin width ≈ ${bin_width:.2f}).")
print(f"    The maximum possible discretisation error per price is ±${bin_width / 2:.2f}")
print(f"    (half a bin width).  RMSE above that reflects genuine model error.")


# ---------------------------------------------------------------------------
# STEP 8 — Show a detailed next-week forecast for the last sample record
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8 — Next-week 5-day forecast (sample record)")
print("=" * 70)

last_src = val_source_list[EVAL_SAMPLE - 1]
last_pred = predicted_prices_all[-1]
last_actual = actual_prices_all[-1]

# Decode the 30-day source window for context.
src_prices = [token_to_price(tid) for tid in last_src]

print(f"\n  30-day source window (most recent 10 shown):")
for day, price in enumerate(src_prices[-10:], start=SOURCE_LEN - 9):
    print(f"    Day {day:>2} : ${price:>8.2f}")

print(f"\n  5-day next-week forecast:")
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
for day_name, pred_price, actual_price in zip(days, last_pred, last_actual):
    diff = pred_price - actual_price
    print(f"    {day_name:<10}  Predicted: ${pred_price:>8.2f}   "
          f"Actual: ${actual_price:>8.2f}   Diff: ${diff:>+8.2f}")


# ---------------------------------------------------------------------------
# STEP 9 — Delete the inference endpoint
# ---------------------------------------------------------------------------
# Always delete the endpoint after use to stop per-hour instance charges.
# The trained model artefact in S3 is NOT affected by endpoint deletion and
# can be redeployed at any time with Estimator.deploy() or Model.deploy().
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9 — Deleting inference endpoint")
print("=" * 70)

predictor.delete_endpoint()
print(f"  Endpoint '{endpoint_name}' deleted successfully.")

print("\n" + "=" * 70)
print("  Done.")
print(f"  Model artefact : {estimator.model_data}")
print("=" * 70)
