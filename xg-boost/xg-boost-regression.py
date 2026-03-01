import csv
import math
import os

import boto3

# ---------------------------------------------------------------------------
# SageMaker Python SDK v2.x imports
# ---------------------------------------------------------------------------
# sagemaker.session.Session        — wraps boto3; provides upload_data() to
#                                    push local files to S3 and acts as the
#                                    shared context for all SDK objects.
# sagemaker.estimator.Estimator    — high-level training-job launcher; manages
#                                    CreateTrainingJob, CloudWatch log streaming,
#                                    and exposes model_data after training.
# sagemaker.image_uris.retrieve    — resolves the correct ECR image URI for the
#                                    XGBoost built-in algorithm and region without
#                                    hard-coding AWS account IDs or image tags.
# sagemaker.inputs.TrainingInput   — wraps an S3 URI as a named data channel
#                                    passed to Estimator.fit().
# sagemaker.serializers.CSVSerializer     — encodes a Python list as a CSV
#                                    string before sending to the endpoint.
# sagemaker.deserializers.CSVDeserializer — decodes the CSV response from the
#                                    endpoint back into a Python list.
from sagemaker.session       import Session
from sagemaker.estimator     import Estimator
from sagemaker.image_uris    import retrieve as get_image_uri
from sagemaker.inputs        import TrainingInput
from sagemaker.serializers   import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# S3 bucket where all data and model artefacts will be stored.
# Must already exist in the same region as REGION.
S3_BUCKET = "xgboost-regression-bucket-12345"

# S3 key prefix to organise all files produced by this script.
S3_PREFIX = "xgboost-regression"

# AWS region for all SageMaker and S3 operations.
REGION = "eu-north-1"

# IAM role SageMaker assumes to access S3 and ECR.
# Minimum required policy: AmazonSageMakerFullAccess
ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# EC2 instance type for the training job.
# ml.m5.large (2 vCPU, 8 GiB RAM) is sufficient for 10,000 rows with 5 features.
TRAINING_INSTANCE_TYPE = "ml.m5.large"

# EC2 instance type for the inference endpoint.
# ml.t3.medium (2 vCPU, 4 GiB RAM) is the smallest available and is enough
# for single-record XGBoost inference requests.
INFERENCE_INSTANCE_TYPE = "ml.t3.medium"


# Local paths for the data files (same directory as this script).
DATA_FILE  = "xg-boost-regression-data.csv"
TRAIN_FILE = "xg-boost-regression-data-training.csv"
TEST_FILE  = "xg-boost-regression-data-testing.csv"

# ---------------------------------------------------------------------------
# STEP 1 — Load the house price dataset from CSV
# ---------------------------------------------------------------------------
# The CSV file has NO header row — all rows are numeric data.
# Column layout (positional — no names in the file):
#   Column 0  →  price     — target label: house sale price in USD
#   Column 1  →  sqft      — input feature: total living area in sq ft
#   Column 2  →  bedrooms  — input feature: number of bedrooms
#   Column 3  →  bathrooms — input feature: number of bathrooms
#   Column 4  →  house_age — input feature: age of house in years

print(f"Loading dataset from {DATA_FILE} ...")
rows = []
with open(DATA_FILE, "r", newline="") as f:
    reader = csv.reader(f)
    for line in reader:
        if not line:
            continue
        # Parse each column as float — XGBoost requires all numeric values.
        # Column 0 = price (label), columns 1–4 = features.
        rows.append([float(v) for v in line])

print(f"  Loaded {len(rows)} rows")
print(f"  Column 0 (label)   : price")
print(f"  Columns 1–4 (features): sqft, bedrooms, bathrooms, house_age")

# ---------------------------------------------------------------------------
# STEP 2 — Split 80 % train / 20 % test
# ---------------------------------------------------------------------------
# The split is based on the actual number of rows loaded from the CSV.
# The data is already in randomised order from the original generation step,
# so a sequential slice gives a representative train/test split without
# needing to re-shuffle.
#
# HOW XGBOOST IDENTIFIES THE LABEL vs FEATURES:
#   SageMaker XGBoost does NOT use column names or any explicit parameter in
#   TrainingInput or estimator.fit() to identify which column is the label.
#   The convention is purely positional — enforced by column order in the CSV:
#
#     Column 0  →  label   (price)        ← value to be predicted
#     Column 1  →  feature (sqft)         ┐
#     Column 2  →  feature (bedrooms)     │ input features used for prediction
#     Column 3  →  feature (bathrooms)    │
#     Column 4  →  feature (house_age)    ┘
#
#   This positional contract is enforced by two things:
#     1. NO header row in any CSV file — all files are purely numeric rows.
#        A header row would be parsed as a data row and corrupt training.
#     2. Passing content_type="text/csv" in TrainingInput (STEP 6) so the
#        XGBoost container applies the positional label convention.
#
# SageMaker XGBoost CSV format rules:
#   • NO header row in any file (DATA_FILE, TRAIN_FILE, TEST_FILE)
#   • Label (price) MUST be column 0 — positional contract, no named parameter
#   • All values must be numeric

total_rows  = len(rows)
split_index = int(total_rows * 0.80)
train_rows  = rows[:split_index]    # 80 % — used to fit the XGBoost model
test_rows   = rows[split_index:]    # 20 % — held out for price predictions only

with open(TRAIN_FILE, "w", newline="") as f:
    # No header row — column 0 = price (label), columns 1–4 = features.
    csv.writer(f).writerows(train_rows)

with open(TEST_FILE, "w", newline="") as f:
    # No header row — same positional contract as TRAIN_FILE.
    csv.writer(f).writerows(test_rows)

print(f"  Total rows    : {total_rows}")
print(f"  Training rows : {len(train_rows)}  (80 %)")
print(f"  Test rows     : {len(test_rows)}   (20 %)")

# ---------------------------------------------------------------------------
# STEP 3 — Create a SageMaker Session and upload data to S3
# ---------------------------------------------------------------------------
# Session wraps the boto3 session and is the shared context passed to
# Estimator and Predictor. upload_data() copies a local file to S3 and
# returns its full s3:// URI.

boto3_session = boto3.Session(region_name=REGION)
sm_session    = Session(boto_session=boto3_session)

print("\nUploading data to S3 ...")
s3_train_uri = sm_session.upload_data(
    path=TRAIN_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/train",
)
s3_test_uri = sm_session.upload_data(
    path=TEST_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/test",
)
print(f"  Training data : {s3_train_uri}")
print(f"  Test data     : {s3_test_uri}")

# ---------------------------------------------------------------------------
# STEP 4 — Retrieve the XGBoost container image URI
# ---------------------------------------------------------------------------
# get_image_uri() (sagemaker.image_uris.retrieve) resolves the correct ECR
# image for the XGBoost algorithm in the given region using the SDK's built-in
# region→account mapping. Version "1.7-1" is a stable release that supports
# regression, binary/multi-class classification, and ranking.

container_image_uri = get_image_uri(
    framework="xgboost",
    region=REGION,
    version="1.7-1",    # Stable XGBoost version — full algorithm support
)
print(f"\nXGBoost container image: {container_image_uri}")

# ---------------------------------------------------------------------------
# STEP 5 — Define XGBoost hyperparameters
# ---------------------------------------------------------------------------
# This implementation is for REGRESSION only.
# All hyperparameters below are tuned specifically for predicting a continuous
# target (house price in USD). Classification hyperparameters such as
# "binary:logistic", "multi:softmax", "num_class", and "scale_pos_weight"
# are intentionally excluded — they do not apply to regression tasks.
#
# ── OBJECTIVE ───────────────────────────────────────────────────────────────
# objective (str) — regression options only:
#   "reg:squarederror"  — MSE loss; standard choice for continuous targets ✓
#   "reg:squaredlogerror" — MSLE loss; use when target spans several orders
#                          of magnitude (e.g. prices from $50k to $5M).
#   "reg:pseudohubererror"— Huber loss; more robust to price outliers than MSE.
#   "reg:absoluteerror" — MAE loss; median regression, very robust to outliers.
#
# ── BOOSTING ROUNDS ─────────────────────────────────────────────────────────
# num_round (int, default=100)
#   Number of boosting rounds (trees). early_stopping_rounds auto-tunes this.
#
# ── TREE STRUCTURE ──────────────────────────────────────────────────────────
# max_depth (int, default=6, range [1,∞))
#   Maximum depth of each tree. Deeper = more complex, higher overfit risk.
#   Typical range for regression: 3–8.
#
# min_child_weight (float, default=1)
#   Minimum sum of Hessian weights in a child node. Higher = more conservative.
#   Increase to 5–20 if model overfits on small price subgroups.
#
# gamma (float, default=0, alias min_split_loss)
#   Minimum loss reduction required to split a node. Higher = fewer splits.
#
# ── SAMPLING ────────────────────────────────────────────────────────────────
# subsample (float, default=1, range (0,1])
#   Fraction of training rows sampled per boosting round.
#   Values < 1 add stochasticity and help generalisation.
#
# colsample_bytree (float, default=1, range (0,1])
#   Fraction of features sampled per tree. Reduces inter-tree correlation.
#
# colsample_bylevel (float, default=1, range (0,1])
#   Fraction of features sampled at each depth level of a tree.
#
# colsample_bynode (float, default=1, range (0,1])
#   Fraction of features sampled at each node split.
#
# ── REGULARISATION ──────────────────────────────────────────────────────────
# lambda (float, default=1, alias reg_lambda)
#   L2 regularisation on leaf weights. Higher = smoother predictions.
#
# alpha (float, default=0, alias reg_alpha)
#   L1 regularisation on leaf weights. Encourages weight sparsity.
#
# ── LEARNING RATE ───────────────────────────────────────────────────────────
# eta (float, default=0.3, range [0,1], alias learning_rate)
#   Shrinkage per boosting round. Lower eta + more rounds = better generalisation.
#   Typical range: 0.01–0.2 for regression.
#
# ── EARLY STOPPING ──────────────────────────────────────────────────────────
# early_stopping_rounds (int)
#   Stop if validation eval_metric does not improve for N consecutive rounds.
#   Requires a "validation" channel in estimator.fit().
#
# ── EVALUATION METRIC ───────────────────────────────────────────────────────
# eval_metric (str) — regression options only:
#   "rmse"  — root mean squared error  ✓ (default for reg:squarederror)
#   "rmsle" — root mean squared log error
#   "mae"   — mean absolute error
#   "mape"  — mean absolute percentage error
#   "mphe"  — mean Pseudo-Huber error
#
# ── MISC ────────────────────────────────────────────────────────────────────
# seed (int)       — random seed for reproducibility.
# verbosity (int)  — 0=silent, 1=warnings, 2=info, 3=debug.

hyperparameters = {
    # Regression with mean squared error loss
    "objective":             "reg:squarederror",

    # 200 rounds with eta=0.1 is a balanced starting point for 10k rows.
    # early_stopping_rounds will halt training sooner if RMSE plateaus.
    "num_round":             200,

    # Tree depth of 5 avoids overfitting on a 10,000-row dataset.
    "max_depth":             5,

    # Learning rate — 0.1 is conservative enough for stable convergence.
    "eta":                   0.1,

    # Sample 80 % of rows per round — adds stochasticity to reduce overfitting.
    "subsample":             0.8,

    # Sample 80 % of features per tree — decorrelates individual trees.
    "colsample_bytree":      0.8,

    # L2 regularisation — mild smoothing on leaf weights (default = 1).
    "lambda":                1.0,

    # L1 regularisation — disabled; all 4 features are informative.
    "alpha":                 0.0,

    # Minimum child weight — default; increase to 5–10 if overfitting occurs.
    "min_child_weight":      1,

    # Minimum split loss — 0 means split freely at every node.
    "gamma":                 0.0,

    # Stop if validation RMSE does not improve for 10 consecutive rounds.
    # Requires the "validation" channel passed to estimator.fit().
    "early_stopping_rounds": 10,

    # Evaluate RMSE on the validation set after each boosting round.
    "eval_metric":           "rmse",

    # Fixed seed for reproducible training.
    "seed":                  42,

    # Print warnings only during training.
    "verbosity":             1,
}

# ---------------------------------------------------------------------------
# STEP 6 — Create the Estimator and launch the training job
# ---------------------------------------------------------------------------
# sagemaker.estimator.Estimator is the v2 SDK high-level training API.
# It wraps CreateTrainingJob, streams CloudWatch logs, and populates
# model_data with the S3 URI of the trained model artefact after .fit().
#
# Key constructor parameters:
#   image_uri         — XGBoost ECR image (resolved in STEP 4)
#   role              — IAM role SageMaker assumes to access S3 and ECR
#   instance_type     — ml.m5.large: 2 vCPU, 8 GiB — sufficient for 10k rows
#   instance_count    — 1 for single-node training
#   volume_size       — 10 GiB EBS — ample for a small CSV dataset
#   output_path       — S3 prefix where SageMaker writes model.tar.gz
#   max_run           — hard wall-clock limit (seconds) to prevent runaway cost
#   hyperparameters   — XGBoost config dict (STEP 5)
#   sagemaker_session — shared Session (STEP 3)

estimator = Estimator(
    image_uri=container_image_uri,                          # XGBoost ECR image
    role=ROLE_ARN,                                          # IAM execution role
    instance_type=TRAINING_INSTANCE_TYPE,                   # ml.m5.large for training
    instance_count=1,                                       # Single-node training
    volume_size=10,                                         # 10 GiB EBS volume
    output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output",    # Destination for model.tar.gz
    max_run=3600,                                           # 1-hour hard limit
    hyperparameters=hyperparameters,                        # XGBoost config (STEP 5)
    sagemaker_session=sm_session,                           # Shared session (STEP 3)
)

# TrainingInput wraps the S3 URI as a named data channel.
# "train" maps to /opt/ml/input/data/train/ inside the XGBoost container.
#
# content_type="text/csv" is the key setting that activates the XGBoost
# positional label convention inside the container:
#   • There is NO label_column=, target=, or predict= parameter available
#     in TrainingInput or estimator.fit() to name the label column.
#   • Setting content_type="text/csv" is the ONLY configuration needed —
#     the container then automatically treats:
#       column 0 → price      (label  — what the model learns to predict)
#       column 1 → sqft       ┐
#       column 2 → bedrooms   │ features — inputs the model uses to predict
#       column 3 → bathrooms  │
#       column 4 → house_age  ┘
train_input = TrainingInput(
    s3_data=s3_train_uri,
    content_type="text/csv",    # Activates positional label convention: col 0 = label
)

# "validation" channel enables early_stopping_rounds.
# The container evaluates eval_metric on this data after every boosting round
# and stops training when RMSE stops improving.
# Same positional convention applies — col 0 = price label, cols 1-4 = features.
validation_input = TrainingInput(
    s3_data=s3_test_uri,
    content_type="text/csv",    # Same positional label convention as train channel
)

print("\nStarting XGBoost training job ...")
print("  This may take several minutes. Monitor progress in the SageMaker console.")

# .fit() launches the managed SageMaker training job.
# wait=True  — blocks this process until the job finishes (success or failure).
# logs=True  — streams CloudWatch training logs to the local console in real time.
estimator.fit(
    inputs={
        "train":      train_input,        # Primary training data channel
        "validation": validation_input,   # Validation data for early stopping
    },
    wait=True,
    logs=True,
)
print("\nTraining complete.")

# model_data is populated by the Estimator after .fit() completes.
# It points to: s3://<bucket>/<prefix>/output/<job-name>/output/model.tar.gz
model_artefact_s3 = estimator.model_data
print(f"  Model artefact: {model_artefact_s3}")

# ---------------------------------------------------------------------------
# STEP 7 — Deploy the trained model to a real-time endpoint
# ---------------------------------------------------------------------------
# Estimator.deploy() calls CreateModel, CreateEndpointConfig, and CreateEndpoint
# in sequence, then polls until the endpoint status reaches "InService".
# It returns a Predictor bound to the new endpoint.
#
# CSVSerializer   — converts a Python list [x1,x2,x3,x4,x5] into the CSV
#                   string "x1,x2,x3,x4,x5" sent over HTTPS to the endpoint.
# CSVDeserializer — parses the plain-text predicted value from the container
#                   back into a nested Python list e.g. [["23.4512"]].

endpoint_name = "xgboost-regression-endpoint"

print(f"\nDeploying model to endpoint '{endpoint_name}' (may take 5 – 10 minutes) ...")
predictor = estimator.deploy(
    initial_instance_count=1,               # One serving instance
    instance_type=INFERENCE_INSTANCE_TYPE,  # ml.t3.medium — cheapest inference instance
    endpoint_name=endpoint_name,            # Fixed name for easy reference
    serializer=CSVSerializer(),             # Encodes feature list as CSV string
    deserializer=CSVDeserializer(),         # Decodes predicted value to nested list
)
print(f"  Endpoint '{endpoint_name}' is InService.")

# ---------------------------------------------------------------------------
# STEP 8 — Run predictions on the 20 % holdout test set
# ---------------------------------------------------------------------------
# The XGBoost endpoint accepts CSV rows containing FEATURES ONLY (no label).
# The container returns the predicted float as a plain CSV string.
# CSVDeserializer wraps it in a nested list: [["<value>"]].
# We print every 500th prediction to avoid flooding the console.

print("\nRunning predictions on the test set ...")

actual_labels    = []   # Ground-truth target values from the test split
predicted_labels = []   # Values predicted by the deployed XGBoost model

try:
    for i, row in enumerate(test_rows):
        actual_price = row[0]       # First column is the house price label
        features     = row[1:]      # sqft, bedrooms, bathrooms, house_age

        # predictor.predict() serialises features as "sqft,bedrooms,bathrooms,house_age",
        # calls invoke_endpoint(), and deserialises the CSV response.
        result          = predictor.predict(features)
        predicted_score = float(result[0][0])   # Extract scalar from [["value"]]

        actual_labels.append(actual_price)
        predicted_labels.append(predicted_score)

        # Print every 500th record and the final record to track progress
        if i % 500 == 0 or i == len(test_rows) - 1:
            sqft      = row[1]
            bedrooms  = int(row[2])
            bathrooms = row[3]
            age       = row[4]
            print(
                f"  [{i+1:>4}/{len(test_rows)}]  "
                f"{sqft:>6.0f} sqft  {bedrooms}bd/{bathrooms}ba  {age:>4.0f}yr  |  "
                f"Actual: ${actual_price:>10,.0f}  |  "
                f"Predicted: ${predicted_score:>10,.0f}  |  "
                f"Error: ${abs(actual_price - predicted_score):>8,.0f}"
            )

    # -----------------------------------------------------------------------
    # Accuracy metrics
    # -----------------------------------------------------------------------
    # Four complementary metrics computed on the full 20 % holdout test set:
    #
    #   RMSE (Root Mean Squared Error)
    #     Penalises large errors more than small ones due to squaring.
    #     For house prices, RMSE < $20,000 is considered good.
    #     Lower is better.
    #
    #   MAE (Mean Absolute Error)
    #     Average absolute dollar difference between actual and predicted price.
    #     More robust to outlier sales than RMSE. Lower is better.
    #
    #   R² (Coefficient of Determination)
    #     Fraction of price variance explained by the 4 features.
    #     1.0 = perfect fit. Reference: Ames Housing R² ~ 0.85–0.92.
    #     Higher is better.
    #
    #   Tolerance Accuracy (within ± $20,000 of actual price)
    #     % of predictions within $20,000 of the actual sale price.
    #     $20,000 is ~8 % of the median $250,000 price — a realistic threshold.
    #     Higher is better.

    n = len(actual_labels)

    # RMSE — root mean squared error
    squared_errors  = [(a - p) ** 2 for a, p in zip(actual_labels, predicted_labels)]
    rmse            = math.sqrt(sum(squared_errors) / n)

    # MAE — mean absolute error
    absolute_errors = [abs(a - p) for a, p in zip(actual_labels, predicted_labels)]
    mae             = sum(absolute_errors) / n

    # R² — coefficient of determination
    mean_actual = sum(actual_labels) / n
    ss_total    = sum((a - mean_actual) ** 2 for a in actual_labels)   # total variance
    ss_residual = sum(squared_errors)                                    # unexplained variance
    r_squared   = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    # Tolerance accuracy — % predictions within ± $20,000 of actual price
    TOLERANCE          = 20000.0    # $20,000 — ~8 % of median house price
    within_tolerance   = sum(1 for e in absolute_errors if e <= TOLERANCE)
    tolerance_accuracy = (within_tolerance / n) * 100

    print("\n" + "=" * 58)
    print("  Model Accuracy Report — 20 % holdout test set")
    print("=" * 58)
    print(f"  Records evaluated      : {n}")
    print(f"  RMSE                   : ${rmse:>12,.2f}  (lower is better)")
    print(f"  MAE                    : ${mae:>12,.2f}  (lower is better)")
    print(f"  R²                     : {r_squared:>13.4f}  (closer to 1.0 is better)")
    print(f"  Tolerance accuracy     : {tolerance_accuracy:>12.1f} %  (predictions within ±${TOLERANCE:,.0f})")
    print("=" * 58)

finally:
    # -----------------------------------------------------------------------
    # STEP 9 — Delete the endpoint to stop incurring charges
    # -----------------------------------------------------------------------
    # Using finally guarantees cleanup even if prediction raises an exception,
    # preventing orphaned endpoints that accrue hourly instance charges.
    #
    # predictor.delete_endpoint(delete_endpoint_config=True) removes:
    #   • the live endpoint              (stops billing immediately)
    #   • the endpoint configuration
    #   • the registered model object
    print("\nCleaning up AWS resources ...")
    try:
        predictor.delete_endpoint(delete_endpoint_config=True)
        print(f"  Deleted endpoint: {endpoint_name}")
    except Exception as exc:
        print(f"  Warning — could not delete endpoint: {exc}")

print("\nDone.")

