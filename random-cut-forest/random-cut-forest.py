import csv
import math
import os
import random

import boto3

# ---------------------------------------------------------------------------
# SageMaker Python SDK v2.x imports
# ---------------------------------------------------------------------------
# sagemaker.session.Session      — wraps boto3; provides upload_data() to push
#                                  local files to S3 and acts as the shared
#                                  context for all SDK objects (Estimator,
#                                  Predictor, etc.).
# sagemaker.estimator.Estimator  — high-level training-job launcher; manages
#                                  CreateTrainingJob, CloudWatch log streaming,
#                                  and exposes model_data after training.
# sagemaker.image_uris.retrieve  — resolves the correct ECR image URI for the
#                                  Random Cut Forest built-in algorithm and
#                                  region without hard-coding AWS account IDs.
# sagemaker.inputs.TrainingInput — wraps an S3 URI as a named data channel
#                                  passed to Estimator.fit().
# sagemaker.serializers.CSVSerializer       — encodes a Python list as a CSV
#                                  string before sending it to the endpoint.
# sagemaker.deserializers.JSONDeserializer  — decodes the JSON anomaly-score
#                                  response returned by the RCF endpoint.
from sagemaker.session       import Session
from sagemaker.estimator     import Estimator
from sagemaker.image_uris    import retrieve as get_image_uri
from sagemaker.inputs        import TrainingInput
from sagemaker.serializers   import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
# S3 bucket where all data and model artefacts will be stored.
# Must already exist in the same region as REGION before running this script.
S3_BUCKET = "random-cut-forest-bucket-12345"

# S3 key prefix used to organise all files produced by this script under a
# single folder-like namespace within the bucket.
S3_PREFIX = "random-cut-forest"

# AWS region for all SageMaker and S3 API calls.
REGION = "eu-north-1"

# IAM role that SageMaker assumes to access S3 and pull the ECR image.
# The role must have at minimum the AmazonSageMakerFullAccess managed policy.
ROLE_ARN = "arn:aws:iam::148469447057:role/service-role/AmazonSageMaker-ExecutionRole-20260302T181367"

# EC2 instance type for the training job.
# ml.m5.large (2 vCPU, 8 GiB RAM) is more than sufficient for 8,000 training
# rows with 19 features.  Downgrade to ml.t3.medium if cost is a concern.
TRAINING_INSTANCE_TYPE = "ml.m5.large"

# EC2 instance type for the real-time inference endpoint.
# ml.t3.medium (2 vCPU, 4 GiB RAM) is the smallest available SageMaker
# instance type and is enough for single-record RCF inference requests.
INFERENCE_INSTANCE_TYPE = "ml.m5.large"

# Path to the raw CSV data file relative to this script.
# Schema (no header row, 19 features + 1 label column — label is LAST):
#   duration, protocol_type_num, src_bytes, dst_bytes, land,
#   wrong_fragment, urgent, hot, num_failed_logins, logged_in,
#   num_compromised, count, srv_count, serror_rate, rerror_rate,
#   same_srv_rate, diff_srv_rate, dst_host_count, dst_host_srv_count, label
# Dataset is based on the KDD Cup 1999 Network Intrusion Detection benchmark.
# label = 0 → normal traffic,  label = 1 → anomaly / attack traffic.
DATA_FILE  = "random-cut-forest-data.csv"

# Local paths where the 80 % / 20 % splits will be written before upload.
TRAIN_FILE = "random-cut-forest-train.csv"
TEST_FILE  = "random-cut-forest-test.csv"

# Fixed random seed — ensures the same 80/20 split every time the script runs,
# making results reproducible across different machines and runs.
RANDOM_SEED = 42

# Number of feature columns per record (label column is excluded from features).
# Matches the 19 numeric columns in random-cut-forest-data.csv.
NUM_FEATURES = 19

# ---------------------------------------------------------------------------
# Threshold method for converting RCF scores to binary anomaly predictions.
# ---------------------------------------------------------------------------
# RCF returns a continuous anomaly score per record — not a class label.
# A threshold must be chosen to decide which scores are flagged as anomalies.
# Two methods are supported:
#
#   "gaussian"   — threshold = mean + THRESHOLD_K × std
#                  Assumes the score distribution is approximately normal
#                  (Gaussian).  Fast and interpretable.  The 3-sigma rule means
#                  ~0.3 % of normally distributed scores are flagged as outliers.
#                  WARNING: RCF scores are typically RIGHT-SKEWED (long tail on
#                  the high end), so this assumption is often violated.  On a
#                  skewed distribution the threshold can land further right than
#                  intended, causing genuine anomalies to be MISSED (higher false
#                  negative rate).  Use only if you have verified that the score
#                  histogram is roughly bell-shaped.
#
#   "percentile" — threshold = score at the Nth percentile of all test scores.
#                  Makes NO assumption about the shape of the distribution.
#                  Works correctly for skewed, multimodal, or heavy-tailed score
#                  distributions.  ANOMALY_PERCENTILE = 95 means the top 5 % of
#                  scores are flagged as anomalies regardless of their absolute
#                  value or the distribution shape.  Adjust the percentile to
#                  control sensitivity (higher → fewer flags, lower → more flags).
#
# Switch between methods by changing THRESHOLD_METHOD below.
THRESHOLD_METHOD    = "percentile"   # "gaussian" or "percentile"

# Used only when THRESHOLD_METHOD = "gaussian".
# k = 3 follows the 3-sigma rule (flags top ~0.3 % assuming normal distribution).
# Decrease k (e.g. 2) to flag more points; increase (e.g. 4) to flag fewer.
THRESHOLD_K         = 3

# Used only when THRESHOLD_METHOD = "percentile".
# 95 means the top 5 % of scores are flagged as anomalies.
# Tune this to match the known or expected anomaly rate in your data.
# This dataset has ~15 % anomalies, so a value of 85 would be appropriate.
# Set conservatively high (e.g. 95) to reduce false positives.
ANOMALY_PERCENTILE  = 85   # top (100 - ANOMALY_PERCENTILE) % flagged as anomaly

# ---------------------------------------------------------------------------
# STEP 1 — Load and shuffle the raw data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — Loading data from:", DATA_FILE)
print("=" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, DATA_FILE)

with open(data_path, newline="") as f:
    reader = csv.reader(f)
    # Read all non-empty rows and convert each value to float.
    all_rows = [list(map(float, row)) for row in reader if row]

print(f"  Total records loaded : {len(all_rows)}")

# Count class distribution before shuffling.
normal_count  = sum(1 for r in all_rows if r[-1] == 0)
anomaly_count = sum(1 for r in all_rows if r[-1] == 1)
print(f"  Normal  (label=0)    : {normal_count}  ({100*normal_count/len(all_rows):.1f}%)")
print(f"  Anomaly (label=1)    : {anomaly_count}  ({100*anomaly_count/len(all_rows):.1f}%)")

# Shuffle so the 80/20 split is random rather than ordered by label/class,
# which could cause the test set to be dominated by one class.
random.seed(RANDOM_SEED)
random.shuffle(all_rows)

# ---------------------------------------------------------------------------
# STEP 2 — Split 80 % training / 20 % testing
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Splitting data 80 % training / 20 % testing")
print("=" * 70)

# Calculate the index that separates the training set from the test set.
# int() floors the result so any remainder goes to the test set.
split_index  = int(len(all_rows) * 0.80)
train_rows   = all_rows[:split_index]   # 80 % — used to train the model
test_rows    = all_rows[split_index:]   # 20 % — held out for evaluation only

print(f"  Training records : {len(train_rows)}  (80 %)")
print(f"  Test records     : {len(test_rows)}   (20 %)")

# For training, SageMaker RCF expects FEATURES ONLY — no label column.
# Strip the last column (label) before writing the training split.
train_features = [row[:NUM_FEATURES] for row in train_rows]

# Keep both features AND label for the test split so we can measure accuracy.
test_features = [row[:NUM_FEATURES] for row in test_rows]
test_labels   = [int(row[-1]) for row in test_rows]

def write_csv(path: str, data: list) -> None:
    """Write a list of numeric rows to a CSV file (no header)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
train_path = os.path.join(script_dir, TRAIN_FILE)
test_path  = os.path.join(script_dir, TEST_FILE)

write_csv(train_path, train_features)
write_csv(test_path,  test_features)
print(f"  Training split written to : {train_path}")
print(f"  Test split written to     : {test_path}")

# ---------------------------------------------------------------------------
# STEP 3 — Create a SageMaker Session and upload data to S3
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — Uploading data to S3")
print("=" * 70)

# Session wraps the boto3 session and is the shared context passed to every
# SageMaker SDK object (Estimator, Predictor).  upload_data() copies a local
# file to S3 under the given key prefix and returns the full s3:// URI.
boto3_session = boto3.Session(region_name=REGION)
sm_session    = Session(boto_session=boto3_session)

# upload_data() copies the local file to S3 and returns:
#   s3://<bucket>/<key_prefix>/<filename>
s3_train_uri = sm_session.upload_data(
    path=train_path,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/train",
)
print(f"  Training data uploaded : {s3_train_uri}")

# ---------------------------------------------------------------------------
# STEP 4 — Retrieve the RCF ECR image URI
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — Resolving Random Cut Forest container image URI")
print("=" * 70)

# get_image_uri() looks up the ECR image registry, account and tag for the
# named built-in algorithm in the given region.  This avoids hard-coding
# AWS account IDs that differ per region.
rcf_image_uri = get_image_uri(
    region_name=REGION,              # AWS region to resolve the correct ECR account
    framework="randomcutforest",     # built-in algorithm name in SageMaker
    version="1",                     # latest stable version of the RCF container
)
print(f"  RCF image URI : {rcf_image_uri}")

# ---------------------------------------------------------------------------
# STEP 5 — Configure the RCF Estimator with hyperparameters
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Configuring Random Cut Forest Estimator")
print("=" * 70)

# Estimator is the SageMaker SDK class that manages CreateTrainingJob calls.
# It accepts the ECR image, IAM role, instance configuration, and output path.
rcf_estimator = Estimator(
    image_uri          = rcf_image_uri,
    role               = ROLE_ARN,
    instance_count     = 1,
    instance_type      = TRAINING_INSTANCE_TYPE,
    # S3 location where SageMaker writes the trained model artefact (model.tar.gz).
    output_path        = f"s3://{S3_BUCKET}/{S3_PREFIX}/output",
    sagemaker_session  = sm_session,
)

# ---------------------------------------------------------------------------
# Hyperparameters for the Random Cut Forest algorithm
# ---------------------------------------------------------------------------
# num_samples_per_tree (int, default=256, range [1, 2048])
#   Number of data points stored in each tree's reservoir sample.
#   Larger values improve anomaly sensitivity at the cost of more memory.
#   Rule of thumb: sqrt(training_set_size).  For 8000 rows → ~89, rounded to 256.
#
# num_trees (int, default=100, range [1, 1000])
#   Number of Random Cut Trees in the forest ensemble.
#   More trees = smoother, more stable anomaly scores.  100 is a good default.
#   Increasing to 200–500 improves accuracy on noisy datasets.
#
# feature_dim (int, required)
#   Number of features per record.  Must match the number of columns in the
#   training CSV (label column excluded).  Here: 19 numeric network features.
#
# eval_metrics (list of str, default=["accuracy", "precision_recall_fscore"])
#   Metrics computed on the optional validation channel during training.
#   Available: "accuracy", "precision_recall_fscore".
rcf_estimator.set_hyperparameters(
    num_samples_per_tree = 256,         # reservoir sample size per tree
    num_trees            = 100,         # number of trees in the forest ensemble
    feature_dim          = NUM_FEATURES,  # 19 input features (no label)
)

print(f"  num_samples_per_tree : 256")
print(f"  num_trees            : 100")
print(f"  feature_dim          : {NUM_FEATURES}")

# ---------------------------------------------------------------------------
# STEP 6 — Launch the training job
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 — Launching training job (this may take several minutes) ...")
print("=" * 70)

# TrainingInput wraps the S3 URI and tells SageMaker the content type.
# content_type="text/csv;label_size=0" indicates there is NO label column in
# the training CSV — all columns are features.
train_input = TrainingInput(
    s3_data      = s3_train_uri,
    content_type = "text/csv;label_size=0",
)

# fit() calls CreateTrainingJob, streams CloudWatch logs to the console,
# and blocks until the job reaches a terminal state (Completed / Failed).
rcf_estimator.fit({"train": train_input})
print(f"\n  Training job complete.")
print(f"  Model artefact : {rcf_estimator.model_data}")

# ---------------------------------------------------------------------------
# STEP 7 — Deploy a real-time inference endpoint
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 — Deploying inference endpoint ...")
print("=" * 70)

# deploy() creates an Endpoint Configuration and an Endpoint backed by the
# trained model.  It returns a Predictor that can call invoke_endpoint().
#
# CSVSerializer    — converts a Python list of numbers to a comma-separated
#                    string before POSTing to the endpoint.
# JSONDeserializer — parses the JSON response body {"scores":[{"score":X},...]}
#                    returned by the RCF container back into a Python dict.
rcf_predictor = rcf_estimator.deploy(
    initial_instance_count = 1,
    instance_type          = INFERENCE_INSTANCE_TYPE,
    serializer             = CSVSerializer(),
    deserializer           = JSONDeserializer(),
)
print(f"  Endpoint name : {rcf_predictor.endpoint_name}")
# ---------------------------------------------------------------------------
# STEP 8 — Score the test set and derive binary predictions
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8 — Scoring test records and computing anomaly threshold ...")
print("=" * 70)

# Collect the raw anomaly score for every test record.
# The RCF endpoint accepts one or more records per request and returns:
#   {"scores": [{"score": <float>}, ...]}
# Higher scores indicate a higher likelihood of anomaly.
anomaly_scores = []
BATCH_SIZE = 25  # records per HTTP request — keeps payload size manageable

for i in range(0, len(test_features), BATCH_SIZE):
    batch    = test_features[i : i + BATCH_SIZE]
    print(f"Scoring records {i} to {min(i + BATCH_SIZE, len(test_features)) - 1} ...")
    response = rcf_predictor.predict(batch)
    # The response dict has the shape: {"scores": [{"score": X}, ...]}
    for score_obj in response["scores"]:
        anomaly_scores.append(score_obj["score"])

print(f"Scored {len(anomaly_scores)} test records.")

# ---------------------------------------------------------------------------
# Anomaly threshold — two methods supported
# ---------------------------------------------------------------------------

# --- Shared statistics (used by both methods for reporting) ----------------
mean_score = sum(anomaly_scores) / len(anomaly_scores)
variance   = sum((s - mean_score) ** 2 for s in anomaly_scores) / len(anomaly_scores)
std_score  = math.sqrt(variance)

print(f"\n  Score distribution — mean : {mean_score:.4f},  std : {std_score:.4f}")

# --- Method 1 : Gaussian (3-sigma rule) ------------------------------------
# Assumes the anomaly scores are approximately normally distributed.
# threshold = mean + k × std
# On a true normal distribution, only ~0.3 % of scores exceed mean + 3×std,
# so those points are treated as anomalies.
#
# LIMITATION: RCF scores are typically RIGHT-SKEWED — the bulk of scores
# cluster near zero (normal traffic) with a long tail toward high values
# (anomalies).  On a skewed distribution the 3-sigma threshold can land
# further to the right than intended, causing genuine anomalies to be MISSED
# (false negatives increase).  Only use this method if you have confirmed
# that the score histogram is roughly bell-shaped.
gaussian_threshold = mean_score + THRESHOLD_K * std_score

# --- Method 2 : Percentile (distribution-free) -----------------------------
# Makes NO assumption about the shape of the score distribution.
# threshold = score value at the Nth percentile of all test scores.
# Flagging the top (100 - ANOMALY_PERCENTILE) % of scores as anomalies
# works correctly regardless of whether the distribution is skewed, normal,
# multimodal, or heavy-tailed.
sorted_scores        = sorted(anomaly_scores)
percentile_index     = int(math.ceil(ANOMALY_PERCENTILE / 100.0 * len(sorted_scores))) - 1
percentile_index     = max(0, min(percentile_index, len(sorted_scores) - 1))
percentile_threshold = sorted_scores[percentile_index]

# --- Select the active threshold based on THRESHOLD_METHOD -----------------
if THRESHOLD_METHOD == "gaussian":
    threshold = gaussian_threshold
    print(f"  Threshold method     : Gaussian (mean + {THRESHOLD_K}×std)")
    print(f"  NOTE: assumes scores are normally distributed — may miss anomalies")
    print(f"        if the score distribution is right-skewed.")
else:
    threshold = percentile_threshold
    print(f"  Threshold method     : Percentile (top {100 - ANOMALY_PERCENTILE}% flagged)")
    print(f"  NOTE: distribution-free — works correctly for skewed score distributions.")

print(f"\n  Gaussian  threshold (mean + {THRESHOLD_K}×std)        : {gaussian_threshold:.4f}")
print(f"  Percentile threshold ({ANOMALY_PERCENTILE}th percentile)       : {percentile_threshold:.4f}")
print(f"  Active threshold                              : {threshold:.4f}")

# Convert continuous scores to binary predictions using the active threshold:
#   score > threshold → predicted anomaly (1)
#   score ≤ threshold → predicted normal  (0)
predictions = [1 if s > threshold else 0 for s in anomaly_scores]

# ---------------------------------------------------------------------------
# STEP 9 — Calculate and print classification accuracy
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9 — Classification accuracy")
print("=" * 70)

total_test = len(test_labels)
correct    = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
accuracy   = correct / total_test * 100

# Confusion matrix components
# tp — anomaly correctly detected  (true positive)
# tn — normal  correctly classified (true negative)
# fp — normal  flagged as anomaly  (false positive)
# fn — anomaly missed              (false negative)
tp = sum(1 for p, t in zip(predictions, test_labels) if p == 1 and t == 1)
tn = sum(1 for p, t in zip(predictions, test_labels) if p == 0 and t == 0)
fp = sum(1 for p, t in zip(predictions, test_labels) if p == 1 and t == 0)
fn = sum(1 for p, t in zip(predictions, test_labels) if p == 0 and t == 1)

# Precision — of all records flagged as anomaly, what fraction actually was?
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
# Recall    — of all actual anomalies, what fraction did the model catch?
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
# F1 Score  — harmonic mean of precision and recall; balances both metrics.
f1_score  = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0.0)

print(f"\n  Test set size    : {total_test}")
print(f"  Correct          : {correct}")
print(f"  Threshold method : {THRESHOLD_METHOD}  (threshold = {threshold:.4f})")
print(f"  Accuracy         : {accuracy:.2f} %")
print(f"\n  Confusion matrix :")
print(f"    True  Positives (anomaly correctly detected)  : {tp}")
print(f"    True  Negatives (normal  correctly classified) : {tn}")
print(f"    False Positives (normal  flagged as anomaly)   : {fp}")
print(f"    False Negatives (anomaly missed)               : {fn}")
print(f"\n  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1_score:.4f}")

# ---------------------------------------------------------------------------
# STEP 10 — Delete the inference endpoint
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10 — Deleting inference endpoint ...")
print("=" * 70)

# Always delete the endpoint after use to avoid ongoing hourly charges.
# The trained model artefact in S3 is unaffected by endpoint deletion and
# can be redeployed at any time via Estimator.deploy() or Model.deploy().
rcf_predictor.delete_endpoint()
print(f"  Endpoint '{rcf_predictor.endpoint_name}' deleted successfully.")
print("\nDone.")


