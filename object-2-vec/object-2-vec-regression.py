import json
import math
import os
import random
import time

import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve as get_image_uri
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import JSONSerializer
from sagemaker.session import Session

# ---------------------------------------------------------------------------
# SageMaker Python SDK v2.245.0 imports
# ---------------------------------------------------------------------------
# sagemaker.session.Session        — wraps a boto3 session; provides
#                                    upload_data() to push local files to S3
#                                    and acts as the shared context for all SDK
#                                    objects (Estimator, Predictor, etc.).
# sagemaker.estimator.Estimator    — high-level training-job launcher; manages
#                                    CreateTrainingJob, CloudWatch log streaming,
#                                    and exposes model_data after training.
# sagemaker.image_uris.retrieve    — looks up the correct ECR image URI for a
#                                    named built-in algorithm + region without
#                                    hard-coding AWS account IDs or image tags.
# sagemaker.inputs.TrainingInput   — wraps an S3 URI as a named data channel
#                                    passed to Estimator.fit().
# sagemaker.predictor.Predictor    — thin wrapper around invoke_endpoint() that
#                                    handles serialisation and deserialisation.
# sagemaker.serializers.JSONSerializer     — encodes a Python dict/list as JSON
#                                    bytes before sending to the endpoint.
# sagemaker.deserializers.JSONDeserializer — decodes the JSON response body back
#                                    into a Python dict automatically.

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# The S3 bucket where training data and model artifacts will be stored.
# The bucket must already exist and be in the same region as REGION below.
S3_BUCKET = "product-relation-data-bucket-12345"
# S3 key prefix used to organise all files written by this script.
S3_PREFIX = "product-relations/object-2-vec-regression"
# AWS region where the SageMaker training job and endpoint will be created.
REGION = "eu-north-1" # Stockholm is a good choice for low-latency access from Europe, but feel
# IAM role ARN that SageMaker will assume to access S3 and run training.
# Minimum required policy: AmazonSageMakerFullAccess
ROLE_ARN = "arn:aws:iam::148469447057:role/service-role/AmazonSageMaker-ExecutionRole-20260301T082995"
# Instance type used for BOTH training and endpoint hosting.
# ml.m5.xlarge is a good general-purpose starting point for small datasets.
# Switch to "local" to run with SageMaker Local Mode (no cloud cost, Docker required).
INSTANCE_TYPE = "ml.m5.xlarge"
# Total number of unique users in the dataset (IDs 0 – 9).
NUM_USERS = 10
# Total number of unique products in the dataset (IDs 0 – 19).
NUM_PRODUCTS = 20
# Path to the raw JSON Lines data file relative to this script.
DATA_FILE = "object-2-vec-regression-data.jsonl"
# Local paths where the 80 % / 20 % splits will be written before upload.
TRAIN_FILE = "object-2-vec-regression-train.jsonl"
TEST_FILE  = "object-2-vec-regression-testing.jsonl"
# Fixed random seed — ensures the same 80/20 split every time the script runs,
# making results reproducible across different machines and runs.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# STEP 1 — Load and shuffle the raw data
# ---------------------------------------------------------------------------

print("Loading data from:", DATA_FILE)
with open(DATA_FILE, "r") as f:
    # Read every non-empty line and parse it as a JSON object.
    records = [json.loads(line) for line in f if line.strip()]
print(f"  Total records loaded: {len(records)}")

# Shuffle the records so that the train/test split is random rather than
# ordered by user or product ID, which could introduce sampling bias.
random.seed(RANDOM_SEED)
random.shuffle(records)

# ---------------------------------------------------------------------------
# STEP 2 — Split 80 % training / 20 % testing
# ---------------------------------------------------------------------------
# Calculate the index that separates the training set from the test set.
# int() floors the result so all remaining records go to the test set.
split_index   = int(len(records) * 0.80)
train_records = records[:split_index]   # 80 % — used to train the model
test_records  = records[split_index:]   # 20 % — held out for predictions only
print(f"  Training records : {len(train_records)}  (80 %)")
print(f"  Test records     : {len(test_records)}   (20 %)")
# Write the splits to local JSON Lines files so they can be uploaded to S3.
def write_jsonl(path: str, data: list) -> None:
    """Write a list of dicts to a JSON Lines file (one JSON object per line)."""
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

write_jsonl(TRAIN_FILE, train_records)
write_jsonl(TEST_FILE, test_records)
print(f"  Splits written to disk: {TRAIN_FILE}, {TEST_FILE}")

# ---------------------------------------------------------------------------
# STEP 3 — Create a SageMaker Session and upload data splits to S3
# ---------------------------------------------------------------------------
# Session wraps the boto3 session and is the shared context passed to every
# SageMaker SDK object (Estimator, Predictor).  upload_data() copies a local
# file to S3 under the given key prefix and returns the full s3:// URI.

boto3_session = boto3.Session(region_name=REGION)
sm_session    = Session(boto_session=boto3_session)

print("\nUploading data to S3 ...")
# upload_data() copies the file to S3 and returns:
#   s3://<bucket>/<key_prefix>/<filename>
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
# STEP 4 — Retrieve the Object2Vec container image URI
# ---------------------------------------------------------------------------
# get_image_uri() (sagemaker.image_uris.retrieve) looks up the correct ECR
# image URI for the given algorithm name and region using the SDK's built-in
# region → AWS-account mapping.  No hard-coded account IDs or image tags needed.
# "object2vec" is the algorithm identifier recognised by the SageMaker SDK.
container_image_uri = get_image_uri(
    framework="object2vec",
    region=REGION,
)
print(f"\nObject2Vec container image: {container_image_uri}")

# ---------------------------------------------------------------------------
# STEP 5 — Define Object2Vec hyperparameters
# ---------------------------------------------------------------------------
# Hyperparameters are passed as a plain dict to the Estimator constructor.
# The SDK automatically converts all values to strings when sending them to
# the container — no manual casting to str is required.

hyperparameters = {
    # Network type for encoder 0 (users).
    # "pooled_embedding" maps each integer token to a dense vector then
    # averages them — ideal for single-ID inputs like user IDs.
    "enc0_network": "pooled_embedding",
    # Network type for encoder 1 (products).
    # Same pooled-embedding approach as enc0, suitable for single product IDs.
    "enc1_network": "pooled_embedding",
    # Vocabulary size for encoder 0 — must equal the total number of unique
    # user IDs in the dataset. IDs must be in the range [0, enc0_vocab_size).
    "enc0_vocab_size": NUM_USERS,

    # Maximum sequence length for encoder 0.
    # Required by the Object2Vec container when enc0_network is "pooled_embedding"
    # or "bilstm". Each user input is a single integer ID wrapped in a list
    # e.g. [3], so the sequence length is always 1.
    "enc0_max_seq_len": 1,

    # Vocabulary size for encoder 1 — must equal the total number of unique
    # product IDs in the dataset.
    "enc1_vocab_size": NUM_PRODUCTS,

    # Maximum sequence length for encoder 1.
    # Required by the Object2Vec container when enc1_network is "pooled_embedding"
    # or "bilstm". Each product input is a single integer ID wrapped in a list
    # e.g. [14], so the sequence length is always 1.
    "enc1_max_seq_len": 1,

    # Dimensionality of the embedding vector produced by each encoder.
    # Larger values capture more nuance but require more data and compute.
    # 64 is a good starting point for small-to-medium datasets.
    "enc_dim": 64,

    # --- Comparison MLP ---

    # Number of neurons in each hidden layer of the multi-layer perceptron
    # that compares the two encoder outputs to produce a prediction.
    "mlp_dim": 256,

    # Number of hidden layers in the comparison MLP.
    # 2 layers is a standard choice for capturing non-linear relationships.
    "mlp_layers": 2,

    # Activation function applied after each MLP hidden layer.
    # "relu" is computationally efficient and avoids vanishing gradients.
    "mlp_activation": "relu",

    # --- Output layer ---

    # Determines how the model produces its final output and which loss is used.
    # Valid values accepted by the Object2Vec container:
    #   "softmax"            — classification; predicts a probability over N classes.
    #                          Requires num_classes to be set in range (2, 30).
    #   "mean_squared_error" — regression; predicts a single continuous scalar
    #                          using MSE loss. num_classes must NOT be set.
    "output_layer": "mean_squared_error",

    # NOTE: num_classes is intentionally omitted here.
    # It is only valid when output_layer = "softmax" (classification mode)
    # and must be in the range (2, 30). Setting num_classes = 1 causes:
    #   ValidationError: 'num_classes' should be within range (2, 30)

    # --- Optimiser & training schedule ---

    # "adam" adapts per-parameter learning rates and typically converges
    # faster than plain SGD.
    "optimizer": "adam",

    # Step size for each gradient update. 0.001 is the canonical Adam default.
    "learning_rate": 0.001,

    # Number of full passes over the training dataset.
    "epochs": 20,

    # Samples processed per forward/backward pass.
    "mini_batch_size": 16,

    # Dropout rate — 0.0 disables regularisation (safe for small datasets).
    "dropout": 0.0,

    # --- Early stopping ---

    # Stop training if validation loss does not improve for this many epochs.
    "early_stopping_patience": 3,

    # Minimum absolute improvement required to reset the patience counter.
    "early_stopping_tolerance": 0.001,
}

# ---------------------------------------------------------------------------
# STEP 6 — Create the Estimator and launch the training job
# ---------------------------------------------------------------------------
# sagemaker.estimator.Estimator is the v2 SDK high-level training API.
# It wraps CreateTrainingJob, streams CloudWatch logs, and populates
# model_data with the S3 URI of the trained model artefact after .fit().
#
# Key constructor parameters:
#   image_uri         — ECR image used for training (resolved in STEP 4).
#   role              — IAM role SageMaker assumes to access S3 and ECR.
#   instance_type     — EC2 instance type for the training job.
#   instance_count    — number of training instances (1 for single-node).
#   volume_size       — EBS storage in GiB attached to the training instance.
#   output_path       — S3 prefix where SageMaker writes model.tar.gz.
#   max_run           — maximum wall-clock seconds (prevents runaway cost).
#   hyperparameters   — algorithm configuration dict (defined in STEP 5).
#                       The SDK converts all values to strings automatically.
#   sagemaker_session — shared Session created in STEP 3.

estimator = Estimator(
    image_uri=container_image_uri,                          # Object2Vec ECR image
    role=ROLE_ARN,                                          # IAM execution role
    instance_type=INSTANCE_TYPE,                            # Training EC2 instance type
    instance_count=1,                                       # Single-node training
    volume_size=10,                                         # 10 GiB EBS volume
    output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output",    # Destination for model.tar.gz
    max_run=3600,                                           # 1-hour training time limit
    hyperparameters=hyperparameters,                        # Object2Vec config (STEP 5)
    sagemaker_session=sm_session,                           # Shared session (STEP 3)
)

# TrainingInput wraps an S3 URI as a named data channel.
# "train" is the channel name the Object2Vec container reads from
# /opt/ml/input/data/train/ inside the training container.
train_input = TrainingInput(
    s3_data=s3_train_uri,
    content_type="application/jsonlines",   # Tell the container the file format
)

print("\nStarting Object2Vec training job ...")
print("  This may take several minutes. Monitor progress in the SageMaker console.")

# .fit() launches the managed SageMaker training job.
# wait=True  — blocks this process until the job finishes (success or failure).
# logs=True  — streams CloudWatch training logs to the local console in real time.
estimator.fit(
    inputs={"train": train_input},
    wait=True,
    logs=True,
)
print("\nTraining complete.")

# model_data is automatically populated by the Estimator after .fit() completes.
# It points to: s3://<bucket>/<prefix>/output/<job-name>/output/model.tar.gz
model_artefact_s3 = estimator.model_data
print(f"  Model artefact: {model_artefact_s3}")

# ---------------------------------------------------------------------------
# STEP 7 — Deploy the trained model to a real-time endpoint
# ---------------------------------------------------------------------------
# Estimator.deploy() calls CreateModel, CreateEndpointConfig, and CreateEndpoint
# in sequence, then polls until the endpoint status reaches "InService".
# It returns a Predictor instance bound to the new endpoint.
#
# Key parameters:
#   initial_instance_count — number of EC2 instances behind the endpoint.
#   instance_type          — EC2 instance type for inference.
#   endpoint_name          — fixed name so the endpoint can be referenced later.
#                            Re-using the same name on subsequent runs performs
#                            a rolling update of the existing endpoint.
#   serializer             — JSONSerializer encodes the Python dict payload to
#                            JSON bytes before sending it over HTTPS.
#   deserializer           — JSONDeserializer parses the JSON response body back
#                            into a Python dict automatically.

endpoint_name = "object2vec-regression-endpoint"

print(f"\nDeploying model to endpoint '{endpoint_name}' (may take 5 – 10 minutes) ...")
predictor = estimator.deploy(
    initial_instance_count=1,           # One serving instance
    instance_type=INSTANCE_TYPE,        # Same instance type as training
    endpoint_name=endpoint_name,        # Fixed name for easy reference
    serializer=JSONSerializer(),        # Encodes request payload as JSON
    deserializer=JSONDeserializer(),    # Decodes JSON response to Python dict
)
print(f"  Endpoint '{endpoint_name}' is InService.")

# ---------------------------------------------------------------------------
# STEP 8 — Run predictions on the 20 % holdout test set and compute RMSE
# ---------------------------------------------------------------------------
# Predictor.predict() serialises the payload with the JSONSerializer,
# calls invoke_endpoint() under the hood, and deserialises the response
# with the JSONDeserializer — all in a single method call.

print("\nRunning predictions on the test set ...")
actual_labels    = []   # Ground-truth affinity scores from the test split
predicted_labels = []   # Scores predicted by the deployed Object2Vec model
for i, record in enumerate(test_records):
    # Object2Vec inference expects a JSON payload with an "instances" key.
    # Each instance mirrors the training format: in0 (user) and in1 (product).
    payload = {
        "instances": [
            {
                "in0": record["in0"],   # User ID as a list, e.g. [3]
                "in1": record["in1"],   # Product ID as a list, e.g. [14]
            }
        ]
    }

    # predictor.predict() sends the payload to the live endpoint over HTTPS
    # and returns the deserialised response as a Python dict.
    result = predictor.predict(payload)

    # Print the raw response shape on the first record so the actual keys
    # returned by the container are visible in the logs for debugging.
    if i == 0:
        print(f"  [DEBUG] Raw response from endpoint: {result}")

    # Object2Vec response structure for mean_squared_error (regression) mode:
    #   { "predictions": [ { "scores": [<float>] } ] }
    #
    # Note: the key is "scores" (plural, a list) NOT "score" (singular).
    # "score" (singular) is only returned in classification (softmax) mode.
    # We take index [0] of the "scores" list to get the single regression value.
    prediction      = result["predictions"][0]
    # Handle both "scores" (regression) and "score" (classification) gracefully.
    if "scores" in prediction:
        predicted_score = prediction["scores"][0]   # regression: list with one float
    else:
        predicted_score = prediction["score"]       # classification fallback

    actual_labels.append(record["label"])
    predicted_labels.append(predicted_score)

    print(
        f"  [{i+1:>2}/{len(test_records)}]  "
        f"User {record['in0'][0]:>2} → Product {record['in1'][0]:>2}  |  "
        f"Actual: {record['label']:.1f}  |  Predicted: {predicted_score:.4f}"
    )

# --- Compute Root Mean Squared Error (RMSE) ---
# RMSE measures average prediction error in the same units as the label.
# Lower is better; RMSE < 0.5 on a 1.0 – 5.0 scale is generally good.
squared_errors = [(a - p) ** 2 for a, p in zip(actual_labels, predicted_labels)]
rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

print(f"\n  RMSE on 20 % test set: {rmse:.4f}")
print("  (RMSE is in the same units as the affinity score, range 1.0 – 5.0)")

# ---------------------------------------------------------------------------
# STEP 9 — Clean up: delete the endpoint to stop incurring charges
# ---------------------------------------------------------------------------
# predictor.delete_endpoint(delete_endpoint_config=True) is a single convenience
# call that removes:
#   - the live endpoint (stops hourly instance charges immediately)
#   - the endpoint configuration
#   - the underlying registered model object
# Keeping any of these alive after use incurs ongoing AWS charges.
print("\nCleaning up AWS resources ...")
try:
    predictor.delete_endpoint(delete_endpoint_config=True)
    print(f"  Deleted endpoint: {endpoint_name}")
except Exception as exc:
    print(f"  Warning — could not delete endpoint: {exc}")

print("\nDone.")

