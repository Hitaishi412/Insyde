"""
=============================================================================
Insyde UEBA System — Step 3: Train Isolation Forest Model
=============================================================================
Purpose  : Train an unsupervised anomaly detection model on the engineered
           behavioural features from cert_features.csv.

           Pipeline:
             1. Load cert_features.csv
             2. Isolate numeric features (drop user ID, labels, metadata)
             3. Scale features with StandardScaler
             4. Train IsolationForest (contamination=0.05, n_estimators=200)
             5. Persist model.pkl, scaler.pkl, features.pkl via joblib

Output Artifacts:
  model.pkl    — Trained IsolationForest model
  scaler.pkl   — Fitted StandardScaler
  features.pkl — List of feature column names used during training
=============================================================================
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILE    = "cert_features.csv"
MODEL_PATH    = "model.pkl"
SCALER_PATH   = "scaler.pkl"
FEATURES_PATH = "features.pkl"

# IsolationForest hyperparameters
CONTAMINATION  = 0.05   # Expected fraction of anomalous users (~5%)
N_ESTIMATORS   = 200    # Number of isolation trees
RANDOM_STATE   = 42     # Reproducibility seed

# Columns that are identifiers, labels, or non-feature metadata
# These are explicitly excluded regardless of dtype
NON_FEATURE_COLS = {
    "user",          # User identifier
    "label",         # Ground-truth insider-threat label (Step 2 output)
    "odd_onboarding" # Binary flag — kept separate to avoid data leakage
}

# ─────────────────────────────────────────────────────────────────────────────

def separator(title: str = "") -> None:
    """Print a visual section separator to the terminal."""
    width = 70
    if title:
        pad   = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print(f"\n{'═' * width}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load the feature CSV and perform basic sanity checks."""
    separator("STEP 1 — Load Data")

    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        print("        Ensure you have run step2_feature_engineering.py first.")
        sys.exit(1)

    print(f"[1/5] Loading data from  →  {filepath}")
    df = pd.read_csv(filepath)

    print(f"      Rows     : {df.shape[0]:,}")
    print(f"      Columns  : {df.shape[1]:,}")

    # Warn on missing values before imputation
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        print(f"\n[WARN] {len(cols_with_nulls)} column(s) contain NaN values — "
              "will be imputed with column median.")
    else:
        print("      No missing values detected.")

    return df


def select_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Isolate only numeric feature columns.

    Strategy:
      1. Select columns with numeric dtype (int64 / float64).
      2. Drop any column names that appear in NON_FEATURE_COLS.
      3. Impute remaining NaNs with column-wise median (robust to outliers).
    """
    separator("STEP 2 — Feature Selection")

    print(f"[2/5] Selecting numeric features …")
    print(f"      Total columns in dataset : {df.shape[1]}")

    # Step A — keep only numeric dtype columns
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"      Numeric dtype columns     : {numeric_df.shape[1]}")

    # Step B — drop known non-feature columns that happened to be numeric
    dropped = [c for c in NON_FEATURE_COLS if c in numeric_df.columns]
    if dropped:
        numeric_df = numeric_df.drop(columns=dropped)
        print(f"      Excluded non-feature cols : {dropped}")

    feature_cols = numeric_df.columns.tolist()
    print(f"      Final training features   : {len(feature_cols)}")

    # Step C — impute NaNs with column median
    null_mask = numeric_df.isnull().any()
    if null_mask.any():
        imputed_cols = null_mask[null_mask].index.tolist()
        numeric_df[imputed_cols] = numeric_df[imputed_cols].fillna(
            numeric_df[imputed_cols].median()
        )
        print(f"      Imputed {len(imputed_cols)} column(s) with column median.")

    # Print feature group summary based on naming conventions
    groups = {
        "Email Exfiltration"  : [c for c in feature_cols if "email" in c or "bcc" in c or "recipient" in c or "attachment" in c],
        "File Staging"        : [c for c in feature_cols if "file" in c or "filetype" in c or "doc" in c or "exe" in c or "zip" in c],
        "Temporal / Logon"    : [c for c in feature_cols if "logon" in c or "logoff" in c or "weekend" in c or "off_hour" in c or "after_hour" in c],
        "Network / URL"       : [c for c in feature_cols if "url" in c or "domain" in c or "brows" in c or "network" in c],
        "Device / Endpoint"   : [c for c in feature_cols if "device" in c or "pc" in c or "disconnect" in c],
        "Composite / Chain"   : [c for c in feature_cols if "composite" in c or "chain" in c or "score" in c],
    }
    print(f"\n      {'Feature Group':<28}  {'Count':>5}")
    print(f"      {'─'*35}")
    for group, cols in groups.items():
        if cols:
            print(f"      {group:<28}  {len(cols):>5}")
    print(f"      {'─'*35}")
    print(f"      {'TOTAL':<28}  {len(feature_cols):>5}")

    return numeric_df, feature_cols


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the training data and transform the features.

    StandardScaler is preferred here because IsolationForest partitions the
    feature space uniformly — features on vastly different scales would bias
    the random split selection toward high-variance dimensions.
    """
    separator("STEP 3 — Feature Scaling")

    print("[3/5] Fitting StandardScaler on training data …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Print per-feature scale summary (mean & std before scaling)
    means = X.mean()
    stds  = X.std()
    print(f"\n      Sample of pre-scaling statistics (first 5 features):")
    print(f"      {'Feature':<40} {'Mean':>10} {'Std':>10}")
    print(f"      {'─'*62}")
    for col in X.columns[:5]:
        print(f"      {col:<40} {means[col]:>10.4f} {stds[col]:>10.4f}")
    print(f"      … (+{len(X.columns) - 5} more features)")

    print(f"\n      Scaler fitted on {X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features.")
    print(f"      Post-scaling: mean ≈ 0, std ≈ 1 per feature.")

    return X_scaled, scaler


def train_isolation_forest(X_scaled: np.ndarray) -> IsolationForest:
    """
    Train the IsolationForest anomaly detector.

    IsolationForest is well-suited for insider threat detection because:
      • Unsupervised — labels are rare and expensive in this domain.
      • Efficient at isolating outliers in high-dimensional behavioural data.
      • Contamination parameter directly models the expected anomaly rate.
    """
    separator("STEP 4 — Train IsolationForest")

    print(f"[4/5] Initialising IsolationForest …")
    print(f"      contamination  = {CONTAMINATION}  (≈ {CONTAMINATION*100:.0f}% of users flagged as anomalous)")
    print(f"      n_estimators   = {N_ESTIMATORS}")
    print(f"      random_state   = {RANDOM_STATE}")
    print(f"      Training on    {X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features")
    print(f"\n      Fitting model … ", end="", flush=True)

    t0 = time.perf_counter()
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1          # Use all available CPU cores
    )
    model.fit(X_scaled)
    elapsed = time.perf_counter() - t0

    print(f"done  ({elapsed:.2f}s)")

    # Post-training diagnostics
    scores    = model.decision_function(X_scaled)   # Higher = more normal
    preds     = model.predict(X_scaled)             # -1 = anomaly, +1 = normal
    n_anomaly = (preds == -1).sum()
    n_normal  = (preds == 1).sum()

    print(f"\n      ── Training Set Predictions ──────────────────────────")
    print(f"      Normal users   (+1) : {n_normal:>5,}  ({n_normal/len(preds)*100:.1f}%)")
    print(f"      Anomalous users (-1) : {n_anomaly:>5,}  ({n_anomaly/len(preds)*100:.1f}%)")
    print(f"\n      ── Anomaly Score Distribution (decision_function) ────")
    print(f"      Min    : {scores.min():.4f}")
    print(f"      Max    : {scores.max():.4f}")
    print(f"      Mean   : {scores.mean():.4f}")
    print(f"      Std    : {scores.std():.4f}")
    print(f"      Threshold (contamination boundary): ~{np.percentile(scores, CONTAMINATION*100):.4f}")

    return model


def save_artifacts(model: IsolationForest,
                   scaler: StandardScaler,
                   feature_cols: list[str]) -> None:
    """Persist all three training artifacts using joblib."""
    separator("STEP 5 — Save Artifacts")

    print(f"[5/5] Saving model artifacts …\n")

    artifacts = {
        MODEL_PATH    : model,
        SCALER_PATH   : scaler,
        FEATURES_PATH : feature_cols,
    }

    for path, obj in artifacts.items():
        joblib.dump(obj, path)
        size_kb = Path(path).stat().st_size / 1024
        kind = type(obj).__name__ if not isinstance(obj, list) else f"list[{len(obj)} features]"
        print(f"      ✓  {path:<20}  ({kind})  —  {size_kb:.1f} KB")

    print(f"\n      All artifacts saved successfully.")
    print(f"\n      To load in downstream scripts:")
    print(f"        import joblib")
    print(f"        model    = joblib.load('{MODEL_PATH}')")
    print(f"        scaler   = joblib.load('{SCALER_PATH}')")
    print(f"        features = joblib.load('{FEATURES_PATH}')")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    separator("Insyde UEBA — Phase 3: Train Isolation Forest Model")
    print("  Model    : IsolationForest (unsupervised anomaly detection)")
    print("  Input    : cert_features.csv")
    print("  Outputs  : model.pkl | scaler.pkl | features.pkl")

    total_start = time.perf_counter()

    # Pipeline
    df                    = load_data(INPUT_FILE)
    X, feature_cols       = select_numeric_features(df)
    X_scaled, scaler      = scale_features(X)
    model                 = train_isolation_forest(X_scaled)
    save_artifacts(model, scaler, feature_cols)

    total_elapsed = time.perf_counter() - total_start
    separator("Training Complete")
    print(f"  Total time          : {total_elapsed:.2f}s")
    print(f"  Features trained on : {len(feature_cols)}")
    print(f"  Users processed     : {X_scaled.shape[0]:,}")
    print(f"  Anomaly rate        : {CONTAMINATION*100:.0f}%  (contamination parameter)")
    print(f"\n  Next step → run step4_score_users.py to generate risk predictions.")
    separator()


if __name__ == "__main__":
    main()
