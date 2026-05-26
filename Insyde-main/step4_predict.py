"""
=============================================================================
Insyde UEBA System — Step 4: Predict & Score Users
=============================================================================
Purpose  : Apply the trained IsolationForest model to every user in
           cert_features.csv and produce anomaly predictions that feed
           directly into Step 5 risk scoring.

Pipeline:
  1. Load model.pkl, scaler.pkl, features.pkl  (from Step 3)
  2. Load cert_features.csv                    (from Step 2)
  3. Align features to the exact columns seen during training
  4. Scale with the fitted StandardScaler
  5. Predict anomaly flag  (-1 = anomaly,  +1 = normal)
  6. Compute decision_function score (more negative = more suspicious)
  7. Convert to intuitive iforest_anomaly_score (0–100, higher = riskier)
  8. Save cert_predictions.csv  — used as input to step5_risk_scoring.py
  9. Patch step5_risk_scoring.py to activate ANOMALY_SCORE_COL

Output   : cert_predictions.csv — all original features + model predictions
           console report        — prediction summary and top flagged users
=============================================================================
"""

import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_CSV    = "cert_features.csv"    # Step 2 output
MODEL_PKL       = "model.pkl"            # Step 3 output
SCALER_PKL      = "scaler.pkl"           # Step 3 output
FEATURES_PKL    = "features.pkl"         # Step 3 output
OUTPUT_CSV      = "cert_predictions.csv" # Consumed by step5_risk_scoring.py

# Column name that step5_risk_scoring.py looks for
ANOMALY_SCORE_COL = "iforest_anomaly_score"

# ─────────────────────────────────────────────────────────────────────────────

def separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print(f"\n{'═' * width}")


def check_artifacts() -> None:
    """Verify all required files exist before loading anything."""
    separator("Pre-flight Checks")
    required = {
        MODEL_PKL    : "Trained IsolationForest  (run train_model.py first)",
        SCALER_PKL   : "Fitted StandardScaler    (run train_model.py first)",
        FEATURES_PKL : "Training feature list    (run train_model.py first)",
        FEATURES_CSV : "Engineered features CSV  (run step2_feature_engineering.py first)",
    }
    all_ok = True
    for filepath, description in required.items():
        exists = Path(filepath).exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {filepath:<25}  {description}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n[ERROR] One or more required files are missing. See above.")
        sys.exit(1)

    print("\n  All required files found. Proceeding …")


def load_artifacts() -> tuple:
    """Load the three model artifacts saved by train_model.py."""
    separator("STEP 1 — Load Model Artifacts")

    print("[1/5] Loading trained artifacts …\n")
    model    = joblib.load(MODEL_PKL)
    scaler   = joblib.load(SCALER_PKL)
    features = joblib.load(FEATURES_PKL)

    print(f"  model.pkl    →  IsolationForest")
    print(f"                  n_estimators   = {model.n_estimators}")
    print(f"                  contamination  = {model.contamination} "
          f"(~{model.contamination*100:.0f}% expected anomalies)")
    print(f"                  random_state   = {model.random_state}")
    print(f"\n  scaler.pkl   →  StandardScaler  ({scaler.n_features_in_} features)")
    print(f"  features.pkl →  {len(features)} feature names")

    return model, scaler, features


def load_and_align(features: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cert_features.csv and align columns to match exactly what
    the model was trained on — same columns, same order.
    """
    separator("STEP 2 — Load & Align Features")

    print(f"[2/5] Loading {FEATURES_CSV} …")
    df = pd.read_csv(FEATURES_CSV)
    print(f"      Loaded {df.shape[0]:,} users × {df.shape[1]} columns")

    # Identify any feature drift between training time and now
    current_cols = set(df.columns)
    trained_cols = set(features)

    missing_in_data   = trained_cols - current_cols
    extra_in_data     = current_cols - trained_cols - {"user", "label"}

    if missing_in_data:
        print(f"\n  [WARN] {len(missing_in_data)} feature(s) in model not found in CSV "
              "— will be filled with 0:")
        for c in sorted(missing_in_data):
            print(f"         • {c}")
        for c in missing_in_data:
            df[c] = 0.0

    if extra_in_data:
        print(f"\n  [INFO] {len(extra_in_data)} extra column(s) in CSV not used "
              "by model (ignored).")

    # Preserve user column for output
    users = df["user"].copy() if "user" in df.columns else pd.Series(
        range(len(df)), name="user"
    )

    # Select exactly the trained feature columns in the correct order
    X = df[features].copy()

    # Impute any NaNs with column median (same strategy as train_model.py)
    null_cols = X.columns[X.isnull().any()].tolist()
    if null_cols:
        X[null_cols] = X[null_cols].fillna(X[null_cols].median())
        print(f"\n  [INFO] Imputed NaNs in {len(null_cols)} column(s) with median.")

    print(f"\n      Feature matrix ready: {X.shape[0]:,} users × {X.shape[1]} features")
    print(f"      Feature alignment   : ✓  ({len(features)} columns, correct order)")

    return users, X, df


def scale_and_predict(
    model, scaler, users: pd.Series, X: pd.DataFrame, df_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Scale features and run both predict() and decision_function().
    Converts raw IF scores to an intuitive 0–100 anomaly score.
    """
    separator("STEP 3 — Scale Features")

    print(f"[3/5] Applying StandardScaler …")
    X_scaled = scaler.transform(X)
    print(f"      Scaled {X_scaled.shape[0]:,} users × {X_scaled.shape[1]} features")
    print(f"      Using scaler fitted during training (no data leakage).")

    separator("STEP 4 — Run Model Predictions")

    print(f"[4/5] Running IsolationForest predictions …", end="", flush=True)
    t0 = time.perf_counter()

    # -1 = anomaly (insider threat candidate), +1 = normal
    predictions = model.predict(X_scaled)

    # Raw decision scores: more negative → more anomalous
    # Typical range: [-0.5, 0.5] depending on contamination
    raw_scores = model.decision_function(X_scaled)

    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.3f}s)")

    # ── Convert to intuitive 0–100 anomaly score ──────────────────────────
    # Flip sign so higher = more suspicious, then min-max normalise to 0–100
    flipped = -raw_scores
    score_min, score_max = flipped.min(), flipped.max()
    if score_max - score_min > 1e-9:
        anomaly_score_100 = ((flipped - score_min) / (score_max - score_min)) * 100
    else:
        anomaly_score_100 = np.zeros(len(flipped))

    # Build output dataframe
    results = pd.DataFrame({
        "user"                   : users.values,
        "iforest_prediction"     : predictions,           # -1 or +1
        "iforest_raw_score"      : raw_scores,            # raw decision score
        ANOMALY_SCORE_COL        : anomaly_score_100,     # 0–100 (higher = riskier)
        "is_anomaly"             : (predictions == -1).astype(int),
    })

    # Attach ground-truth label if available (for evaluation only)
    if "label" in df_full.columns:
        results["ground_truth_label"] = df_full["label"].values

    return results


def print_prediction_report(results: pd.DataFrame) -> None:
    """Print a clear summary of prediction outcomes to the terminal."""
    separator("STEP 5 — Prediction Summary")

    n_total   = len(results)
    n_anomaly = results["is_anomaly"].sum()
    n_normal  = n_total - n_anomaly

    scores = results[ANOMALY_SCORE_COL]

    print(f"[5/5] Prediction results for {n_total:,} users:\n")
    print(f"  {'Outcome':<22} {'Count':>6}  {'% of Total':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Normal      (+1)':<22} {n_normal:>6,}  {n_normal/n_total*100:>9.1f}%")
    print(f"  {'Anomalous   (-1)':<22} {n_anomaly:>6,}  {n_anomaly/n_total*100:>9.1f}%")
    print(f"  {'─'*42}")
    print(f"  {'TOTAL':<22} {n_total:>6,}  {'100.0%':>10}")

    print(f"\n  Anomaly Score Distribution  (0 = normal, 100 = most suspicious)")
    print(f"  {'─'*42}")
    print(f"  Min     : {scores.min():.2f}")
    print(f"  Max     : {scores.max():.2f}")
    print(f"  Mean    : {scores.mean():.2f}")
    print(f"  Std Dev : {scores.std():.2f}")
    print(f"  P75     : {np.percentile(scores, 75):.2f}")
    print(f"  P90     : {np.percentile(scores, 90):.2f}")
    print(f"  P95     : {np.percentile(scores, 95):.2f}")

    # Evaluate against ground truth if available
    if "ground_truth_label" in results.columns:
        print(f"\n  Ground Truth Evaluation (label column from cert_features.csv)")
        print(f"  {'─'*42}")
        actual_positives = results["ground_truth_label"].sum()
        predicted_positives = results["is_anomaly"].sum()
        true_positives  = ((results["is_anomaly"] == 1) &
                           (results["ground_truth_label"] == 1)).sum()
        false_positives = ((results["is_anomaly"] == 1) &
                           (results["ground_truth_label"] == 0)).sum()
        false_negatives = ((results["is_anomaly"] == 0) &
                           (results["ground_truth_label"] == 1)).sum()

        precision = true_positives / (predicted_positives + 1e-9)
        recall    = true_positives / (actual_positives + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"  Actual insiders (label=1)  : {actual_positives:>5,}")
        print(f"  Predicted anomalies        : {predicted_positives:>5,}")
        print(f"  True  Positives (TP)       : {true_positives:>5,}")
        print(f"  False Positives (FP)       : {false_positives:>5,}")
        print(f"  False Negatives (FN)       : {false_negatives:>5,}")
        print(f"\n  Precision : {precision:.3f}")
        print(f"  Recall    : {recall:.3f}")
        print(f"  F1 Score  : {f1:.3f}")
        print(f"\n  Note: IsolationForest is unsupervised — labels were NOT used")
        print(f"        during training. Low recall is expected and normal.")

    # Top flagged users
    print(f"\n  Top 10 Highest-Risk Users (by anomaly score):")
    print(f"  {'─'*52}")
    print(f"  {'User':<12} {'Anomaly Score':>14} {'Prediction':>12} {'Anomaly?':>9}")
    print(f"  {'─'*52}")
    top10 = results.nlargest(10, ANOMALY_SCORE_COL)
    for _, row in top10.iterrows():
        flag    = "⚠ YES" if row["is_anomaly"] else "  no"
        pred    = "ANOMALY (-1)" if row["iforest_prediction"] == -1 else "Normal  (+1)"
        print(f"  {str(row['user']):<12} {row[ANOMALY_SCORE_COL]:>14.2f} {pred:>12} {flag:>9}")


def save_predictions(results: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    Merge anomaly scores back into the full feature dataframe and save.
    step5_risk_scoring.py will read this file and activate the IF blend.
    """
    separator("Save Output")

    # Merge predictions into the original feature dataframe
    if "user" in df_full.columns:
        df_out = df_full.merge(
            results[["user", "iforest_prediction", "iforest_raw_score",
                     ANOMALY_SCORE_COL, "is_anomaly"]],
            on="user",
            how="left"
        )
    else:
        df_out = pd.concat([df_full, results.drop(columns=["user"])], axis=1)

    df_out.to_csv(OUTPUT_CSV, index=False)
    size_kb = Path(OUTPUT_CSV).stat().st_size / 1024
    print(f"  ✓  Saved → {OUTPUT_CSV}  ({df_out.shape[0]:,} rows × {df_out.shape[1]} cols, {size_kb:.0f} KB)")
    print(f"\n  New columns added:")
    print(f"    • iforest_prediction   — raw model output (-1 anomaly / +1 normal)")
    print(f"    • iforest_raw_score    — raw decision_function score")
    print(f"    • {ANOMALY_SCORE_COL:<25}— intuitive 0–100 risk score")
    print(f"    • is_anomaly           — binary flag (1 = anomalous)")


def patch_step5() -> None:
    """
    Automatically update ANOMALY_SCORE_COL in step5_risk_scoring.py
    so it picks up the IF scores without any manual edits.
    """
    step5_path = Path("step5_risk_scoring.py")
    if not step5_path.exists():
        print(f"\n  [INFO] step5_risk_scoring.py not found — skipping auto-patch.")
        print(f"         Manually set ANOMALY_SCORE_COL = \"{ANOMALY_SCORE_COL}\"")
        print(f"         and INPUT_FILE = \"{OUTPUT_CSV}\" in that script.")
        return

    content = step5_path.read_text()

    patched = False

    # Patch 1: Activate the anomaly score column
    old_score_line = 'ANOMALY_SCORE_COL = None   # e.g., "iforest_anomaly_score"'
    new_score_line = f'ANOMALY_SCORE_COL = "{ANOMALY_SCORE_COL}"'
    if old_score_line in content:
        content = content.replace(old_score_line, new_score_line)
        patched = True

    # Patch 2: Point INPUT_FILE to cert_predictions.csv
    old_input_line = 'INPUT_FILE   = "cert_features.csv"   # output of step2_feature_engineering.py'
    new_input_line = f'INPUT_FILE   = "{OUTPUT_CSV}"'
    if old_input_line in content:
        content = content.replace(old_input_line, new_input_line)
        patched = True

    if patched:
        step5_path.write_text(content)
        print(f"\n  ✓  Auto-patched step5_risk_scoring.py:")
        print(f"       ANOMALY_SCORE_COL = \"{ANOMALY_SCORE_COL}\"")
        print(f"       INPUT_FILE        = \"{OUTPUT_CSV}\"")
        print(f"     Step 5 will now blend 70% indicator score + 30% IF model score.")
    else:
        print(f"\n  [INFO] step5_risk_scoring.py already patched or format changed.")
        print(f"         Ensure these values are set manually if needed:")
        print(f"           ANOMALY_SCORE_COL = \"{ANOMALY_SCORE_COL}\"")
        print(f"           INPUT_FILE        = \"{OUTPUT_CSV}\"")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    separator("Insyde UEBA — Phase 4: Predict & Score Users")
    print("  Model    : IsolationForest  (loaded from model.pkl)")
    print("  Input    : cert_features.csv")
    print("  Output   : cert_predictions.csv  →  fed into step5_risk_scoring.py")

    total_start = time.perf_counter()

    # Pipeline
    check_artifacts()
    model, scaler, features    = load_artifacts()
    users, X, df_full          = load_and_align(features)
    results                    = scale_and_predict(model, scaler, users, X, df_full)
    print_prediction_report(results)
    save_predictions(results, df_full)
    patch_step5()

    total_elapsed = time.perf_counter() - total_start
    separator("Complete")
    print(f"  Total time   : {total_elapsed:.2f}s")
    print(f"  Users scored : {len(results):,}")
    print(f"  Anomalies    : {results['is_anomaly'].sum()} "
          f"({results['is_anomaly'].mean()*100:.1f}%)")
    print(f"\n  ── Next Step ───────────────────────────────────────────────")
    print(f"  Run:  python step5_risk_scoring.py")
    print(f"  This will blend your IF anomaly scores with the behavioural")
    print(f"  indicator weights for a final risk-tiered output.")
    separator()


if __name__ == "__main__":
    main()
