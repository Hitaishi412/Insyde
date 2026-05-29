"""
=============================================================================
UEBA System — Step 5: Risk Scoring
=============================================================================
Purpose  : Assign a final, interpretable risk score to every user based on
           outputs from Steps 2 (Feature Engineering) and Step 4 (Isolation
           Forest anomaly scores). Implements a weighted scoring framework
           that prioritises HIGH-IMPACT security indicators over general noise,
           and buckets users into Low / Medium / High risk tiers.

Scoring philosophy
  - Security-critical signals (malicious URLs, BCC abuse, after-hours file
    copy) carry the highest weights because they most directly map to
    confirmed exfiltration behaviours in the CERT ground truth.
  - General noise (e.g., slightly elevated email volume) receives low weight.
  - The Isolation Forest anomaly score acts as a multiplier / tie-breaker,
    not as the primary score, to avoid over-trusting an unsupervised signal.
  - Risk buckets are chosen using adaptive percentile thresholds (configurable)
    so the system remains calibrated as new data flows in.

Output   : cert_risk_scored.csv  — users with final scores and risk tier
           risk_summary_report.txt — human-readable summary
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "cert_predictions.csv"
OUTPUT_FILE  = "cert_risk_scored.csv"
REPORT_FILE  = "risk_summary_report.txt"

# Risk bucket thresholds (percentiles of final risk score)
MEDIUM_PERCENTILE = 70   # bottom 70% = Low
HIGH_PERCENTILE   = 90   # top 10% = High,  70–90% = Medium

# Isolation Forest anomaly score column (set to None if not yet computed)
ANOMALY_SCORE_COL = "iforest_anomaly_score"

# ── Weight Registry ──────────────────────────────────────────────────────────
# Each entry: (column_name, weight, impact_tier, description)
# impact_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
# Weights will be normalised to sum to 1.0 within each tier group,
# then tier groups are combined with tier-level multipliers.

WEIGHT_REGISTRY = [
    # ── CRITICAL (direct exfiltration evidence) ────────────────────────────
    ("malicious_url_flag",          10.0, "CRITICAL", "Any malicious URL access"),
    ("malicious_url_zscore",         8.0, "CRITICAL", "Malicious URL spike vs population"),
    ("after_hours_file_spike",       8.0, "CRITICAL", "After-hours file copy anomaly"),
    ("bcc_abuse_flag",               7.0, "CRITICAL", "BCC usage rate anomaly (data hiding)"),
    ("email_file_exfil_chain",       9.0, "CRITICAL", "Email + file exfiltration combo"),
    ("exfil_chain_score",            9.0, "CRITICAL", "Multi-pillar exfiltration chain"),

    # ── HIGH (strong behavioural indicators) ──────────────────────────────
    ("ext_email_ratio_zscore",       6.0, "HIGH",     "External email ratio spike"),
    ("off_hours_email_ratio",        6.0, "HIGH",     "Off-hours email proportion"),
    ("risky_filetype_score",         5.0, "HIGH",     "EXE + ZIP file copy ratio"),
    ("after_hours_file_ratio",       6.0, "HIGH",     "After-hours file copy ratio"),
    ("other_pc_logon_flag",          5.0, "HIGH",     "Logons on other users PCs"),
    ("lateral_movement_score",       5.0, "HIGH",     "Multi-PC + other-user logon combo"),
    ("dual_off_hours_signal",        5.0, "HIGH",     "Concurrent off-hours email + file"),
    ("network_file_chain",           5.0, "HIGH",     "Malicious URL + risky file types"),

    # ── MEDIUM (suspicious but ambiguous signals) ──────────────────────────
    ("after_hours_logon_zscore",     3.0, "MEDIUM",   "After-hours logon frequency spike"),
    ("weekend_logon_flag",           3.0, "MEDIUM",   "Weekend logon presence"),
    ("files_copy_volume_zscore",     3.0, "MEDIUM",   "File copy volume spike"),
    ("domain_spread_zscore",         3.0, "MEDIUM",   "Unusually many distinct domains"),
    ("missing_disc_zscore",          2.0, "MEDIUM",   "Missing device disconnects"),
    ("never_used_device_flag",       3.0, "MEDIUM",   "New/never-used device suddenly active"),
    ("high_attachment_flag",         2.0, "MEDIUM",   "High email attachment density"),
    ("sensitive_doc_ratio",          2.0, "MEDIUM",   "High PDF/DOC copy ratio"),
    ("after_hours_browse_flag",      2.0, "MEDIUM",   "After-hours browsing"),

    # ── LOW (general behavioural noise) ───────────────────────────────────
    ("attachment_density",           1.0, "LOW",      "Average email attachment count"),
    ("multi_pc_flag",                1.0, "LOW",      "Multiple PCs used"),
    ("logon_erratic_flag",           1.0, "LOW",      "Erratic logon time variability"),
    ("recipient_spread_zscore",      1.0, "LOW",      "Wide email recipient spread"),
    ("device_conn_zscore",           0.5, "LOW",      "Device connection frequency"),
]

# Tier-level multipliers — these scale the normalised sub-scores
TIER_MULTIPLIERS = {
    "CRITICAL": 2.50,
    "HIGH":     1.50,
    "MEDIUM":   0.80,
    "LOW":      0.30,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if it exists, else a zero series."""
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series(0.0, index=df.index)


def _normalise_series(s: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]; handle zero-range gracefully."""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def compute_tier_score(df: pd.DataFrame, tier: str) -> pd.Series:
    """
    Compute a normalised score for one impact tier.
    Each feature is normalised independently, then averaged with its weight.
    """
    tier_entries = [(c, w, d) for c, w, t, d in WEIGHT_REGISTRY if t == tier]
    if not tier_entries:
        return pd.Series(0.0, index=df.index)

    total_weight = sum(w for _, w, _ in tier_entries)
    weighted_sum = pd.Series(0.0, index=df.index)

    for col, weight, _ in tier_entries:
        raw    = _safe_col(df, col).clip(lower=0)
        normed = _normalise_series(raw)
        weighted_sum += normed * (weight / total_weight)

    return weighted_sum.clip(0, 1)


def compute_weighted_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final weighted risk score by:
      1. Computing a normalised sub-score per tier.
      2. Scaling each tier sub-score by its multiplier.
      3. Summing across tiers and normalising to [0, 100].
      4. Optionally incorporating an Isolation Forest anomaly score.
    """
    out = df[["user"]].copy() if "user" in df.columns else df.iloc[:, :1].copy()

    # Per-tier sub-scores
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        score = compute_tier_score(df, tier)
        out[f"score_tier_{tier.lower()}"] = score

    # Weighted combination across tiers
    raw_total = sum(
        out[f"score_tier_{tier.lower()}"] * TIER_MULTIPLIERS[tier]
        for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    # Normalise to 0–100
    out["base_risk_score"] = _normalise_series(raw_total) * 100

    # ── Isolation Forest integration (optional) ──────────────────────────
    if ANOMALY_SCORE_COL and ANOMALY_SCORE_COL in df.columns:
        # IF scores are negative (more negative = more anomalous); flip sign
        if_raw  = -df[ANOMALY_SCORE_COL].fillna(0)
        if_norm = _normalise_series(if_raw) * 100

        # Blend: 70% weighted indicator score + 30% anomaly model score
        out["anomaly_model_score"] = if_norm
        out["final_risk_score"]    = (0.70 * out["base_risk_score"] +
                                      0.30 * if_norm).clip(0, 100)
    else:
        out["anomaly_model_score"] = np.nan
        out["final_risk_score"]    = out["base_risk_score"]

    return out


def assign_risk_buckets(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Classify users into Low / Medium / High risk buckets using
    adaptive percentile thresholds from the score distribution.
    """
    s = scores["final_risk_score"]
    medium_threshold = np.percentile(s, MEDIUM_PERCENTILE)
    high_threshold   = np.percentile(s, HIGH_PERCENTILE)

    conditions = [
        s >= high_threshold,
        s >= medium_threshold,
    ]
    choices = ["High", "Medium"]
    scores["risk_tier"]       = np.select(conditions, choices, default="Low")
    scores["medium_threshold"] = medium_threshold
    scores["high_threshold"]   = high_threshold

    return scores


def compute_indicator_flags(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Attach human-readable indicator flags to each user explaining
    which signals drove their risk score.
    """
    flags = []
    for _, row in df.iterrows():
        active = []
        for col, weight, tier, desc in WEIGHT_REGISTRY:
            if col not in df.columns:
                continue
            val = row[col]
            if pd.isna(val) or val <= 0:
                continue
            # Threshold: flag if value is in top-25% of its column
            threshold = df[col].quantile(0.75)
            if val >= threshold and tier in ("CRITICAL", "HIGH"):
                active.append(f"[{tier}] {desc}")
        flags.append(" | ".join(active) if active else "No significant indicators")

    scores["risk_indicators"] = flags
    return scores


def generate_report(scores: pd.DataFrame) -> str:
    """Produce a human-readable plain-text risk summary report."""
    tier_counts = scores["risk_tier"].value_counts()
    high_users  = scores[scores["risk_tier"] == "High"].sort_values(
        "final_risk_score", ascending=False
    )
    med_threshold  = scores["medium_threshold"].iloc[0]
    high_threshold = scores["high_threshold"].iloc[0]

    lines = [
        "=" * 70,
        "  UEBA SYSTEM — RISK SCORING REPORT",
        "=" * 70,
        "",
        f"  Total users analysed  : {len(scores):,}",
        f"  Medium threshold (P{MEDIUM_PERCENTILE}) : {med_threshold:.2f}",
        f"  High threshold   (P{HIGH_PERCENTILE}) : {high_threshold:.2f}",
        "",
        "  RISK DISTRIBUTION",
        "  " + "-" * 40,
        f"  {'Risk Tier':<12} {'Count':>8} {'% of Total':>12}",
        "  " + "-" * 40,
    ]

    for tier in ["High", "Medium", "Low"]:
        cnt = tier_counts.get(tier, 0)
        pct = 100 * cnt / len(scores)
        lines.append(f"  {tier:<12} {cnt:>8,} {pct:>11.1f}%")

    lines += [
        "",
        "  TOP HIGH-RISK USERS",
        "  " + "-" * 68,
        f"  {'User':<12} {'Risk Score':>10} {'Tier':<8}  Key Indicators",
        "  " + "-" * 68,
    ]

    for _, row in high_users.head(20).iterrows():
        indicators = str(row.get("risk_indicators", ""))[:45]
        lines.append(
            f"  {str(row.get('user','?')):<12} {row['final_risk_score']:>10.2f} "
            f"{row['risk_tier']:<8}  {indicators}"
        )

    lines += [
        "",
        "  SCORING WEIGHT BREAKDOWN",
        "  " + "-" * 68,
        f"  {'Tier':<10} {'Multiplier':>12}  Description",
        "  " + "-" * 68,
    ]
    for tier, mult in TIER_MULTIPLIERS.items():
        count = sum(1 for _, _, t, _ in WEIGHT_REGISTRY if t == tier)
        lines.append(f"  {tier:<10} {mult:>12.2f}  ({count} indicators)")

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[Step 5] Loading feature-engineered data …")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {df.shape[0]:,} users × {df.shape[1]} columns")

    print("[Step 5] Computing tier sub-scores …")
    scores = compute_weighted_risk_score(df)

    print("[Step 5] Assigning risk buckets …")
    scores = assign_risk_buckets(scores)

    print("[Step 5] Attaching indicator flags …")
    scores = compute_indicator_flags(df, scores)

    # Merge back with original features for downstream use
    df_out = df.merge(
        scores.drop(columns=["medium_threshold", "high_threshold"]),
        on="user",
        how="left"
    ) if "user" in df.columns else pd.concat([df, scores], axis=1)

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"[Step 5] Risk scores saved → {OUTPUT_FILE}")

    # Generate and save report
    report = generate_report(scores)
    Path(REPORT_FILE).write_text(report)
    print(f"[Step 5] Summary report  → {REPORT_FILE}")
    print()
    print(report)

    # Return for inspection
    return df_out, scores


if __name__ == "__main__":
    df_out, scores = main()
