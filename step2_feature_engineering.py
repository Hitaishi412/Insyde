"""
=============================================================================
UEBA System — Step 2: Feature Engineering
=============================================================================
Purpose  : Derive MITRE ATT&CK-aligned behavioural features, with emphasis
           on the 'Exfiltration' tactic (TA0010), from the CERT dataset.
           Features are grouped into four pillars:
             1. Email Exfiltration Indicators  (T1048, T1567)
             2. File / Removable-Media Staging  (T1074, T1052)
             3. Off-Hours / After-Hours Activity (T1078 behavioural signal)
             4. Network / Web Anomalies          (T1071)
Output   : cert_features.csv  — original columns + all engineered features
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "cert_cleaned.csv"
OUTPUT_FILE = "cert_features.csv"

# Working-hours definition (inclusive, 24-h clock)
WORK_HOUR_START = 8
WORK_HOUR_END   = 18

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[Step 2] Loading data …")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {df.shape[0]:,} users × {df.shape[1]} columns")


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR A — Email Exfiltration Indicators
# MITRE: T1048 (Exfiltration Over Alternative Protocol),
#        T1567 (Exfiltration Over Web Service)
# ═══════════════════════════════════════════════════════════════════════════

def build_email_exfiltration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives features that flag abnormal outbound email behaviour consistent
    with data exfiltration via email channels.
    """
    feat = pd.DataFrame(index=df.index)

    # 1a. External email ratio spike
    #     High ratio of external vs total recipients = potential exfiltration
    ext_ratio = df["external_emails_ratio"].fillna(0)
    feat["ext_email_ratio"] = ext_ratio

    # Z-score-based spike: how many std-devs above the population mean?
    mean_ext = ext_ratio.mean()
    std_ext  = ext_ratio.std() + 1e-9
    feat["ext_email_ratio_zscore"] = (ext_ratio - mean_ext) / std_ext

    # 1b. Off-hours email volume (absolute + ratio of total sent)
    off_hrs_emails = df["emails_sent_off_hours"].fillna(0)
    total_emails   = df["emails_sent_per_day"].fillna(0) * df.get("active_days", 1).fillna(1)

    feat["off_hours_email_count"]  = off_hrs_emails
    feat["off_hours_email_ratio"]  = off_hrs_emails / (total_emails + 1)

    # 1c. BCC abuse rate — BCC hides recipients, common in insider threat cases
    feat["bcc_abuse_flag"]  = (df["bcc_usage_rate"] > df["bcc_usage_rate"].quantile(0.90)).astype(int)
    feat["bcc_usage_rate"]  = df["bcc_usage_rate"].fillna(0)

    # 1d. Attachment density — large # of attachments signals data staging
    att_avg = df["attachment_count_avg"].fillna(0)
    feat["attachment_density"]       = att_avg
    feat["high_attachment_flag"]     = (att_avg > att_avg.quantile(0.90)).astype(int)

    # 1e. Unique recipient spread — many distinct recipients = spray-and-exfil
    recip = df["unique_recipients_count"].fillna(0)
    feat["unique_recipient_count"]   = recip
    feat["recipient_spread_zscore"]  = (recip - recip.mean()) / (recip.std() + 1e-9)

    # 1f. Composite Email Exfiltration Score (weighted sum, scaled 0–1)
    feat["email_exfil_composite"] = (
        0.35 * feat["ext_email_ratio_zscore"].clip(0, 5) / 5 +
        0.25 * feat["off_hours_email_ratio"].clip(0, 1) +
        0.20 * feat["bcc_usage_rate"].clip(0, 1) +
        0.10 * feat["high_attachment_flag"] +
        0.10 * feat["recipient_spread_zscore"].clip(0, 5) / 5
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR B — File / Removable-Media Staging
# MITRE: T1074 (Data Staged), T1052 (Exfiltration over Physical Medium)
# ═══════════════════════════════════════════════════════════════════════════

def build_file_staging_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies large-scale file copying, risky file types, and after-hours
    file movement — hallmarks of data staging before exfiltration.
    """
    feat = pd.DataFrame(index=df.index)

    total_files  = df["total_files_copied"].fillna(0)
    avg_files    = df["avg_files_copied_per_day"].fillna(0)
    aft_hrs_ratio = df["after_hours_file_copy_ratio"].fillna(0)

    # 2a. Raw volume
    feat["total_files_copied"]         = total_files
    feat["avg_files_per_day"]          = avg_files

    # 2b. Volume spike (z-score)
    feat["files_copy_volume_zscore"] = (
        (total_files - total_files.mean()) / (total_files.std() + 1e-9)
    )

    # 2c. After-hours file copy ratio — key Exfiltration signal
    feat["after_hours_file_ratio"]     = aft_hrs_ratio
    feat["after_hours_file_spike"]     = (aft_hrs_ratio > aft_hrs_ratio.quantile(0.85)).astype(int)

    # 2d. Risky file-type mix (executables + archives = staging/packaging)
    exe_ratio = df["filetype_exe"].fillna(0) / (total_files + 1)
    zip_ratio = df["filetype_zip"].fillna(0) / (total_files + 1)
    feat["exe_file_ratio"]             = exe_ratio
    feat["zip_file_ratio"]             = zip_ratio
    feat["risky_filetype_score"]       = (exe_ratio + zip_ratio).clip(0, 1)

    # 2e. Sensitive doc ratio (PDF + DOC)
    pdf_ratio = df["filetype_pdf"].fillna(0) / (total_files + 1)
    doc_ratio = df["filetype_doc"].fillna(0) / (total_files + 1)
    feat["sensitive_doc_ratio"]        = (pdf_ratio + doc_ratio).clip(0, 1)

    # 2f. Composite File Staging Score
    feat["file_staging_composite"] = (
        0.30 * feat["files_copy_volume_zscore"].clip(0, 5) / 5 +
        0.30 * feat["after_hours_file_ratio"].clip(0, 1) +
        0.25 * feat["risky_filetype_score"] +
        0.15 * feat["sensitive_doc_ratio"]
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR C — Off-Hours / Behavioural Temporal Anomalies
# MITRE: T1078 (Valid Accounts) — behavioural signal of misuse
# ═══════════════════════════════════════════════════════════════════════════

def build_temporal_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Captures off-hours logon patterns, weekend activity, and multi-PC usage
    that deviate from normal working behaviour.
    """
    feat = pd.DataFrame(index=df.index)

    after_hrs_logons = df["after_hours_logons"].fillna(0)
    weekend_logons   = df["weekend_logons"].fillna(0)
    other_pc_logons  = df["logons_on_other_user_pcs"].fillna(0)
    logon_std        = df["logon_std_dev_hours"].fillna(0)

    # 3a. After-hours logon intensity
    feat["after_hours_logon_count"]   = after_hrs_logons
    feat["after_hours_logon_zscore"]  = (
        (after_hrs_logons - after_hrs_logons.mean()) / (after_hrs_logons.std() + 1e-9)
    )

    # 3b. Weekend logon anomaly
    feat["weekend_logon_count"]       = weekend_logons
    feat["weekend_logon_flag"]        = (weekend_logons > weekend_logons.quantile(0.80)).astype(int)

    # 3c. Other-user PC logons — lateral movement signal
    feat["other_pc_logon_count"]      = other_pc_logons
    feat["other_pc_logon_flag"]       = (other_pc_logons > 0).astype(int)

    # 3d. Logon time variability — erratic hours = concealment behaviour
    feat["logon_time_std"]            = logon_std
    feat["logon_erratic_flag"]        = (logon_std > logon_std.quantile(0.85)).astype(int)

    # 3e. Late-hour device connection use
    feat["late_hour_device_use"]      = df["late_hour_device_use"].fillna(0)

    # 3f. Off-hours intensity composite (from existing engineered column)
    feat["off_hours_intensity"]       = df.get("after_hours_intensity", pd.Series(0, index=df.index)).fillna(0)

    # 3g. Composite Temporal Anomaly Score
    feat["temporal_anomaly_composite"] = (
        0.30 * feat["after_hours_logon_zscore"].clip(0, 5) / 5 +
        0.20 * feat["weekend_logon_flag"] +
        0.25 * feat["other_pc_logon_flag"] +
        0.15 * feat["logon_erratic_flag"] +
        0.10 * feat["late_hour_device_use"].clip(0, 1)
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR D — Network / Web Anomalies
# MITRE: T1071 (Application Layer Protocol), T1567 (Web Service Exfil)
# ═══════════════════════════════════════════════════════════════════════════

def build_network_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags malicious URL access, excessive domain browsing, and after-hours
    web activity as network-layer exfiltration signals.
    """
    feat = pd.DataFrame(index=df.index)

    malicious_urls   = df["avg_malicious_urls_per_day"].fillna(0)
    total_malicious  = df["total_malicious_urls"].fillna(0)
    distinct_domains = df["avg_distinct_domains_per_day"].fillna(0)
    aft_hrs_browse   = df["avg_after_hours_browsing_per_day"].fillna(0)

    # 4a. Malicious URL access (hard security signal)
    feat["malicious_url_avg"]         = malicious_urls
    feat["malicious_url_total"]       = total_malicious
    feat["malicious_url_flag"]        = (total_malicious > 0).astype(int)

    # 4b. Malicious URL spike (z-score among users who accessed any)
    feat["malicious_url_zscore"]      = (
        (malicious_urls - malicious_urls.mean()) / (malicious_urls.std() + 1e-9)
    )

    # 4c. Domain spread anomaly — many distinct domains = C2 / staging recon
    feat["distinct_domain_avg"]       = distinct_domains
    feat["domain_spread_zscore"]      = (
        (distinct_domains - distinct_domains.mean()) / (distinct_domains.std() + 1e-9)
    )

    # 4d. After-hours browsing
    feat["after_hours_browsing_avg"]  = aft_hrs_browse
    feat["after_hours_browse_flag"]   = (
        aft_hrs_browse > aft_hrs_browse.quantile(0.80)
    ).astype(int)

    # 4e. Composite Network Anomaly Score
    feat["network_anomaly_composite"] = (
        0.40 * feat["malicious_url_zscore"].clip(0, 5) / 5 +
        0.20 * feat["malicious_url_flag"] +
        0.25 * feat["domain_spread_zscore"].clip(0, 5) / 5 +
        0.15 * feat["after_hours_browse_flag"]
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR E — Device / Endpoint Anomalies
# MITRE: T1052 (Removable Media), T1078 (Valid Accounts)
# ═══════════════════════════════════════════════════════════════════════════

def build_device_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags unusual device usage patterns: multiple PCs, missing disconnects,
    and never-used devices suddenly becoming active.
    """
    feat = pd.DataFrame(index=df.index)

    distinct_pcs      = df["distinct_pcs_used"].fillna(0)
    missing_disc      = df["missing_disconnects"].fillna(0)
    never_used_device = df["never_used_device"].fillna(0)
    avg_conn          = df["avg_daily_device_connections"].fillna(0)

    feat["distinct_pcs_used"]         = distinct_pcs
    feat["multi_pc_flag"]             = (distinct_pcs > 1).astype(int)
    feat["missing_disconnects"]       = missing_disc
    feat["missing_disc_zscore"]       = (
        (missing_disc - missing_disc.mean()) / (missing_disc.std() + 1e-9)
    )
    feat["never_used_device_flag"]    = (never_used_device > 0).astype(int)
    feat["avg_device_connections"]    = avg_conn
    feat["device_conn_zscore"]        = (
        (avg_conn - avg_conn.mean()) / (avg_conn.std() + 1e-9)
    )

    feat["device_anomaly_composite"] = (
        0.25 * feat["multi_pc_flag"] +
        0.25 * feat["missing_disc_zscore"].clip(0, 5) / 5 +
        0.25 * feat["never_used_device_flag"] +
        0.25 * feat["device_conn_zscore"].clip(0, 5) / 5
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PILLAR F — Cross-Pillar Interaction Features (Exfiltration chain signals)
# ═══════════════════════════════════════════════════════════════════════════

def build_interaction_features(email_f, file_f, temporal_f, network_f, device_f) -> pd.DataFrame:
    """
    Constructs higher-order interaction features that capture multi-vector
    attack chains — a hallmark of sophisticated insider threats.
    """
    feat = pd.DataFrame(index=email_f.index)

    # Email + File: staging + emailing = classic exfiltration combo
    feat["email_file_exfil_chain"] = (
        email_f["email_exfil_composite"] * file_f["file_staging_composite"]
    )

    # After-hours email + after-hours file copy (concurrent off-hours activity)
    feat["dual_off_hours_signal"] = (
        email_f["off_hours_email_ratio"] * temporal_f["temporal_anomaly_composite"]
    )

    # Malicious URL + risky file types (network recon + data packaging)
    feat["network_file_chain"] = (
        network_f["malicious_url_flag"] * file_f["risky_filetype_score"]
    )

    # Multi-PC + other-user logons (lateral movement)
    feat["lateral_movement_score"] = (
        device_f["multi_pc_flag"] * temporal_f["other_pc_logon_flag"]
    )

    # Combined Exfiltration Chain Indicator (all pillars contributing)
    feat["exfil_chain_score"] = (
        0.30 * email_f["email_exfil_composite"] +
        0.25 * file_f["file_staging_composite"] +
        0.20 * temporal_f["temporal_anomaly_composite"] +
        0.15 * network_f["network_anomaly_composite"] +
        0.10 * device_f["device_anomaly_composite"]
    ).clip(0, 1)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Orchestrate feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load
    df = pd.read_csv(INPUT_FILE)
    n_original = df.shape[1]

    # Build pillars
    print("[Step 2] Engineering features …")
    email_f    = build_email_exfiltration_features(df)
    file_f     = build_file_staging_features(df)
    temporal_f = build_temporal_anomaly_features(df)
    network_f  = build_network_anomaly_features(df)
    device_f   = build_device_anomaly_features(df)
    interact_f = build_interaction_features(email_f, file_f, temporal_f, network_f, device_f)

    # Concatenate all engineered features
    engineered = pd.concat(
        [email_f, file_f, temporal_f, network_f, device_f, interact_f],
        axis=1
    )

    # Scale composite scores with RobustScaler (resilient to outlier users)
    composite_cols = [c for c in engineered.columns if "composite" in c or "score" in c]
    scaler = RobustScaler()
    engineered[composite_cols] = scaler.fit_transform(engineered[composite_cols])

    # Merge with original dataframe
    df_out = pd.concat([df, engineered], axis=1)

    # Remove duplicate columns (keep engineered versions)
    df_out = df_out.loc[:, ~df_out.columns.duplicated(keep="last")]

    df_out.to_csv(OUTPUT_FILE, index=False)

    n_new = df_out.shape[1] - n_original
    print(f"[Step 2] Done. Added {n_new} engineered features.")
    print(f"[Step 2] Output saved → {OUTPUT_FILE}")
    print(f"\n{'─'*60}")
    print(f"{'Feature Group':<35} {'Count':>5}")
    print(f"{'─'*60}")
    print(f"{'Email Exfiltration (TA0010/T1048)':<35} {len(email_f.columns):>5}")
    print(f"{'File Staging (T1074/T1052)':<35} {len(file_f.columns):>5}")
    print(f"{'Temporal/Behavioural (T1078)':<35} {len(temporal_f.columns):>5}")
    print(f"{'Network Anomalies (T1071/T1567)':<35} {len(network_f.columns):>5}")
    print(f"{'Device Anomalies (T1052)':<35} {len(device_f.columns):>5}")
    print(f"{'Cross-Pillar Interaction Chains':<35} {len(interact_f.columns):>5}")
    print(f"{'─'*60}")
    print(f"{'TOTAL ENGINEERED FEATURES':<35} {n_new:>5}")
    print(f"{'─'*60}")

    return df_out


if __name__ == "__main__":
    result = main()
