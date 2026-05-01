# UEBA System — CERT Dataset
## Insider Threat Detection via Behavioral Analytics

---

### Project Overview

This project implements a **User and Entity Behavior Analytics (UEBA)** pipeline
on the CERT Insider Threat dataset using Isolation Forest for anomaly detection.
Features are explicitly mapped to MITRE ATT&CK tactics, with emphasis on the
**Exfiltration tactic (TA0010)**.

---

### Files in this Package

| File | Description |
|------|-------------|
| `step2_feature_engineering.py` | MITRE-aligned feature derivation across 5 behavioural pillars |
| `step5_risk_scoring.py` | Weighted risk scoring with Low/Medium/High tier classification |
| `cert_cleaned.csv` | Input: pre-cleaned CERT dataset (place in working directory) |

---

### Step 2 — Feature Engineering

**Input:** `cert_cleaned.csv`  
**Output:** `cert_features.csv`

Derives **6 feature pillars** mapped to MITRE ATT&CK:

| Pillar | MITRE Techniques | Key Features |
|--------|------------------|--------------|
| Email Exfiltration | T1048, T1567 | External ratio spike, off-hours email, BCC abuse |
| File Staging | T1074, T1052 | After-hours file copy, EXE/ZIP ratios, volume spike |
| Temporal Anomalies | T1078 | Off-hours logons, weekend activity, logon variability |
| Network Anomalies | T1071, T1567 | Malicious URLs, domain spread, after-hours browsing |
| Device Anomalies | T1052 | Multi-PC usage, missing disconnects, new device activity |
| Cross-Pillar Chains | TA0010 | Email+File combo, lateral movement, dual off-hours |

```bash
python step2_feature_engineering.py
```

---

### Step 5 — Risk Scoring

**Input:** `cert_features.csv` (output of Step 2)  
**Output:** `cert_risk_scored.csv`, `risk_summary_report.txt`

Implements a **4-tier weighted scoring framework**:

| Impact Tier | Multiplier | Example Indicators |
|-------------|------------|--------------------|
| CRITICAL | 2.50× | Malicious URL access, BCC abuse, exfiltration chain |
| HIGH | 1.50× | External email spike, after-hours file copy, lateral movement |
| MEDIUM | 0.80× | After-hours logons, risky file types, domain spread |
| LOW | 0.30× | General volume noise, device count |

**Risk Buckets** (adaptive percentile thresholds):
- 🟢 **Low** — bottom 70% of risk scores
- 🟡 **Medium** — 70th–90th percentile
- 🔴 **High** — top 10% (priority investigation list)

```bash
python step5_risk_scoring.py
```

---

### Full Pipeline Order

```
Step 1: Data Preprocessing  →  cert_cleaned.csv
Step 2: Feature Engineering →  cert_features.csv        ← this package
Step 3: EDA / Correlation Analysis
Step 4: Isolation Forest    →  adds iforest_anomaly_score column
Step 5: Risk Scoring        →  cert_risk_scored.csv      ← this package
```

**To integrate the Isolation Forest score into Step 5**, set this in `step5_risk_scoring.py`:
```python
ANOMALY_SCORE_COL = "iforest_anomaly_score"
```
The final risk score will then blend 70% indicator score + 30% model score.

---

### Dependencies

```bash
pip install pandas numpy scikit-learn
```

---

### MITRE ATT&CK Reference

- **TA0010** — Exfiltration (tactic)
- **T1048** — Exfiltration Over Alternative Protocol
- **T1052** — Exfiltration over Physical Medium
- **T1067** — Exfiltration Over Web Service
- **T1074** — Data Staged
- **T1071** — Application Layer Protocol (C2 / recon)
- **T1078** — Valid Accounts (behavioural misuse)
