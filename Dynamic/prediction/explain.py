def generate_reason(row):

    reasons = []

    if row.get("external_ratio", 0) > 0.5:
        reasons.append(
            "High external email communication"
        )

    if row.get("late_night_ratio", 0) > 0.3:
        reasons.append(
            "Frequent late-night activity"
        )

    if row.get("weekend_login_ratio", 0) > 0.3:
        reasons.append(
            "Frequent weekend access"
        )

    if row.get("copy_ratio", 0) > 0.5:
        reasons.append(
            "High file copy activity"
        )

    if row.get("sensitive_ratio", 0) > 0.5:
        reasons.append(
            "Sensitive file access"
        )

    if row.get("usb_ratio", 0) > 0.5:
        reasons.append(
            "Heavy USB device usage"
        )

    if row.get("unknown_device_ratio", 0) > 0.3:
        reasons.append(
            "Unknown device detected"
        )

    if len(reasons) == 0:
        reasons.append(
            "No major threat indicators detected"
        )

    return " | ".join(reasons)