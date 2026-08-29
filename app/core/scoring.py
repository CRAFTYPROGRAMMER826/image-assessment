from __future__ import annotations

TARGET_NAMES = ["blur", "underexposure", "overexposure", "noise", "severe_degradation"]
PENALTY_WEIGHTS = {
    "blur": 20.0,
    "underexposure": 20.0,
    "overexposure": 20.0,
    "noise": 15.0,
    "severe_degradation": 25.0,
    "potential_visual_defect": 20.0,
}


def severity(confidence: float, detected: bool) -> str:
    if not detected:
        return "none"
    if confidence < 0.70:
        return "low"
    if confidence < 0.85:
        return "medium"
    return "high"


def aggregate(probabilities: dict[str, float], thresholds: dict[str, float]) -> tuple[float, str, list[dict]]:
    issues: list[dict] = []
    penalty = 0.0
    for issue_type, confidence in probabilities.items():
        threshold = float(thresholds.get(issue_type, 0.5))
        detected = confidence >= threshold
        issues.append(
            {
                "type": issue_type,
                "detected": detected,
                "confidence": round(float(confidence), 4),
                "severity": severity(float(confidence), detected),
            }
        )
        penalty += PENALTY_WEIGHTS.get(issue_type, 0.0) * float(confidence)
    score = round(max(0.0, min(100.0, 100.0 - penalty)), 1)
    label = "ACCEPTABLE" if score >= 80 else "DEGRADED" if score >= 50 else "POTENTIALLY_DEFECTIVE"
    return score, label, issues
