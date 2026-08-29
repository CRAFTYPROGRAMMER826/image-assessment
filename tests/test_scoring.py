from app.core.scoring import aggregate


def test_score_is_bounded_and_all_issues_are_returned():
    probabilities = {"blur": 1.0, "noise": 1.0, "severe_degradation": 1.0, "potential_visual_defect": 1.0}
    score, label, issues = aggregate(probabilities, {})
    assert 0 <= score <= 100
    assert label == "DEGRADED" or label == "POTENTIALLY_DEFECTIVE"
    assert len(issues) == len(probabilities)


def test_good_probabilities_are_acceptable():
    probabilities = {"blur": 0.01, "underexposure": 0.01, "overexposure": 0.01, "noise": 0.01, "severe_degradation": 0.01}
    score, label, _ = aggregate(probabilities, {})
    assert score >= 80
    assert label == "ACCEPTABLE"
