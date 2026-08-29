from training.generate_degradations import split_for


def test_source_split_is_stable():
    assert split_for("source-a") == split_for("source-a")
    assert split_for("source-a") in {"train", "validation", "test"}
