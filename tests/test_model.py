"""Simple automated test that runs without downloading the dataset."""

from train_model import build_pipeline


def test_pipeline_predict_and_probabilities():
    messages = [
        "Free cash prize claim now",
        "Win money click here",
        "Limited offer claim your reward",
        "Congratulations you won a gift",
        "Hi are we meeting tomorrow",
        "See you at school today",
        "Can you call me tonight",
        "Let's have lunch together",
    ]
    labels = ["spam"] * 4 + ["ham"] * 4
    model = build_pipeline()
    model.fit(messages, labels)

    predictions = model.predict(["Hello friend", "Win a free prize"])
    probabilities = model.predict_proba(["Hello friend", "Win a free prize"])

    assert len(predictions) == 2
    assert set(predictions).issubset({"ham", "spam"})
    assert probabilities.shape == (2, 2)
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in probabilities)
