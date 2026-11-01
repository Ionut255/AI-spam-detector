"""Download SMS data, train and evaluate a simple spam classifier."""
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "SMSSpamCollection"
MODEL_FILE = ROOT / "model" / "spam_pipeline.joblib"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset() -> None:
    """Download the original UCI dataset once; do not ship the data in Git."""
    if DATA_FILE.exists():
        return

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Descarc setul de date SMS Spam Collection de la UCI...")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=40) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))
    except Exception as exc:
        raise RuntimeError(
            "Nu am putut descărca datasetul. Descarcă-l manual de la "
            "https://archive.ics.uci.edu/dataset/228/sms+spam+collection "
            "și pune fișierul SMSSpamCollection în folderul data/."
        ) from exc


def load_data() -> pd.DataFrame:
    download_dataset()
    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        encoding_errors="replace",
        quoting=csv.QUOTE_NONE,
    )
    data = data.dropna(subset=["label", "message"])
    data = data[data["label"].isin(["ham", "spam"])]
    # Eliminate identical messages to avoid leakage between training and testing.
    return data.drop_duplicates(subset=["message"]).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main() -> None:
    data = load_data()
    print(f"Mesaje unice disponibile: {len(data)}")
    print(data["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\nRezultate pe datele de test (20% din date):")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='spam', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='spam', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, predictions, pos_label='spam', zero_division=0):.3f}")
    print("\nRaport detaliat:")
    print(classification_report(y_test, predictions, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
        ).ravel()

    print("\n--- Analiza erorilor ---")
    print(f"Mesaje normale detectate corect: {tn}")
    print(f"Mesaje normale marcate ca SPAM: {fp}")
    print(f"Mesaje SPAM nedetectate: {fn}")
    print(f"Mesaje SPAM detectate corect: {tp}")

    # Al doilea model: Multinomial Naive Bayes
    nb_pipeline = Pipeline(
       steps=[
           ("tfidf", TfidfVectorizer(
               ngram_range=(1, 2),
                max_features=5000
          )),
         ("classifier", MultinomialNB())
        ]
    )

    # Antrenarea modelului
    nb_pipeline.fit(X_train, y_train)

# Predicții pe aceleași date de test
    nb_predictions = nb_pipeline.predict(X_test)

# Evaluarea modelului
    print("\n--- Multinomial Naive Bayes ---")

    print(f"Accuracy : {accuracy_score(y_test, nb_predictions):.3f}")
    print(f"Precision: {precision_score(y_test, nb_predictions, pos_label='spam', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, nb_predictions, pos_label='spam', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, nb_predictions, pos_label='spam', zero_division=0):.3f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\nModel salvat: {MODEL_FILE}")


if __name__ == "__main__":
    main()
