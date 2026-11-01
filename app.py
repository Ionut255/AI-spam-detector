"""Streamlit interface for the SMS spam classifier."""

from pathlib import Path

import joblib
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "model" / "spam_pipeline.joblib"

st.set_page_config(page_title="Smart Spam Detector", page_icon="🛡️")
st.title("🛡️ Smart Spam Detector")
st.write("Introdu un SMS în engleză și modelul va estima dacă este spam.")

if not MODEL_FILE.exists():
    st.warning("Modelul nu este antrenat încă. Rulează mai întâi: `python train_model.py`.")
    st.stop()


@st.cache_resource
def load_model():
    # Load only the model generated locally by train_model.py.
    return joblib.load(MODEL_FILE)


model = load_model()
examples = {
    "Mesaj normal": "Hi! Are we still meeting at 5 pm tomorrow?",
    "Mesaj suspect": "Congratulations! Claim your FREE prize now! Call us today!",
}
selected = st.selectbox("Exemple rapide (opțional)", ["Scriu propriul mesaj"] + list(examples))
initial_text = examples.get(selected, "")
message = st.text_area("Mesaj de analizat", value=initial_text, height=140)

if st.button("Analizează mesajul", type="primary"):
    if not message.strip():
        st.warning("Scrie un mesaj înainte de analiză.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        spam_index = list(model.classes_).index("spam")
        spam_probability = float(probabilities[spam_index])

        if prediction == "spam":
            st.error("Rezultat: SPAM")
        else:
            st.success("Rezultat: NOT SPAM")
        st.metric("Probabilitate estimată de spam", f"{spam_probability:.1%}")
        st.progress(spam_probability)

        # Explicabilitate: contribuția cuvintelor prezente în mesaj
        # la scorul produs de modelul liniar Logistic Regression.
        with st.expander("🔎 De ce a decis modelul astfel?"):
            vectorizer = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]
            text_vector = vectorizer.transform([message]).tocoo()
            vocabulary = vectorizer.get_feature_names_out()

            # Pentru clasificarea binară, coeficienții corespund
            # celei de-a doua clase din classes_.
            spam_direction = 1 if classifier.classes_[1] == "spam" else -1
            weights = classifier.coef_[0]
            contributions = [
                (str(vocabulary[index]), float(weight * weights[index] * spam_direction))
                for index, weight in zip(text_vector.col, text_vector.data)
            ]

            spam_words = sorted(
                [(word, score) for word, score in contributions if score > 0],
                key=lambda pair: pair[1],
                reverse=True,
            )[:5]
            normal_words = sorted(
                [(word, score) for word, score in contributions if score < 0],
                key=lambda pair: pair[1],
            )[:5]

            left, right = st.columns(2)
            with left:
                st.markdown("**Indicii către SPAM**")
                if spam_words:
                    for word, score in spam_words:
                        st.write(f"• {word} (+{score:.3f})")
                else:
                    st.write("Nu au fost identificate indicii pozitive.")

            with right:
                st.markdown("**Indicii către NOT SPAM**")
                if normal_words:
                    for word, score in normal_words:
                        st.write(f"• {word} ({score:.3f})")
                else:
                    st.write("Nu au fost identificate indicii negative.")

            st.caption(
                "Aceste valori reprezintă contribuțiile termenilor în modelul liniar, "
                "nu explicații cauzale sau certitudini. Scorul include și bias-ul modelului."
            )

st.caption("Model demonstrativ antrenat pe SMS-uri în engleză; poate greși și nu este un filtru de securitate.")
