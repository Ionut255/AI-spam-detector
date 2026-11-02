"""Streamlit interface for the SMS spam classifier."""

from pathlib import Path

import joblib
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "model" / "spam_pipeline.joblib"

st.set_page_config(page_title="Smart Spam Detector", page_icon="🛡️")
st.title("🛡️ Smart Spam Detector")
st.write("Enter an English-language SMS message to check whether it is spam.")

@st.cache_resource(show_spinner="Preparing the AI model for the first launch...")
def load_model():
    """Load the fitted pipeline, training it first on a fresh deployment."""
    if not MODEL_FILE.is_file():
        # Git excludes model artifacts; cloud containers start with no model.
        # main() uses the reproducible train/test split from train_model.py.
        from train_model import main as train_model_main

        train_model_main()
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception:
    import logging

    logging.exception("Could not initialize the spam classifier")
    st.error(
        "Could not load or train the model. "
        "Check the application logs and access to the UCI dataset."
    )
    st.stop()
examples = {
    "Regular message": "Hi! Are we still meeting at 5 pm tomorrow?",
    "Suspicious message": "Congratulations! Claim your FREE prize now! Call us today!",
}
selected = st.selectbox("Quick examples (optional)", ["Write my own message"] + list(examples))
initial_text = examples.get(selected, "")
message = st.text_area("Message to analyze", value=initial_text, height=140)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        spam_index = list(model.classes_).index("spam")
        spam_probability = float(probabilities[spam_index])

        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM")
        st.metric("Estimated spam probability", f"{spam_probability:.1%}")
        st.progress(spam_probability)

        # Explainability: each recognized term’s contribution to the
        # score calculated by the Logistic Regression model.
        with st.expander("🔎 Why did the model make this prediction?"):
            vectorizer = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]
            text_vector = vectorizer.transform([message]).tocoo()
            vocabulary = vectorizer.get_feature_names_out()

            # In binary classification, coefficients correspond to
            # the second class in classes_.
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
                st.markdown("**Evidence toward SPAM**")
                if spam_words:
                    for word, score in spam_words:
                        st.write(f"• {word} (+{score:.3f})")
                else:
                    st.write("No positive contributions found.")

            with right:
                st.markdown("**Evidence toward NOT SPAM**")
                if normal_words:
                    for word, score in normal_words:
                        st.write(f"• {word} ({score:.3f})")
                else:
                    st.write("No negative contributions found.")

            st.caption(
                "These values show feature contributions within the linear model, "
                "not causal explanations or guarantees. The model score also includes an intercept."
            )

st.caption("Demo model trained on English-language SMS messages. Predictions may be wrong; do not use as a security filter.")
