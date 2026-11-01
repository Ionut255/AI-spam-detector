# Model evaluation and selection

## Goal

Build a simple text-classification baseline and choose a model that balances spam detection with false alarms. The positive class is **spam**.

## Experimental setup

- Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- Identical SMS messages are deduplicated before splitting.
- Stratified train/test split: 80% training / 20% testing, random state 42.
- Text features: TF-IDF, unigrams and bigrams, maximum 5,000 features.
- Classifiers: Logistic Regression (`class_weight="balanced"`) and Multinomial Naive Bayes.
- Test examples: 1,035 messages (904 ham / 131 spam).

## Observed hold-out results

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 0.979 | 0.968 |
| Precision (spam) | 0.916 | 1.000 |
| Recall (spam) | 0.916 | 0.748 |
| F1-score (spam) | 0.916 | 0.856 |

**Logistic Regression — confusion matrix** (rows: actual; columns: predicted):

|  | Ham | Spam |
|---|---:|---:|
| Ham | 893 | 11 |
| Spam | 11 | 120 |

Thus, the model made 11 false positives (ham classified as spam) and 11 false negatives (spam classified as ham).

**Naive Bayes — approximate counts inferred from rounded metrics:** about 98 detected spam, 33 missed spam, 0 false positives. Recompute its confusion matrix from raw predictions if exact counts are needed.

## Selection rationale

For a **reviewable, non-destructive spam detector**, Logistic Regression is the preferred default. It detected a substantially larger proportion of spam and achieved a better F1-score, while generating 11 false alarms on this test set.

In a system that **automatically deletes** or blocks messages, false positives may be more harmful. A different probability threshold or even a different model might be warranted. The choice must follow application requirements.

## Evaluation cautions

- The test set should not be repeatedly used for tuning thresholds or hyperparameters; use cross-validation or a separate validation set.
- The results above were provided by the local training run and have not been independently reproduced in this documentation package.
- High accuracy can conceal poor spam recall because ham messages are much more frequent.
- Model probabilities are not known to be calibrated.
- The classifier was tested on English SMS messages, not production emails or multilingual messages.

## Future experiments

1. Plot precision-recall curves on a validation set and choose a threshold aligned with false-positive costs.
2. Compare cross-validated F1 and recall, with variability across folds.
3. Review examples of false positives and false negatives without publishing private message data.
4. Calibrate predicted probabilities and test drift on newer data.
