import pickle

# ===== Load FINAL models (once) =====

# Final classifier (SVM)
classifier = pickle.load(open("models/classifier_svm.pkl", "rb"))
clf_vectorizer = pickle.load(open("models/vectorizer_svm.pkl", "rb"))

# Final regressor (Random Forest)
regressor = pickle.load(open("models/regressor_rf.pkl", "rb"))
reg_vectorizer = pickle.load(open("models/vectorizer_rf.pkl", "rb"))


def predict_difficulty(title, description, input_description, output_description):
    """
    Predict difficulty class and difficulty score
    using FINAL trained models.
    """

    # Combine text exactly like training
    text = f"{title} {description} {input_description} {output_description}"

    # ----- Classification -----
    X_clf = clf_vectorizer.transform([text])
    difficulty_class = classifier.predict(X_clf)[0]

    # ----- Regression -----
    X_reg = reg_vectorizer.transform([text])
    difficulty_score = regressor.predict(X_reg)[0]

    return difficulty_class, round(float(difficulty_score), 2)
