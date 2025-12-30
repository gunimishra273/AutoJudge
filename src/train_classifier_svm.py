import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from preprocess import load_and_preprocess
from features import build_vectorizer_svm

df = load_and_preprocess()

X = df["combined_text"]
y = df["problem_class"]

vectorizer = build_vectorizer_svm()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LinearSVC(
    C=0.3,           
    max_iter=5000
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))

pickle.dump(model, open("models/classifier_svm.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer_svm.pkl", "wb"))
