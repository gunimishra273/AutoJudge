import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocess import load_and_preprocess
from features import build_vectorizer

df = load_and_preprocess()

X = df["combined_text"]
y = df["problem_class"]

vectorizer = build_vectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

pickle.dump(model, open("models/classifier.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

