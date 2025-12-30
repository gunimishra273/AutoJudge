import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import load_and_preprocess
from features import build_vectorizer

df = load_and_preprocess()

X = df["combined_text"]
y = df["problem_score"]

vectorizer = build_vectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("RF MAE:", mean_absolute_error(y_test, preds))
print("RF RMSE:", np.sqrt(mean_squared_error(y_test,preds)))

pickle.dump(vectorizer, open("models/vectorizer_rf.pkl", "wb"))
pickle.dump(model, open("models/regressor_rf.pkl", "wb"))
