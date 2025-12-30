import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import load_and_preprocess
from features import build_vectorizer

# Load and preprocess data
df = load_and_preprocess()

X_text = df["combined_text"]
y = df["problem_score"]

# TF-IDF features (sparse, perfect for Ridge)
vectorizer = build_vectorizer()
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Ridge Regression (linear, L2-regularized)
model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Ridge MAE:", mae)
print("Ridge RMSE:", rmse)

# Save final regressor
pickle.dump(model, open("models/regressor_ridge.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer_ridge.pkl", "wb"))
