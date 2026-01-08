
# AutoJudge

**AI-Powered Programming Problem Difficulty Prediction**

AutoJudge is a machine learning system that automatically predicts the **difficulty class** (Easy / Medium / Hard) and a **numerical difficulty score** for programming problems using only their **textual descriptions**.

The project is inspired by online competitive programming platforms (Codeforces, CodeChef, Kattis), where problem difficulty is typically assigned manually. AutoJudge aims to **automate this process** using Natural Language Processing (NLP) and classical machine learning models.

---

## 🚀 Features

* Predicts **problem difficulty class** (Easy / Medium / Hard)
* Predicts a **numerical difficulty score**
* Uses **only text input** (no code or metadata)
* Implements **baseline and improved models**
* Clean and interactive **Flask web interface**
* No deep learning — fully interpretable ML models

---

## 📂 Project Structure

```
AutoJudge/
│
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── README.md
│
├── data/
│   └── problems.csv            # Preprocessed dataset
│
├── src/
│   ├── jsonl_to_csv.py          # Dataset conversion
│   ├── preprocess.py            # Text cleaning & preprocessing
│   ├── features.py              # TF-IDF feature extraction
│   ├── train_classifier.py      # Baseline classifier (Logistic Regression)
│   ├── train_classifier_svm.py  # Improved classifier (SVM)
│   ├── train_regressor.py       # Baseline regressor (Ridge)
│   ├── train_regressor_rf.py    # Improved regressor (Random Forest)
│   └── predict.py               # Prediction pipeline
│
├── models/
│   ├── classifier.pkl
│   ├── classifier_svm.pkl
│   ├── vectorizer.pkl
│   ├── vectorizer_svm.pkl
│   ├── regressor_ridge.pkl
│   ├── regressor_rf.pkl
│   ├── vectorizer_ridge.pkl
│   └── vectorizer_rf.pkl
│
└── templates/
    └── index.html               # Web UI
```

---

## 📊 Dataset

The dataset is sourced from:

**TaskComplexityEval-24**
[https://github.com/AREEG94FAHAD/TaskComplexityEval-24](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)

Each sample contains:

* `title`
* `description`
* `input_description`
* `output_description`
* `problem_class` (easy / medium / hard)
* `problem_score` (numerical)

The raw JSONL file is converted into CSV using `jsonl_to_csv.py`, followed by text preprocessing and feature extraction.

---

## ⚙️ Data Preprocessing

* Combined all text fields into a single input
* Removed missing values
* Normalized and cleaned text
* Generated TF-IDF features with unigrams and bigrams

---

## 🧠 Models Used

### 🔹 Classification (Difficulty Class)

**Baseline Model**

* Logistic Regression
* Accuracy ≈ **0.496**

**Improved Model**

* Support Vector Machine (LinearSVC)
* Accuracy ≈ **0.503**
* Uses class balancing and tuned regularization

---

### 🔹 Regression (Difficulty Score)

**Baseline Model**

* Ridge Regression
* MAE ≈ **1.72**
* RMSE ≈ **2.06**

**Improved Model**

* Random Forest Regressor
* MAE ≈ **1.71**
* RMSE ≈ **2.05**

Although numerical improvements are modest, Random Forest captures non-linear relationships better than linear regression.

---

## 📈 Evaluation Metrics

* **Classification**

  * Accuracy
  * Confusion Matrix
  * Precision / Recall / F1-Score

* **Regression**

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

---

## 🌐 Web Interface

The project includes a Flask-based web application that allows users to:

1. Enter:

   * Problem Title
   * Problem Description
   * Input Description
   * Output Description
2. Click **Predict Difficulty**
3. View:

   * Predicted Difficulty Class (color-coded)
   * Predicted Difficulty Score

The UI is clean, responsive, and designed for easy demonstration.

---
## 📄 Project Report
The detailed project report explaining the problem statement, dataset, preprocessing, feature engineering, models, evaluation metrics, and web interface is available here:

👉 [Project Report (PDF)](https://drive.google.com/file/d/1zeX8r4hvQt6gNe2tsB2ktc5Z5dNY0guq/view?usp=sharing)

---

## 🎥 Demo Video
A demo video showing the project overview, model approach, and working web interface is available here:

👉 [Demo Video Link](https://drive.google.com/file/d/1a7yupkj6ZiORg8bL0AXlOAOt_zfjL7AR/view?usp=sharing)


## ▶️ How to Run

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/gunimishra273/AutoJudge.git
cd AutoJudge
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask app

```bash
python app.py
```

### 4️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 🧪 Training the Models (Optional)

To retrain models from scratch:

```bash
python src/train_classifier.py
python src/train_classifier_svm.py
python src/train_regressor.py
python src/train_regressor_rf.py
```

Trained models are saved automatically in the `models/` directory.

---

## 🔮 Future Improvements

* Use transformer-based embeddings (BERT)
* Add dataset balancing techniques
* Improve regression accuracy with ensemble tuning
* Support multi-language problem descriptions

---

## 🧾 Conclusion

AutoJudge demonstrates that **problem difficulty can be reasonably predicted using only textual information**.
The project showcases a complete ML pipeline — from preprocessing and modeling to deployment and UI — making it suitable for academic evaluation and live demos.

---

## 👩‍💻 Author

**Guni Mishra**

---
