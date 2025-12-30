from flask import Flask, render_template, request
from src.predict import predict_difficulty

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        title = request.form.get("title", "")
        description = request.form.get("description", "")
        input_description = request.form.get("input_description", "")
        output_description = request.form.get("output_description", "")

        difficulty_class, difficulty_score = predict_difficulty(
            title,
            description,
            input_description,
            output_description
        )

        result = {
            "class": difficulty_class,
            "score": difficulty_score
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
