from flask import Flask, render_template, request

from predictor import predict_url
from feature_extractor import extract_features

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    url = request.form["url"]

    try:

        result = predict_url(url)

        features = extract_features(url)

        prediction = result["prediction"]

        confidence = result["confidence"]

        if prediction == "Legitimate":

            result_class = "safe"

            risk_score = round(100 - confidence, 2)

        else:

            result_class = "danger"

            risk_score = confidence

        return render_template(

            "index.html",

            url=url,

            prediction=prediction,

            confidence=confidence,

            risk_score=risk_score,

            result_class=result_class,

            features=features

        )

    except Exception as e:

        return render_template(

            "index.html",

            error=str(e)

        )


if __name__ == "__main__":

    app.run(debug=True)