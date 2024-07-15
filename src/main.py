from flask import Flask, request, jsonify
from src import prediction
from src import logger_config
from src import constants


logger = logger_config.get_logger()

app = Flask(__name__)


@app.route("/")
def home():
    return "Jane Austen Predictor API"


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.json
    paragraph = data.get("paragraph", "")

    try:
        model, tokenizer = prediction.load_model_and_tokenizer(
            constants.MODEL_FILE_PATH
        )
        pred = prediction.predict([paragraph], model, tokenizer)
        return jsonify({"prediction": pred[0]})
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
