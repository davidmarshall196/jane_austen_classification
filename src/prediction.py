import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List
import logger_config

logger = logger_config.get_logger()


def load_model_and_tokenizer(model_dir: str):
    """
    Load the trained model and tokenizer from a directory.

    Args:
        model_dir (str): Directory where the model and tokenizer are saved.

    Returns:
        model (DistilBertForSequenceClassification): The loaded BERT model.
        tokenizer (DistilBertTokenizer): The loaded BERT tokenizer.
    """
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise IOError(f"Error loading model or tokenizer: {e}")


def predict(
    texts: List[str],
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
) -> List[int]:
    """
    Predict the labels for the given texts.

    Args:
        texts (List[str]): A list of texts to predict.
        model (DistilBertForSequenceClassification): The loaded BERT model.
        tokenizer (DistilBertTokenizer): The loaded BERT tokenizer.

    Returns:
        List[int]: Predicted labels for the input texts.
    """
    try:
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        return preds.tolist()
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Error during prediction: {e}")


def load_and_predict(model_dir: str, texts: List[str]) -> List[int]:
    """
    Load the model and tokenizer, and predict the labels for the given texts.

    Args:
        model_dir (str): Directory where the model and tokenizer are saved.
        texts (List[str]): A list of texts to predict.

    Returns:
        List[int]: Predicted labels for the input texts.
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        predictions = predict(texts, model, tokenizer)
        return predictions
    except Exception as e:
        logger.error(f"Error in load_and_predict: {e}")
        raise RuntimeError(f"Error in load_and_predict: {e}")
