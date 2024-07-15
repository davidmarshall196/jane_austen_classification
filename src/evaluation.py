import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src import logger_config

logger = logger_config.get_logger()


def evaluate_model(
    model: DistilBertForSequenceClassification, test_loader: DataLoader
) -> None:
    """
    Evaluate the model on the test set.

    Args:
        model (DistilBertForSequenceClassification): The BERT model.
        test_loader (DataLoader): DataLoader for test data.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_preds, test_labels = [], []
    logger.info(f"Evaluation rows: {len(test_loader.dataset)}")
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    logger.info("Test Set Evaluation")
    logger.info("\n" + classification_report(test_labels, test_preds))
