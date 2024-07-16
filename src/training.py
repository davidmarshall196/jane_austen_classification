import torch
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src import logger_config

logger = logger_config.get_logger()


def save_model(model: DistilBertForSequenceClassification, directory: str) -> None:
    """
    Save the model and tokenizer to a directory.

    Args:
        model (DistilBertForSequenceClassification): The BERT model.
        directory (str): Directory to save the model.

    Raises:
        IOError: If there is an error saving the model.
    """
    try:
        model.save_pretrained(directory)
        logger.info(f"Model saved to {directory}")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise IOError(f"Error saving the model: {e}")


def train_and_validate_model(
    model_type: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    directory: str,
) -> None:
    """
    Train the model.

    Args:
        model_type (str): The chosen BERT model.
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for training data.
        num_epochs (int, optional): Number of epochs for training. Defaults to 3.
        directory (str): Location to save the model.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_type)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Beginning training and validation process.")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info(f"Training rows: {len(train_loader.dataset)}")
    logger.info(f"Validation rows: {len(validation_loader.dataset)}")
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed.")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in validation_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
    save_model(model, directory)
    return model
