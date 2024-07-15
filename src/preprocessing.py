import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from typing import List, Tuple
from src import logger_config
from src import constants


logger = logger_config.get_logger()


def validate_dataframe(data: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame contains the expected columns and is not empty.

    Args:
        data (pd.DataFrame): The DataFrame to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    expected_columns = constants.DATA_COLUMNS
    if not all(column in data.columns for column in expected_columns):
        logger.error("Data validation error: Missing expected columns.")
        return False
    if data.empty:
        logger.error("Data validation error: DataFrame is empty.")
        return False
    logger.info("Data validation successful.")
    return True


def load_data(
    file_path: str,
    sample_frac: float = None,
) -> pd.DataFrame:
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        sample_frac (float): Optional fraction to sample.

    Returns:
        texts: A list of the texts.
        labels: A Tensor object containing the labels.

    Raises:
        IOError: If there is an error reading the data file.
        ValueError: If the data validation fails.
    """
    try:
        data = pd.read_json(file_path)
        if not validate_dataframe(data):
            raise ValueError("Data validation failed.")
        if sample_frac:
            data = data.sample(frac=sample_frac)
        texts = list(data["text"])
        labels = torch.tensor(list(data["austen"]))
        return texts, labels
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error reading the data file: {e}")
        raise IOError(f"Error reading the data file: {e}")


def tokenize_texts(
    texts: List[str], tokenizer: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize texts using a BERT tokenizer.

    Args:
        texts (List[str]): A list of texts to tokenize.
        tokenizer (str): A BERT tokenizer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        A tuple containing input IDs and attention masks as tensors.
    """
    tokenizer_obj = DistilBertTokenizer.from_pretrained(tokenizer)
    encodings = tokenizer_obj(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    return (encodings["input_ids"], encodings["attention_mask"], tokenizer_obj)


def save_tokenizer(tokenizer: DistilBertTokenizer, directory: str) -> None:
    """
    Save the model and tokenizer to a directory.

    Args:
        tokenizer (DistilBertTokenizer): The BERT tokenizer.
        directory (str): Directory to save the tokenizer.

    Raises:
        IOError: If there is an error saving the tokenizer.
    """
    try:
        tokenizer.save_pretrained(directory)
        logger.info(f"Tokenizer saved to {directory}")
    except Exception as e:
        logger.error(f"Error saving the tokenizer: {e}")
        raise IOError(f"Error saving the tokenizer: {e}")


def preprocess_data(
    input_texts: str,
    labels: torch.Tensor,
    tokenizer: str,
    train_size: float,
    val_size: float,
    directory: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare datasets and data loaders for training, validation, and testing.

    Args:
        input_texts: The input IDs tensor.
        labels (torch.Tensor): The labels tensor.
        tokenizer (torch.Tensor): The attention masks tensor.
        train_size (float): The fraction of data used to train.
        val_size (float): The fraction of data used to validate.
        directory (str): The directory to save the tokenizer.

    Returns:
        A tuple containing data loaders for training, validation, and testing.
    """
    input_ids, attention_masks, tokenizer_obj = tokenize_texts(input_texts, tokenizer)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    save_tokenizer(tokenizer_obj, directory)
    logger.info("Datasets and data loaders prepared successfully.")
    return tokenizer, train_loader, val_loader, test_loader
