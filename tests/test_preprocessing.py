import pytest
import torch
from transformers import DistilBertTokenizer
from typing import List, Tuple
import pandas as pd
try:
    import preprocessing
except ImportError:
    import sys
    sys.path.append('../')
    from src import preprocessing


def test_validate_dataframe():
    valid_data = pd.DataFrame({'text': ['Sample text'], 'austen': [1]})
    invalid_data = pd.DataFrame({'wrong_column': ['Sample text']})

    assert preprocessing.validate_dataframe(valid_data) == True
    assert preprocessing.validate_dataframe(invalid_data) == False

def test_tokenize_texts():
    texts = ["Hello, world!", "How are you?"]
    tokenizer_name = 'distilbert-base-uncased'
    
    input_ids, attention_mask, tokenizer_obj = preprocessing.tokenize_texts(
        texts, tokenizer_name)
    
    assert isinstance(input_ids, torch.Tensor), "Input IDs should be a torch.Tensor"
    assert isinstance(attention_mask, torch.Tensor), "Attention mask should be a torch.Tensor"
    
    # Check if input_ids and attention_mask have the correct shape
    assert input_ids.shape[0] == len(texts), "The batch size should match the number of texts"
    assert attention_mask.shape[0] == len(texts), "The batch size should match the number of texts"
    
    # Ensure tokenizer is of correct type
    assert isinstance(tokenizer_obj, DistilBertTokenizer), "Tokenizer object should be a DistilBertTokenizer"

