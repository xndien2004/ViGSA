import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel
from .absa_dataset import ABSADataset

def prepare_auxiliary_embeddings(texts, tokenizer, model, device='cuda', max_length=256):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()
    model.to(device)

    # Tokenize
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state[:, 0, :]  if hasattr(outputs, 'last_hidden_state') else outputs[0]

    return embeddings.cpu()


class ABSADatamodule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, auxiliary_aspects, model, tokenizer, device="cuda", batch_size=32, max_length=256):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.auxiliary_embeddings =  prepare_auxiliary_embeddings(
            auxiliary_aspects, tokenizer, model, device=device
        )
        self.max_length = max_length
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_dataset = ABSADataset(self.train_df, self.auxiliary_embeddings, self.tokenizer, self.max_length)
        self.val_dataset = ABSADataset(self.val_df, self.auxiliary_embeddings, self.tokenizer, self.max_length)
        self.test_dataset = ABSADataset(self.test_df, self.auxiliary_embeddings, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=4, pin_memory=True, prefetch_factor=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=4, pin_memory=True, prefetch_factor=2
        )