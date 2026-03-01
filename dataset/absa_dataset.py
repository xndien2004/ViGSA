import torch
from torch.utils.data import Dataset

class ABSADataset(Dataset):
    def __init__(self, df, auxiliary_embeddings, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.auxiliary_embeddings = auxiliary_embeddings
        self.aspect_columns = [col for col in df.columns if col != 'Review']
        self.sentiment_map = {
            0: [1, 0, 0, 0], # None
            1: [0, 1, 0, 0], # Positive
            2: [0, 0, 1, 0], # Negative
            3: [0, 0, 0, 1]  # Neutral
        }

    def get_global_label(self, label_list):
        """Build global label theo quy tắc conflict"""
        has_pos = any(label == 1 for label in label_list)
        has_neg = any(label == 2 for label in label_list)
        if has_pos and has_neg:
            return 3  # conflict
        elif has_pos:
            return 2  # positive
        elif has_neg:
            return 0  # negative
        else:
            return 1  # neutral

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['Review']
        if "phobert" not in self.tokenizer.name_or_path.lower() or "vi" not in self.tokenizer.name_or_path.lower():
            text = text.replace("_", " ")
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        aspect_raw_labels = [int(row[col]) for col in self.aspect_columns]
        labels = torch.tensor([self.sentiment_map[label] for label in aspect_raw_labels])
        aspects = torch.tensor([1 if label > 0 else 0 for label in aspect_raw_labels], dtype=torch.float)

        global_label = torch.tensor(self.get_global_label(aspect_raw_labels), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,                  # [num_aspect, 4]
            'aspects': aspects,                # [num_aspect]
            'global_label': global_label,      # scalar
            'auxiliary_embeddings': self.auxiliary_embeddings
        }
