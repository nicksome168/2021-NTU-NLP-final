from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class QADataset(Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, max_seq_length: int, mode: str):
        self._data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._data[idx]
        results = self.preprocess(self.tokenizer, self.max_seq_length, instance)
        return results

    def collate_fn(self, batch: List[tuple]) -> tuple:    
        input_ids_batch, attention_mask_batch, label_batch = [], [], []
        for results in batch:
            label, input_ids, attention_mask = results['label'], results['input_ids'], results['attention_mask']
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            label_batch.append(label)
        if self.mode == "test":
            return torch.stack(input_ids_batch), \
                    torch.stack(attention_mask_batch), \
                    None
        return torch.stack(input_ids_batch), \
                torch.stack(attention_mask_batch), \
                torch.stack(label_batch)
        
    def preprocess(self, tokenizer, max_seq_length: int, x) -> Dict:
        choices_features = []
        label_map = {"A": 0, "B": 1, "C": 2}
        question = x["question"]["stem"]
        article = x["text"]

        option: str
        for option in x["question"]["choices"]:    
            question_option = question + " " + option['text']

            inputs = tokenizer(
                article,
                question_option,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            choices_features.append(inputs)
        
        if self.mode == "test":
            label = None
        else:
            labels = label_map.get(x["answer"].strip(), -1)
            label = torch.tensor(labels).long()
            
        return {
            "label": label,
            "input_ids": torch.stack([cf["input_ids"] for cf in choices_features]).reshape(-1, max_seq_length),
            "attention_mask": torch.stack([cf["attention_mask"] for cf in choices_features]).reshape(-1, max_seq_length),
#             "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1),
        }