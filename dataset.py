from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class QADataset(Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, max_seq_length: int, mode: str, ensemble: bool, pg_stride: int=435, max_pg_len: int=450):
        self._data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.ensemble =ensemble
        self.max_pg_len = max_pg_len
        self.pg_stride = pg_stride
        
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
    
    # def trim_to_max_length(self, max_seq_length: int, passage):
    #     if len(passage) > 2000:
    #         passage = passage[:2000]
    #     return passage

    def preprocess(self, tokenizer, max_seq_length: int, x) -> Dict:
        label_map = {"A": 0, "B": 1, "C": 2}
        question = x["question"]["stem"]
        passage = x["text"]

        if self.ensemble:
            all_splits_choices_features = {"input_ids":[], "attention_mask":[]}
            # max len of ques+opt = 51 
            # max len of pg = 7654
            option: str
            for option in x["question"]["choices"]:
                splits_choices_features = []    
                question_option = question + " " + option['text'] 
                # max_len = 3060
                # len of pg per split = 450
                # stride = 435 (i.e., 15 tokens overlapped)
                for i in range(0, self.pg_stride * 6, self.pg_stride):
                    passage_split = x["text"][i : i + self.max_pg_len]
                    inputs = tokenizer(
                        passage_split,
                        question_option,
                        add_special_tokens=True,
                        max_length=max_seq_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    splits_choices_features.append(inputs)
                
                all_splits_choices_features["input_ids"].append(  \
                    torch.vstack([cf["input_ids"][0] for cf in splits_choices_features]).permute(1,0)
                )
                all_splits_choices_features["attention_mask"].append(  \
                    torch.vstack([cf["attention_mask"][0] for cf in splits_choices_features]).permute(1,0)
                )

            if self.mode == "test":
                label = None
            else:
                labels = label_map.get(x["answer"].strip(), -1)
                label = torch.tensor(labels).long()

            # shape of input_ids and attention_mask in return = [num_options, max_seq_len, num_splits]
            return {
                "label": label,
                "input_ids": torch.stack([ids for ids in all_splits_choices_features["input_ids"]], dim=0),
                "attention_mask": torch.stack([mask for mask in all_splits_choices_features["attention_mask"]], dim=0),
    #             "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1),  # BERT 的話還是需要這個吧（？
            }
                
        else:
            choices_features = []
            option: str
            for option in x["question"]["choices"]:    
                question_option = question + " " + option['text']
                inputs = tokenizer(
                    passage,
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
    #             "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1),  # 同上
            }
