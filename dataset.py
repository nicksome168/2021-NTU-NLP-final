from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class QADataset(Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, summarizer_, max_seq_length: int, mode: str):
        self._data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.summarizer_ = summarizer_
        self.special_tokenizer = False

    def __len__(self) -> int:
        return len(self._data)
    def __getitem__(self, idx: int) -> tuple:
        instance = self._data[idx]
        results = self.preprocess(self.tokenizer, self.max_seq_length, instance)
        return results

    def collate_fn(self, batch: List[tuple]) -> tuple:    
        input_ids_batch, token_type_ids_batch, attention_mask_batch, label_batch = [], [], [], []
        for results in batch:
            label, input_ids, token_type_ids, attention_mask  = results['label'], results['input_ids'], results['token_type_ids'], results['attention_mask']
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)
            label_batch.append(label)
        if self.mode == "test":
            return torch.stack(input_ids_batch), \
                    torch.stack(token_type_ids_batch), \
                    torch.stack(attention_mask_batch), \
                    None
        return torch.stack(input_ids_batch), \
                torch.stack(token_type_ids_batch), \
                torch.stack(attention_mask_batch), \
                torch.stack(label_batch)
    
    def trim_to_max_length(self, max_seq_length: int, passage):
        if len(passage) > max_seq_length:
            passage = passage[-max_seq_length:]
        return passage

    def preprocess(self, tokenizer, max_seq_length: int, x) -> Dict:
        choices_features = []
        label_map = {"A": 0, "B": 1, "C": 2}
        special_tokens = {"cls": "<cls>", "sep":"<sep>", "pad":"<pad>"}
        question = x["question"]["stem"]
        passage = x["text"]
        idx = x["id"]
        options = sorted(x["question"]["choices"], key=lambda k: k['label'])
        options = [opt["text"] for opt in options]

        if self.summarizer_ != None:
            # summarize passage
            if self.mode == 'train':
                passage = self.summarizer_.get_summary(passage, idx, options, max_seq_length - 3 - 51)
            else:
                passage = self.summarizer_.get_summary_local(passage, idx, options, max_seq_length - 3 - 51)
        
        if self.special_tokenizer:
            # self-made tokenizer
            pg_tokens = tokenizer.tokenize(passage)
            q_tokens = tokenizer.tokenize(question)
            for option in options:
                question_option = question + " " + option
                question_option = self.trim_to_max_length(51, question_option)
                pg_tokens = tokenizer.tokenize(passage)
                q_opt_tokens = tokenizer.tokenize(option)
                tokens = [special_tokens['cls']] + pg_tokens + [special_tokens['sep']] + q_opt_tokens + [special_tokens['sep']]
                segment_ids = [0] * (len(pg_tokens) + 2) + [1] * (len(q_opt_tokens) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                if max_seq_length - len(input_ids) > 0:
                    padding = [special_tokens['pad']] * (max_seq_length - len(input_ids))
                    padding_ids = tokenizer.convert_tokens_to_ids(padding)

                    input_ids += padding_ids
                    input_mask += padding_ids
                    segment_ids += padding_ids
                
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

            choices_features.append({
                "input_ids": torch.tensor(inputs.input_ids]),
                "attention_mask": torch.tensor(inputs.attention_mask]),
                "token_type_ids": torch.tensor(inputs.token_type_ids])
            })

        else:        
            for option in options:
                question_option = question + " " + option
                question_option = self.trim_to_max_length(51, question_option)
                inputs = tokenizer(
                    passage,
                    question_option,
                    add_special_tokens=True,
                    max_length=max_seq_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
            choices_features.append({
                "input_ids": torch.tensor(inputs.input_ids[0]),
                "attention_mask": torch.tensor(inputs.attention_mask[0]),
                "token_type_ids": torch.tensor(inputs.token_type_ids[0])
            })
        
        if self.mode == "test":
            label = None
        else:
            labels = label_map.get(x["answer"].strip(), -1)
            label = torch.tensor(labels).long()
            
        return {
            "label": label,
            "input_ids": torch.stack([cf["input_ids"] for cf in choices_features]).reshape(-1, max_seq_length),
            "attention_mask": torch.stack([cf["attention_mask"] for cf in choices_features]).reshape(-1, max_seq_length),
            "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1, max_seq_length),
        }