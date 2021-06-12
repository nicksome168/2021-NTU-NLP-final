import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, XLNetForMultipleChoice
from tqdm import tqdm

from utils import handle_reproducibility, clean_data
from dataset import QADataset

from summarizer import Summarizer

@torch.no_grad()
def test(args):
    with open(args.data_path, encoding='utf-8') as file:
        all_data = json.load(file)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    summarizer_ = Summarizer()
    test_set = QADataset(all_data, tokenizer, summarizer_, args.max_seq_length, mode="test")

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=test_set.collate_fn,
        pin_memory=True,
    )

    model = XLNetForMultipleChoice.from_pretrained(args.base_model, mem_len=args.max_seq_length)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    model.eval()

    pred_dict = {"id":[],"answer":[]}
    label_map = {0: "A", 1: "B", 2: "C"}
    for batch_idx, (input_ids_batch, token_type_ids_batch, attention_mask_batch, _) in enumerate(tqdm(test_loader)):
        input_ids = input_ids_batch.to(args.device)
        token_type_ids = token_type_ids_batch.to(args.device)
        attention_mask = attention_mask_batch.to(args.device)

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        logits = outputs[0]

        pred = torch.argmax(logits, dim=1)
        
        # add to prediction dict
        pred_dict["id"].append(all_data[batch_idx]["id"])
        pred_dict["answer"].append(label_map.get(pred.to('cpu').item(), -1))

    with open(args.pred_path / args.pred_file, "w") as file:
        file.write("id,answer\n")
        for _id, ans in zip(pred_dict["id"], pred_dict["answer"]):
            file.write(str(_id) + "," + str(ans) + "\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument(
        "--data_path",
        type=Path,
        default="data/public.json",
    )
    
    # prediction
    parser.add_argument("--pred_path", type=Path, default="prediction")
    parser.add_argument("--pred_file", type=Path, default="public.csv")
    
    # model
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    
    # testing
    parser.add_argument("--device", type=torch.device, default="cuda:0")

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    handle_reproducibility(True)

    args = parse_args()
    args.pred_path.mkdir(parents=True, exist_ok=True)
    test(args)
