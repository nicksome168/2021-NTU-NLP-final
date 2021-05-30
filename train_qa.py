import os
import argparse
import json
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import handle_reproducibility, loss_fn, clean_data
from dataset import QADataset
from model import BertMultipleChoiceModel, XLNetMultipleChoiceModel


def train(args: argparse.Namespace) -> None:
    with open(args.data_dir / args.train_data) as file:
        all_data = json.load(file)
    all_data = clean_data(all_data)
    
    # Split Data
    valid_data = all_data[:round(len(all_data) * args.train_val_split)]
    train_data = all_data[round(len(all_data) * args.train_val_split):]
    print(f"train data: {len(train_data)} valid data: {len(valid_data)}")

    # Load Dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_set = QADataset(train_data, tokenizer, args.max_seq_length, mode="train", ensemble=args.ensemble)
    valid_set = QADataset(valid_data, tokenizer, args.max_seq_length, mode="valid", ensemble=args.ensemble)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
    )

    # Initiate Model
    if 'xlnet' in args.base_model:
        model = XLNetMultipleChoiceModel(args.base_model, mem_len=args.max_seq_length)
    else:
        model = BertMultipleChoiceModel(args.base_model)
    model.to(args.device)
    
    # Optimizer and Loss func
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fct = CrossEntropyLoss()

    # Logging
    if args.wandb_logging:
        import wandb

        wandb.init(project="2021-NLP-final", entity="nicksome_yc", name=args.exp_name, config=args)
        wandb.watch(model)

    best_metric = 0
    for epoch in range(1, args.num_epoch + 1):
        print(f"----- Epoch {epoch} -----")

        model.train()
        optimizer.zero_grad()
        train_loss = 0
        train_corrects = 0
        for batch_idx, (input_ids_batch, attention_mask_batch, label_batch) in enumerate(tqdm(train_loader)):
            input_ids = input_ids_batch.to(args.device)
            attention_mask = attention_mask_batch.to(args.device)
            labels = label_batch.to(args.device)
            
            # DEBUG
            # print(tokenizer.decode(input_ids[0,0,:,0]))
            # print(input_ids.shape)

            if args.ensemble:
                # input shape = (batch_size, num_options, seq_len, num_split)
                # label shape = (batch_size,)
                mems, loss = None, None
                logits = torch.zeros((args.batch_size, input_ids_batch.shape[1])).to(args.device)  # [batch_size, num_options]
                splits_num = input_ids.shape[3]
                for i in range(splits_num):
                    outputs = model(input_ids=input_ids[:,:,:,i], attention_mask=attention_mask[:,:,:,i], labels=labels, mems=mems)
                    mems = outputs.mems  # XLNet hidden_state
                    logits += outputs.logits
                logits /= splits_num
                loss = loss_fct(logits, labels.view(-1))
            else:
                # input shape = (batch_size, num_options, seq_len)
                # label shape = (batch_size,)
                # print(tokenizer.decode(input_ids[0,0,:]))
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            
            loss.backward()
            if (batch_idx + 1) % args.n_batch_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            train_corrects += loss_fn(logits, labels)
            
        train_log = {
                "train_loss": train_loss / len(train_set),
                "train_acc": train_corrects / len(train_set),
        } 
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")

        # Validation
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_corrects = 0
            for batch_idx, (input_ids_batch, attention_mask_batch, label_batch) in enumerate(tqdm(valid_loader)):
                input_ids = input_ids_batch.to(args.device)
                attention_mask = attention_mask_batch.to(args.device)
                labels = label_batch.to(args.device)
                
                if args.ensemble:
                    mems, loss = None, None
                    logits = torch.zeros((args.batch_size, input_ids_batch.shape[1])).to(args.device)  # [batch_size, num_options]
                    splits_num = input_ids.shape[3]
                    for i in range(splits_num):
                        outputs = model(input_ids=input_ids[:,:,:,i], attention_mask=attention_mask[:,:,:,i], labels=labels, mems=mems)
                        mems = outputs.mems
                        logits += outputs.logits
                    logits /= splits_num
                    loss = loss_fct(logits, labels.view(-1))
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                valid_loss += loss.item()
                logits = outputs.logits
                valid_corrects += loss_fn(logits, labels)

            valid_log = {
                "valid_loss": valid_loss / len(valid_set),
                "valid_acc": valid_corrects / len(valid_set),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")
            if args.wandb_logging:
                wandb.log({**train_log, **valid_log})

        if valid_log[args.metric_for_best] > best_metric:
            best_metric = valid_log[args.metric_for_best]
            best = True
            if args.wandb_logging:
                wandb.run.summary[f"best_{args.metric_for_best}"] = best_metric
        else:
            best = False

        if best:
            torch.save(model.state_dict(), args.model_dir / f"best_model_{args.exp_name}.pt")
            print(f"{'':30s}*** Best model saved ***")

    if args.wandb_logging:
        wandb.save(str(args.model_dir / f"best_model_{args.exp_name}.pt"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    #data
    parser.add_argument("--train_data", type=str, default="train.json")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="data/",
    )

    # model
    parser.add_argument("--base_model", type=str, default="bert-base-chinese") #allenai/longformer-base-4096
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Directory to save model files.",
        default="model/",
    )
    
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-2)

    # data loader
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_val_split", type=int, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--n_batch_per_step", type=int, default=2)
    parser.add_argument("--metric_for_best", type=str, default="valid_loss")

    # logging
    parser.add_argument("--wandb_logging", action="store_true")
    parser.add_argument("--exp_name", type=str, default="bert-base-chinese-512")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    train(args)
