import os
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, XLNetForMultipleChoice
from tqdm import tqdm

from utils import handle_reproducibility, loss_fn, clean_data
from dataset import QADataset

# from dotenv import load_dotenv
# load_dotenv()

from summarizer import Summarizer

def train(args: argparse.Namespace) -> None:
    with open(args.data_dir / args.train_data) as file:
        all_data = json.load(file)
    all_data = clean_data(all_data)
    
    # Split data
    if args.valid_data:
        with open(args.data_dir / args.valid_data) as file:
            valid_data = json.load(file)
        valid_data = clean_data(valid_data)
        train_data = all_data
    else:
        valid_data = all_data[:round(len(all_data) * args.train_val_split)]
        train_data = all_data[round(len(all_data) * args.train_val_split):]
    print(f"train data: {len(train_data)} valid data: {len(valid_data)}")

    if args.need_summarize:
        summarizer_ = Summarizer()
        summarizer_.load_model(args.data_dir / args.train_data)
    else:
        summarizer_ = None

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_set = QADataset(train_data, tokenizer, summarizer_, args.max_seq_length, mode="train")
    valid_set = QADataset(valid_data, tokenizer, summarizer_, args.max_seq_length, mode="valid")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=True,
        num_workers=16,
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
    )

    model = XLNetForMultipleChoice.from_pretrained(args.base_model, mem_len=args.max_seq_length)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

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
        for batch_idx, (input_ids_batch, token_type_ids_batch, attention_mask_batch, label_batch) in enumerate(tqdm(train_loader)):
            input_ids = input_ids_batch.to(args.device)
            token_type_ids = token_type_ids_batch.to(args.device)
            attention_mask = attention_mask_batch.to(args.device)
            labels = label_batch.to(args.device)

            # if batch_idx % 10 == 0:
            #     for i in range(input_ids.shape[1]):
            #         print(tokenizer.decode(input_ids[0,i,:]))

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs[0]
            logits = outputs[1]

            loss_value = loss.item()
            if args.n_batch_per_step > 1:
                loss = loss / args.n_batch_per_step

            loss.backward()

            if (batch_idx + 1) % args.n_batch_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss_value
            train_corrects += loss_fn(logits, labels)
            
        train_log = {
                "train_loss": train_loss / len(train_set),
                "train_acc": train_corrects / len(train_set),
        } 
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")

        # Validation
        
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_corrects = 0
            data = []
            for batch_idx, (input_ids_batch, token_type_ids_batch, attention_mask_batch, label_batch) in enumerate(tqdm(valid_loader)):
                input_ids = input_ids_batch.to(args.device)
                token_type_ids = token_type_ids_batch.to(args.device)
                attention_mask = attention_mask_batch.to(args.device)
                labels = label_batch.to(args.device)
                        
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs[0]
                logits = outputs[1]
                
                logits = softmax(logits)
                valid_loss += loss.item()
                valid_corrects += loss_fn(logits, labels)

                if args.wandb_logging:
                    wandb.log({f"valid/{batch_idx+1}-logits": wandb.Histogram(torch.flatten(logits).to("cpu"))})
                    wandb.log({f"valid/{batch_idx+1}-correct": wandb.Histogram(loss_fn(logits, labels).to("cpu").item())})
                    pred_opt = torch.argmax(logits, dim=1)

                    pg = tokenizer.decode(input_ids[0, labels.to("cpu").item(),:]).split('<sep>')[0].replace('<pad>', '')
                    question_ans = tokenizer.decode(input_ids[0, labels.to("cpu").item(),:]).split('<sep>')[1]
                    question = question_ans.split(" ")[1]
                    ans = question_ans.split(" ")[2]
                    pred = tokenizer.decode(input_ids[0, pred_opt.to("cpu").item(),:]).split('<sep>')[1].split(" ")[2]
                    data.append([f"{batch_idx+1}", pg, question, ans, pred])
                    
            valid_log = {
                "valid_loss": valid_loss / len(valid_set),
                "valid_acc": valid_corrects / len(valid_set),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")
            if args.wandb_logging:
                wandb.log({**train_log, **valid_log})
                wandb.log({f"Pred": wandb.Table(data=data, columns=["idx", "passage", "question", "ans", "pred"])})

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
    parser.add_argument("--valid_data", type=str, default=None)
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
    parser.add_argument("--need_summarize", action="store_true")
    
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_val_split", type=int, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--n_batch_per_step", type=int, default=2)
    parser.add_argument("--metric_for_best", type=str, default="valid_acc")

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
