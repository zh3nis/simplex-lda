import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

from lda import SimplexLDAHead


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformerLDA(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, num_labels - 1)
        self.head = SimplexLDAHead(C=num_labels, D=num_labels - 1)

    def forward(self, **inputs):
        out = self.encoder(**inputs)
        cls = out.last_hidden_state[:, 0]
        z = self.proj(cls)
        return self.head(z)


@torch.no_grad()
def evaluate_softmax(model, loader, device):
    model.eval()
    ok = tot = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(1)
        ok += (preds == batch["labels"]).sum().item()
        tot += batch["labels"].size(0)
    return ok / tot


@torch.no_grad()
def evaluate_simplex(model, loader, device):
    model.eval()
    ok = tot = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs)
        preds = logits.argmax(1)
        ok += (preds == labels).sum().item()
        tot += labels.size(0)
    return ok / tot


def load_ag_news(tokenizer):
    raw_ds = load_dataset("ag_news")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized, 4


def load_clinc150(tokenizer):
    raw_ds = load_dataset("clinc_oos", "plus")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    label_col = "label" if "label" in raw_ds["train"].column_names else "intent"
    remove_cols = [c for c in raw_ds["train"].column_names if c not in [label_col]]
    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=remove_cols)
    if label_col != "labels":
        tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format("torch")

    label_feature = raw_ds["train"].features[label_col]
    if hasattr(label_feature, "num_classes"):
        num_labels = label_feature.num_classes
    else:
        num_labels = len(set(raw_ds["train"][label_col]))
    return tokenized, num_labels


def make_loaders(tokenized, tokenizer, train_batch_size, test_batch_size):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ld = DataLoader(
        tokenized["train"], batch_size=train_batch_size, shuffle=True, collate_fn=data_collator
    )
    test_ld = DataLoader(
        tokenized["test"], batch_size=test_batch_size, shuffle=False, collate_fn=data_collator
    )
    return train_ld, test_ld


def train_run(
    head,
    model_name,
    num_labels,
    tokenized,
    tokenizer,
    epochs,
    train_batch_size,
    test_batch_size,
    lr,
    device,
):
    train_ld, test_ld = make_loaders(tokenized, tokenizer, train_batch_size, test_batch_size)

    if head == "Softmax":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        eval_fn = evaluate_softmax
        loss_fn = None
    else:
        model = TransformerLDA(model_name, num_labels).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        eval_fn = evaluate_simplex
        loss_fn = nn.NLLLoss()

    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
        for batch in train_ld:
            if head == "Softmax":
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                logits = out.logits
                labels = batch["labels"]
            else:
                labels = batch["labels"].to(device)
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs)
                loss = loss_fn(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                acc_sum += (preds == labels).sum().item()
                n_sum += labels.size(0)
                loss_sum += loss.item() * labels.size(0)

        tr_acc = acc_sum / n_sum
        te_acc = eval_fn(model, test_ld, device)
        epoch_metrics.append({"epoch": epoch, "train_acc": tr_acc, "test_acc": te_acc})
        print(
            f"[{head}][{epoch:02d}] train loss={loss_sum/n_sum:.4f} acc={tr_acc:.4f} | test acc={te_acc:.4f}"
        )

    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Transformer on AG News and CLINC150 with SimplexLDA and Softmax heads."
    )
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/transformer_agnews_clinc_two_heads.json",
        help="Path to save results JSON.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    datasets = {
        "AGNews": load_ag_news,
        "CLINC150": load_clinc150,
    }
    heads = ["SimplexLDA", "Softmax"]

    results = {"AGNews": {}, "CLINC150": {}}

    for dataset_name, loader_fn in datasets.items():
        tokenized, num_labels = loader_fn(tokenizer)
        for head in heads:
            head_runs = []
            for run_idx in range(1, args.runs + 1):
                seed = args.seed + run_idx
                set_seed(seed)
                print(f"\n== {dataset_name} | {head} | run {run_idx}/{args.runs} (seed={seed}) ==")
                epoch_metrics = train_run(
                    head,
                    args.model_name,
                    num_labels,
                    tokenized,
                    tokenizer,
                    args.epochs,
                    args.train_batch_size,
                    args.test_batch_size,
                    args.lr,
                    device,
                )
                head_runs.append(
                    {
                        "run": run_idx,
                        "seed": seed,
                        "epoch_metrics": epoch_metrics,
                        "final_train_acc": epoch_metrics[-1]["train_acc"],
                        "final_test_acc": epoch_metrics[-1]["test_acc"],
                    }
                )
            results[dataset_name][head] = head_runs

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "runs": args.runs,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
