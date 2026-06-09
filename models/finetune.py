"""
finetune.py
===========
Fine-tunes CodeT5+ (220M) on the NL → HTML/CSS dataset
using LoRA (Low-Rank Adaptation) for memory-efficient training.

Usage:
  python models/finetune.py                          # default settings
  python models/finetune.py --epochs 5               # train longer
  python models/finetune.py --dataset data/dataset.json
  python models/finetune.py --output models/codet5-html

Requirements:
  pip install transformers datasets peft accelerate torch tqdm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from datasets import Dataset as HFDataset

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME    = "Salesforce/codet5p-220m"
MAX_INPUT_LEN = 128    # max tokens for NL description
MAX_TARGET_LEN = 1024  # max tokens for HTML output


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[list, list]:
    """
    Load and split the dataset into train/validation sets (90/10 split).

    Args:
        path: Path to dataset JSON

    Returns:
        (train_data, val_data) tuple
    """
    with open(path) as f:
        data = json.load(f)

    # Filter out invalid entries
    data = [d for d in data if d.get("html") and d.get("prompt")]
    print(f"   Loaded {len(data)} valid pairs")

    # 90/10 split
    split = int(len(data) * 0.9)
    train = data[:split]
    val   = data[split:]
    print(f"   Train: {len(train)} | Val: {len(val)}")
    return train, val


def tokenize_dataset(data: list,
                     tokenizer,
                     max_input: int,
                     max_target: int) -> HFDataset:
    """
    Tokenize a list of prompt/html pairs into a HuggingFace Dataset.

    Args:
        data:       List of dicts with 'prompt' and 'html'
        tokenizer:  Loaded tokenizer
        max_input:  Max input token length
        max_target: Max target token length

    Returns:
        Tokenized HuggingFace Dataset
    """
    prompts = [f"Generate HTML for: {d['prompt']}" for d in data]
    targets = [d["html"] for d in data]

    model_inputs = tokenizer(
        prompts,
        max_length=max_input,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target,
            padding="max_length",
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]

    # Replace padding token id with -100 so loss ignores padding
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs["labels"]
    ]

    return HFDataset.from_dict(model_inputs)


# ── LoRA config ────────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    """
    LoRA configuration for CodeT5+.
    Targets the attention projection layers for efficient fine-tuning.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,               # LoRA rank — higher = more params, better quality
        lora_alpha=32,      # scaling factor
        lora_dropout=0.1,
        bias="none",
        target_modules=["q", "v"],  # attention query and value projections
    )


# ── Training args ──────────────────────────────────────────────────────────────

def get_training_args(output_dir: str,
                      epochs: int,
                      batch_size: int,
                      lr: float) -> Seq2SeqTrainingArguments:
    """
    Build training arguments.

    Args:
        output_dir:  Where to save checkpoints
        epochs:      Number of training epochs
        batch_size:  Per-device batch size
        lr:          Learning rate

    Returns:
        Seq2SeqTrainingArguments instance
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Evaluation & saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Seq2Seq specific
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,

        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="none",       # set to "wandb" if you want experiment tracking

        # Mixed precision — speeds up training on NVIDIA GPUs
        fp16=torch.cuda.is_available(),

        # Save disk space
        save_total_limit=2,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    dataset_path: str = "data/dataset.json",
    output_dir:   str = "models/codet5-html",
    epochs:       int = 3,
    batch_size:   int = 4,
    lr:           float = 3e-4,
):
    print(f"\n{'═' * 55}")
    print(f"  CodeT5+ Fine-tuning — NL → HTML/CSS")
    print(f"{'═' * 55}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Dataset    : {dataset_path}")
    print(f"  Output     : {output_dir}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {lr}")
    print(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'═' * 55}\n")

    # ── Load tokenizer & model ─────────────────────────────────────────────────
    print("📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    print("🔧 Applying LoRA adapters...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load & tokenize dataset ────────────────────────────────────────────────
    print(f"\n📂 Loading dataset from {dataset_path}...")
    train_data, val_data = load_dataset(dataset_path)

    print("🔠 Tokenizing...")
    train_dataset = tokenize_dataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_dataset   = tokenize_dataset(val_data,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    # ── Data collator ──────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training...\n")
    training_args = get_training_args(output_dir, epochs, batch_size, lr)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    print(f"\n💾 Saving model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'═' * 55}")
    print(f"  ✅ Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"{'═' * 55}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5+ on NL → HTML dataset")
    parser.add_argument("--dataset",    default="data/dataset.json",
                        help="Path to dataset JSON")
    parser.add_argument("--output",     default="models/codet5-html",
                        help="Output directory for saved model")
    parser.add_argument("--epochs",     type=int,   default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate")
    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )