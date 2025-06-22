#!/usr/bin/env python3
"""
train_llama.py 
by vincent
--------------
• TinyLlama(1.1 B)/Llama-2/3 series model finetuning with LoRA
• torchrun  +  NCCL  +  🤗Trainer(Accelerate) + (optional)Deepspeed
• "no data" -> --synthetic for random token data
───────────────────────────────────────────────────────────────
Example (1 node 8GPU):
torchrun --nproc_per_node 8 train_llama.py \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step \
    --dataset wikitext --dataset_config wikitext-2-raw-v1 \
    --output_dir ./out --num_train_epochs 1 --per_device_train_batch_size 4
"""

import os, math, json, argparse, random
import torch, transformers
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)
from peft import (LoraConfig, get_peft_model, TaskType,
                  prepare_model_for_kbit_training)

def parse_args():
    p = argparse.ArgumentParser()
    # model & dataset
    p.add_argument("--model_name", type=str,
                   default="TinyLlama/TinyLlama-1.1B-intermediate-step")
    p.add_argument("--dataset", type=str, default="wikitext",
                   help="'huggingface dataset name' or'synthetic'")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--synthetic", action="store_true",
                   help="using random token")
    # hyper parameters
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    # output & Deepspeed
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # 기타
    p.add_argument("--block_size", type=int, default=512)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
def make_deepspeed_config(tmp_path: str):
    """if needed, write default Deepspeed Zero-1 json for 1 GPU 16GB"""
    cfg = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 4,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1},
        "gradient_accumulation_steps": 1
    }
    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2)

# ──────────────────────────────────────────────────────────────
def build_dataset(args, tokenizer):
    if args.synthetic:
        # random token -> block_size length sequence
        rnd = torch.randint
        def gen(num_samples=10_000):
            vocab = tokenizer.vocab_size
            for _ in range(num_samples):
                yield {"text": rnd(vocab, (args.block_size,)).tolist()}
        ds = load_dataset("json", data_files={"train": gen()}, streaming=False)
        prepare = lambda ex: {"input_ids": torch.tensor(ex["text"],
                                                        dtype=torch.int64)}
        return ds.map(prepare, remove_columns=ds["train"].column_names,
                      batched=False), "input_ids"
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        def tokenize(ex):
            return tokenizer(ex["text"])
        ds = ds.map(tokenize, batched=True, remove_columns=["text"])
        return ds, "input_ids"

# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    transformers.logging.set_verbosity_info()

    # check distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              use_fast=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. model + 4/8-bit prepare(memory saving)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    # LoRA adapter
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. dataset
    dataset, col_name = build_dataset(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer,
                                                    mlm=False)

    # 4. Deepspeed temporary file
    ds_cfg_path = None
    if args.use_deepspeed:
        ds_cfg_path = os.path.join(args.output_dir, "ds_config.json")
        if local_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            make_deepspeed_config(ds_cfg_path)
        torch.distributed.barrier()

    # 5. TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=args.fp16,
        bf16=not args.fp16,
        logging_steps=10,
        save_total_limit=2,
        evaluation_strategy="no",
        deepspeed=ds_cfg_path,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 7. training
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        print("✔️  Fine-tuning finished and model saved.")

if __name__ == "__main__":
    main()
