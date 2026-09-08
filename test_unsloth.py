#!/usr/bin/env python3
"""Does Unsloth's generic transformers-compatible loading actually work on
rumik-oss-1 (a custom, trust_remote_code=True Cohere2 subclass) with zero
Unsloth code changes, as their docs claim? Test it for real."""

import sys

from unsloth import FastModel

repo_dir = sys.argv[1] if len(sys.argv) > 1 else "./rumik-oss-1"

print(f"[1/2] loading {repo_dir} via FastModel.from_pretrained ...", flush=True)
model, tokenizer = FastModel.from_pretrained(
    model_name=repo_dir,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
    trust_remote_code=True,
    full_finetuning=False,  # exercise the LoRA/PEFT path Unsloth is built for
)
print(f"    loaded ok. model class: {type(model).__name__}", flush=True)

print("[2/2] wrapping with LoRA via FastModel.get_peft_model ...", flush=True)
model = FastModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"    LoRA applied ok. trainable params: {n_trainable:,} / {n_total:,}", flush=True)

print("\nSUCCESS: rumik-oss-1 loads and gets LoRA-wrapped through Unsloth with zero code changes.")
