import argparse
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import wandb
from .prompt_builder import build_prompt, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # W&B init
    wandb.init(project="lm-world-model", name=cfg["run_name"], config=cfg)

    # Load dataset
    ds = load_dataset("json",
                      data_files=cfg["data_file"],
                      split="train",
                      streaming=True)

    def format_example(example):
        prompt = build_prompt(example, condition=cfg["condition"])
        return {"text": f"<s>[INST] {SYSTEM_PROMPT} [/INST]\n{prompt}</s>"}

    ds = ds.map(format_example)

    # Load model (4-bit quantised)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    import torch
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        device_map="auto",
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 2e-4),
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_len", 1024),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main()
