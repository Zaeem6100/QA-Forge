import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

HF_TOKEN = os.getenv("HF_TOKEN")

# Model Configuration

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "Llama-3.1-8B-Instruct"

DATA_PATH = "data/processed_dataset.csv"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 1024
SEED = 42

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

DEVICE_MAP = "auto"
LOAD_IN_4BIT = True


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA training.")

    df = pd.read_csv(DATA_PATH)
    dataset = Dataset.from_pandas(df)

    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.0,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=SEED,
        push_to_hub=False,
        optim="paged_adamw_8bit",
    )

    # -------------------------
    # Trainer
    # -------------------------
    args_dist = training_args.to_dict()
    args_dist.pop("push_to_hub_token", None)
    sft_args = SFTConfig(**args_dist)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_args,
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # -------------------------
    # Save
    # -------------------------
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete. Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
