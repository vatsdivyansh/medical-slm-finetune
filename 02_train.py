import torch
import time
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Config ────────────────────────────────────────────
MODEL_NAME       = "google/gemma-3-270m-it"
DATASET_PATH     = "data/medical_dataset"
CHECKPOINT_DIR   = "checkpoints"
MAX_LENGTH       = 256   # keep low for 4GB VRAM
BATCH_SIZE       = 2     # small batch for RTX 3050
EPOCHS           = 2
LEARNING_RATE    = 5e-5

print("=" * 60)
print("  MEDICAL SLM FINE-TUNING")
print("=" * 60)
print(f"  Model    : {MODEL_NAME}")
print(f"  Epochs   : {EPOCHS}")
print(f"  Batch    : {BATCH_SIZE}")
print(f"  Max Len  : {MAX_LENGTH}")
print("=" * 60)


if torch.cuda.is_available():
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU — training will be very slow!")


print("\nLoading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"Train : {len(dataset['train'])} samples")
print(f"Test  : {len(dataset['test'])} samples")


print(f"\nLoading model: {MODEL_NAME}")
print("This may take a few minutes first time...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype         = torch.bfloat16,  # ← changed
    device_map          = "auto",
    attn_implementation = "eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded — {total_params:,} parameters")
print(f"✅ dtype  : {model.dtype}")
print(f"✅ device : {model.device}")

sft_config = SFTConfig(
    output_dir                  = CHECKPOINT_DIR,
    max_length                  = MAX_LENGTH,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_checkpointing      = True,
    gradient_accumulation_steps = 4,
    optim                       = "adamw_torch_fused",
    logging_steps               = 10,
    save_strategy               = "epoch",
    eval_strategy               = "epoch",
    learning_rate               = LEARNING_RATE,
    fp16                        = False,   # ← changed
    bf16                        = True,    # ← added
    lr_scheduler_type           = "constant",
    push_to_hub                 = False,
    report_to                   = "none",
)

trainer = SFTTrainer(
    model            = model,
    args             = sft_config,
    train_dataset    = dataset["train"],
    eval_dataset     = dataset["test"],
    processing_class = tokenizer,
)

start_time = time.time()
trainer.train()
elapsed    = time.time() - start_time

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE")
print(f"  Time taken : {elapsed/60:.1f} minutes")
print("=" * 60)

model.save_pretrained("checkpoints/final_model")
tokenizer.save_pretrained("checkpoints/final_model")
print("\n✅ Model saved to checkpoints/final_model")