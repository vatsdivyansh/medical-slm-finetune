from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

load_dotenv()
login(token=os.environ.get("HF_TOKEN"))
print("Logged in to HuggingFace ✅")

FINETUNED_MODEL = "checkpoints/final_model"
HF_USERNAME     = "divyanshvats2004"
MODEL_NAME      = "medical-gemma-3-270m"
REPO_ID         = f"{HF_USERNAME}/{MODEL_NAME}"

print(f"\nLoading fine-tuned model from {FINETUNED_MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL,
    dtype      = torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
print("Model loaded ✅")


print(f"\nUploading to HuggingFace as {REPO_ID}...")
print("This may take a few minutes...")

model.push_to_hub(
    REPO_ID,
    commit_message = "Upload medical fine-tuned Gemma 3 270M",
    private        = False,
)

tokenizer.push_to_hub(
    REPO_ID,
    commit_message = "Upload tokenizer",
)

print(f"\nModel uploaded successfully ✅")
print(f"View it at: https://huggingface.co/{REPO_ID}")