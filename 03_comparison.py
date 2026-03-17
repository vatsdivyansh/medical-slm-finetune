import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

BASE_MODEL      = "google/gemma-3-270m-it"
FINETUNED_MODEL = "checkpoints/final_model"

TEST_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "What is the treatment for high blood pressure?",
    "What causes chest pain?",
    "How is pneumonia diagnosed?",
    "What are the side effects of ibuprofen?",
]

def load_model_pipeline(model_path, label):
    print(f"\nLoading {label}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype         = torch.bfloat16,
        device_map          = "auto",
        attn_implementation = "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe      = pipeline(
        "text-generation",
        model     = model,
        tokenizer = tokenizer
    )
    print(f" {label} loaded")
    return pipe

def get_answer(pipe, question, max_new_tokens=200):
    gen_config = GenerationConfig(
        max_new_tokens = max_new_tokens,
        do_sample      = False,
    )
    raw       = pipe(
        text_inputs       = [{"role": "user", "content": question}],
        generation_config = gen_config,
    )
    generated = raw[0]["generated_text"]
    answer    = generated[-1]["content"] if isinstance(generated, list) else str(generated)
    return answer


base_pipe      = load_model_pipeline(BASE_MODEL,      "Base Model")
finetuned_pipe = load_model_pipeline(FINETUNED_MODEL, "Fine-Tuned Model")


print("\n" + "=" * 70)
print("  BEFORE vs AFTER COMPARISON")
print("=" * 70)

for i, question in enumerate(TEST_QUESTIONS):
    print(f"\n{'─' * 70}")
    print(f"  Q{i+1}: {question}")
    print(f"{'─' * 70}")

    
    base_answer      = get_answer(base_pipe, question)
    print(f"\n  BASE MODEL:")
    print(f"  {base_answer[:300]}...")

    
    finetuned_answer = get_answer(finetuned_pipe, question)
    print(f"\n FINE-TUNED MODEL:")
    print(f"  {finetuned_answer[:300]}...")

print("\n" + "=" * 70)
print("  COMPARISON COMPLETE ✅")
print("=" * 70)
