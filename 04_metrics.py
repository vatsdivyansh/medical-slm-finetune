import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from rouge_score import rouge_scorer
import statistics

BASE_MODEL      = "google/gemma-3-270m-it"
FINETUNED_MODEL = "checkpoints/final_model"

test_data = [
    {
        "question"  : "What are the symptoms of diabetes?",
        "reference" : "Symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow healing wounds and frequent infections."
    },
    {
        "question"  : "What is the treatment for high blood pressure?",
        "reference" : "Treatment for high blood pressure includes lifestyle changes such as low sodium diet, regular exercise, weight loss, limiting alcohol, and medications like ACE inhibitors or beta blockers."
    },
    {
        "question"  : "What causes chest pain?",
        "reference" : "Chest pain can be caused by heart attack, angina, acid reflux, muscle strain, pneumonia, or anxiety. It is important to seek medical attention immediately for chest pain."
    },
    {
        "question"  : "What are the side effects of ibuprofen?",
        "reference" : "Side effects of ibuprofen include stomach pain, nausea, heartburn, dizziness, headache, and in serious cases kidney problems or increased risk of heart attack."
    },
]

def load_model(model_path, label):
    print(f"Loading {label}...")
    model     = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype         = torch.bfloat16,
        device_map          = "auto",
        attn_implementation = "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe      = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"{label} loaded\n")
    return pipe

def get_answer(pipe, question):
    gen_config = GenerationConfig(
        max_new_tokens = 200,
        do_sample      = False,
    )
    start     = time.time()
    result    = pipe(
        text_inputs       = [{"role": "user", "content": question}],
        generation_config = gen_config,
    )
    elapsed   = round(time.time() - start, 2)
    generated = result[0]["generated_text"]
    answer    = generated[-1]["content"]
    answer    = answer.replace("\n", " ").strip()
    return answer, elapsed

def calculate_rouge(prediction, reference):
    scorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer = True
    )
    scores  = scorer.score(reference, prediction)
    return {
        "rouge1" : round(scores["rouge1"].fmeasure, 4),
        "rouge2" : round(scores["rouge2"].fmeasure, 4),
        "rougeL" : round(scores["rougeL"].fmeasure, 4),
    }

def word_count(text):
    return len(text.split())
base_pipe      = load_model(BASE_MODEL,      "Base Model")
finetuned_pipe = load_model(FINETUNED_MODEL, "Fine-Tuned Model")
base_rouge1      = []
base_rouge2      = []
base_rougeL      = []
base_times       = []
base_lengths     = []
ft_rouge1        = []
ft_rouge2        = []
ft_rougeL        = []
ft_times         = []
ft_lengths       = []
print("=" * 60)
print("  RUNNING EVALUATION ON ALL QUESTIONS")
print("=" * 60)
for i, item in enumerate(test_data):
    question  = item["question"]
    reference = item["reference"]
    print(f"\nQuestion {i+1}: {question}")
    base_answer, base_time = get_answer(base_pipe, question)
    base_scores            = calculate_rouge(base_answer, reference)
    base_rouge1.append(base_scores["rouge1"])
    base_rouge2.append(base_scores["rouge2"])
    base_rougeL.append(base_scores["rougeL"])
    base_times.append(base_time)
    base_lengths.append(word_count(base_answer))
    ft_answer, ft_time = get_answer(finetuned_pipe, question)
    ft_scores          = calculate_rouge(ft_answer, reference)
    ft_rouge1.append(ft_scores["rouge1"])
    ft_rouge2.append(ft_scores["rouge2"])
    ft_rougeL.append(ft_scores["rougeL"])
    ft_times.append(ft_time)
    ft_lengths.append(word_count(ft_answer))
    print(f"  Base     ROUGE-1: {base_scores['rouge1']}  ROUGE-2: {base_scores['rouge2']}  ROUGE-L: {base_scores['rougeL']}  Words: {word_count(base_answer)}  Time: {base_time}s")
    print(f"  FineTune ROUGE-1: {ft_scores['rouge1']}  ROUGE-2: {ft_scores['rouge2']}  ROUGE-L: {ft_scores['rougeL']}  Words: {word_count(ft_answer)}  Time: {ft_time}s")

print("\n\n" + "=" * 60)
print("  FINAL METRICS SUMMARY")
print("=" * 60)
print(f"\n{'Metric':<25} {'Base Model':<20} {'Fine-Tuned Model':<20}")
print("-" * 65)
print(f"{'Avg ROUGE-1':<25} {statistics.mean(base_rouge1):<20.4f} {statistics.mean(ft_rouge1):<20.4f}")
print(f"{'Avg ROUGE-2':<25} {statistics.mean(base_rouge2):<20.4f} {statistics.mean(ft_rouge2):<20.4f}")
print(f"{'Avg ROUGE-L':<25} {statistics.mean(base_rougeL):<20.4f} {statistics.mean(ft_rougeL):<20.4f}")
print(f"{'Avg Response Time(s)':<25} {statistics.mean(base_times):<20.2f} {statistics.mean(ft_times):<20.2f}")
print(f"{'Avg Response Length':<25} {statistics.mean(base_lengths):<20.1f} {statistics.mean(ft_lengths):<20.1f}")


