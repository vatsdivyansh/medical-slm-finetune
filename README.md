#  SLM Fine-Tuning — Gemma 3 270M

Fine-tuning a Small Language Model on medical Q&A data
to act as a domain-specific medical assistant.

---

## What This Project Does

Takes Google's Gemma 3 270M base model and fine-tunes it
on real doctor-patient conversations to make it respond
like a medical domain expert.

---

## Tech Stack

| Component  | Details                              |
|------------|--------------------------------------|
| Base Model | google/gemma-3-270m-it               |
| Dataset    | ChatDoctor-HealthCareMagic-100K      |
| Training   | HuggingFace Transformers + TRL SFT  |
| Hardware   | NVIDIA RTX 3050 (4.3GB VRAM)        |
| Training Time | 76 minutes                        |

---

## Project Structure
```
01_prepare_dataset.py  → Load and format dataset
02_train.py            → Fine-tune the model
03_comparison.py       → Before vs After comparison
04_metrics.py          → ROUGE, response time, length
05_upload_to_hf.py     → Upload model to HuggingFace
```

---

## Results

| Metric              | Base Model | Fine-Tuned |
|---------------------|------------|------------|
| Avg ROUGE-1         | 0.2577     | 0.1724     |
| Avg ROUGE-2         | 0.1139     | 0.0593     |
| Avg ROUGE-L         | 0.2082     | 0.1356     |
| Avg Response Time   | 6.10s      | 3.01s      |
| Avg Response Length | 104.8      | 63.8       |

Fine-tuned model is **2x faster** and gives concise
doctor-style responses compared to the base model.

---

## Setup
```bash
# Clone the repo
git clone https://github.com/vatsdivyansh/medical-slm-finetune.git
cd medical-slm-finetune

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl accelerate huggingface_hub rouge-score python-dotenv sentencepiece
```

---

## Model on HuggingFace

The fine-tuned model is publicly available:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("divyanshvats2004/medical-gemma-3-270m")
tokenizer = AutoTokenizer.from_pretrained("divyanshvats2004/medical-gemma-3-270m")
```

🔗 [HuggingFace Model](https://huggingface.co/divyanshvats2004/medical-gemma-3-270m)

---

## Disclaimer

This project is for educational purposes only.
Not intended for real medical advice or diagnosis.
Always consult a qualified doctor.