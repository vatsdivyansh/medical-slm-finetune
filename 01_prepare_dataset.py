from datasets import load_dataset
import json
import os


print("Loading medical dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
print(f"Total samples: {len(dataset['train'])}")


print("\nRaw sample:")
print(dataset['train'][0])


def format_sample(sample):
    return {
        "messages": [
            {
                "role"   : "user",
                "content": sample["input"]
            },
            {
                "role"   : "assistant",
                "content": sample["output"]
            }
        ]
    }


print("\nFormatting dataset...")
formatted = dataset["train"].map(format_sample)


small_dataset = formatted.select(range(5000))


split = small_dataset.train_test_split(test_size=0.1)
print(f"Train samples : {len(split['train'])}")
print(f"Test samples  : {len(split['test'])}")

split.save_to_disk("data/medical_dataset")
print("\nDataset saved to data/medical_dataset ✅")

print("\nFormatted sample preview:")
print(json.dumps(split['train'][0]['messages'], indent=2))