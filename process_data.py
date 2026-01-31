import ast
import re

import pandas as pd
import unicodedata
from datasets import Dataset

print("Loading and processing datasets...")

df_mcq = pd.read_csv("data/train_dataset_mcq.csv")
df_saq = pd.read_csv("data/train_dataset_saq.csv")


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_text(text, min_len=5, max_len=512):
    if not text:
        return False
    if len(text) < min_len or len(text) > max_len:
        return False
    if re.fullmatch(r"[\W\d_]+", text):
        return False
    return True


# =====================
# Prompt Builders
# =====================
def build_saq_prompt(question, answer):
    return f"""### System:
        You are a knowledgeable and precise assistant.
        Answer questions using factual and verified information only.
        If the answer is unknown or ambiguous, state that clearly.
        
        ### Instruction:
        Read the question carefully and provide a concise and correct answer.
        Use clear, natural language.
        Do not add unnecessary explanations or assumptions.
        
        ### Question:
        {question}
        
        ### Answer:
        {answer}"""


def build_mcq_prompt(question, answer):
    # now if at the end of question there is Answer:, remove it

    question = re.sub(r"(?:answer:?\s*[A-D]?)\s*$", "", question, flags=re.IGNORECASE).strip()

    return f"""### System:
        You are an expert problem solver.
        Your task is to identify the single correct answer.
        
        ### Instruction:
        Analyze the question carefully.
        Select the correct option from the given choices.
        Respond using only the answer index or letter.
        Do not include explanations or extra text.
        
        ### Question:
        {question}
        
        ### Answer:
        {answer}"""


def get_best_saq_answer(annotation_str):
    try:
        annotations = ast.literal_eval(annotation_str)
        if not annotations:
            return ""
        best = max(annotations, key=lambda x: x.get("count", 0))
        if best.get("en_answers"):
            return best["en_answers"][0]
        if best.get("answers"):
            return best["answers"][0]
        return ""
    except Exception:
        return ""


saq_data = []
seen = set()

for _, row in df_saq.iterrows():
    question = row["en_question"] if pd.notna(row.get("en_question")) else row.get("question")
    answer = get_best_saq_answer(row.get("annotations", ""))

    question = normalize_text(question)
    answer = normalize_text(answer)

    if not (is_valid_text(question, min_len=10) and is_valid_text(answer, min_len=2)):
        continue

    text = build_saq_prompt(question, answer)

    if text not in seen:
        saq_data.append({"text": text})
        seen.add(text)

mcq_data = []

for _, row in df_mcq.iterrows():
    prompt = normalize_text(row.get("prompt"))
    answer_idx = row.get("answer_idx")

    if not is_valid_text(prompt, min_len=10):
        continue
    if pd.isna(answer_idx):
        continue

    answer = str(answer_idx).strip()
    # if at the end of prompt there is Answer: <letter>, remove it
    prompt = re.sub(r"Answer:\s*[A-Da-d]\s*$","", prompt).strip()
    text = build_mcq_prompt(prompt, answer)

    if text not in seen:
        mcq_data.append({"text": text})
        seen.add(text)


# save the fata in csv
with open("data/saq_data_sample.txt", "w", encoding="utf-8") as f:
    for item in saq_data[:5]:
        f.write(item["text"] + "\n\n")

with open("data/mcq_data_sample.txt", "w", encoding="utf-8") as f:
    for item in mcq_data[:5]:
        f.write(item["text"] + "\n\n")


full_data = saq_data + mcq_data
dataset = Dataset.from_list(full_data)

print(f"Total training examples: {len(dataset)}")
print("\nSample example:\n")
print(dataset[0]["text"])

# save processed dataset
dataset.to_csv("data/processed_dataset_cleaned.csv", index=False)

