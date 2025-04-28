from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

import torch

dataset = load_dataset("IlyaGusev/gazeta")

model_name = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or '[PAD]'})
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_source_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

train_dataset = dataset["train"].shuffle(seed=42)
val_dataset = dataset["validation"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest'
)

training_args = TrainingArguments(
    output_dir="./mt5_summarization_keywords",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=200,
    logging_dir="./logs_keywords",
    dataloader_num_workers=4,
    report_to="none",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
trainer.train()

trainer.save_model("./mt5_summarization_keywords_final")
tokenizer.save_pretrained("./mt5_summarization_keywords_final")


def generate_summary_long(text, chunk_size=800, overlap=100, ratio=0.2, min_len=30, max_len=300):
    model.eval()
    sentences = nltk.sent_tokenize(text, language='russian')
    chunks, current = [], []
    for sent in sentences:
        current.append(sent)
        num_tokens = len(tokenizer.encode(" ".join(current), add_special_tokens=False))
        if num_tokens > chunk_size:
            last = current.pop()
            chunks.append(" ".join(current))
            overlap_sents = current[-(overlap//len(sent)):] if overlap//len(sent)>0 else []
            current = overlap_sents + [last]
    if current:
        chunks.append(" ".join(current))

    summary_parts = []
    for chunk in chunks:
        prefix = "summarize: " + chunk
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=chunk_size)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        in_len = inputs['input_ids'].size(-1)
        max_l = min(max_len, max(min_len, int(in_len * ratio)))
        out = model.generate(
            **inputs,
            max_length=max_l,
            min_length=min_len,
            num_beams=4,
            early_stopping=True
        )
        summary_parts.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
    return " ".join(summary_parts)

test_dataset = test_dataset.select(range(10))

for example in test_dataset:
    print("Original:\n", example['text'][:500], "...\n")
    print("Summary:\n", generate_summary_long(example['text']), "\n", "-"*50)