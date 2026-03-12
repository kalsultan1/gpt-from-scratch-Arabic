import os
import json
import math
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

sys.path.append("/content/drive/MyDrive/gpt-from-scratch")
from src.model.gpt import GPT


# -----------------------------
# CONFIG
# -----------------------------
project_root = "/content/drive/MyDrive/gpt-from-scratch"

tokenizer_path = f"{project_root}/src/tokenizer/arabic_bpe.model"
sft_path = f"{project_root}/data/finetune/sft_data.json"

pretrained_ckpt = f"{project_root}/checkpoints/pretrained/gpt_epoch_1.pt"
finetuned_dir = f"{project_root}/checkpoints/finetuned"
os.makedirs(finetuned_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

print(f"Training device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

vocab_size = 8000
max_seq_len = 128

batch_size = 8
epochs = 2
lr = 1e-4

d_model = 256
num_heads = 4
num_layers = 4
dropout = 0.1


# -----------------------------
# LOAD TOKENIZER
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
print("Tokenizer loaded")


# -----------------------------
# LOAD SFT DATA
# -----------------------------
with open(sft_path, "r", encoding="utf-8") as f:
    sft_data = json.load(f)

print("SFT samples:", len(sft_data))


# -----------------------------
# FORMAT EXAMPLES
# -----------------------------
def format_example(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    text = (
        "<bos>\n"
        "### التعليمات:\n"
        f"{instruction}\n\n"
        "### المدخلات:\n"
        f"{input_text}\n\n"
        "### الاستجابة:\n"
        f"{output}\n"
        "<eos>"
    )
    return text


encoded_examples = []
for ex in sft_data:
    text = format_example(ex)
    ids = sp.encode(text, out_type=int)
    if len(ids) >= 2:
        encoded_examples.append(ids[:max_seq_len + 1])

print("Encoded samples:", len(encoded_examples))


# -----------------------------
# DATASET
# -----------------------------
class SFTDataset(Dataset):
    def __init__(self, encoded_examples, seq_len, pad_id=0):
        self.examples = encoded_examples
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]

        if len(ids) < self.seq_len + 1:
            ids = ids + [self.pad_id] * (self.seq_len + 1 - len(ids))
        else:
            ids = ids[:self.seq_len + 1]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


dataset = SFTDataset(encoded_examples, max_seq_len)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device == "cuda")
)

print("Dataset size:", len(dataset))


# -----------------------------
# MODEL
# -----------------------------
model = GPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
print("Loaded pretrained checkpoint")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

num_params = sum(p.numel() for p in model.parameters())
print("Model parameters:", num_params)


# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0:
            print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss)

    print(f"\nEpoch {epoch} complete")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}\n")

    ckpt_path = f"{finetuned_dir}/gpt_finetuned_epoch_{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


# -----------------------------
# SAMPLE GENERATION
# -----------------------------
model.eval()

prompt = (
    "### التعليمات:\n"
    "اكتب قصة قصيرة عن طفل وجد مفتاحاً غامضاً.\n\n"
    "### المدخلات:\n"
    "\n\n"
    "### الاستجابة:\n"
)

ids = sp.encode(prompt, out_type=int)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    generated = model.generate(
        x,
        max_new_tokens=80,
        temperature=0.9,
        top_k=40
    )

tokens = generated[0].tolist()
text = sp.decode(tokens)

print("\nGenerated fine-tuned text:\n")
print(text)