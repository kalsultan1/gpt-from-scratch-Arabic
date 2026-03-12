import os
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
data_path = f"{project_root}/data/pretrain/data.txt"
checkpoint_dir = f"{project_root}/checkpoints/pretrained"

os.makedirs(checkpoint_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

print(f"Training device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

vocab_size = 8000
max_seq_len = 128

batch_size = 16
epochs = 2
lr = 3e-4

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
# LOAD + TOKENIZE DATA
# -----------------------------
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

token_ids = sp.encode(text, out_type=int)
print("Total tokens:", len(token_ids))


# -----------------------------
# DATASET
# -----------------------------
class GPTDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


dataset = GPTDataset(token_ids, max_seq_len)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=(device == "cuda"),
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

        if step % 100 == 0:
            print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss)

    print(f"\nEpoch {epoch} complete")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}\n")

    ckpt_path = f"{checkpoint_dir}/gpt_epoch_{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


# -----------------------------
# SAMPLE GENERATION
# -----------------------------
model.eval()

prompt = "كان يا ما كان"
ids = sp.encode(prompt, out_type=int)

x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    generated = model.generate(
        x,
        max_new_tokens=50,
        temperature=1.0,
        top_k=40
    )

tokens = generated[0].tolist()
text = sp.decode(tokens)

print("\nGenerated text:\n")
print(text)