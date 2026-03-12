import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer import DecoderBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token embedding: converts token IDs -> dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding: gives each position its own learned vector
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Stack of decoder-only Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # LM head: projects hidden states -> vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: weight tying (common in GPT-style models)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token IDs
        targets: (B, T) token IDs for next-token prediction

        returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.shape

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # Positions: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        # Embeddings
        tok_emb = self.token_embedding(idx)          # (B, T, d_model)
        pos_emb = self.position_embedding(positions) # (1, T, d_model)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Final logits over vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross entropy
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: (B, T) starting token IDs
        returns: extended token sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Keep only the last max_seq_len tokens
            idx_cond = idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)

            # Take logits of last token position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Optional top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_topk = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, float("-inf")),
                    logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx