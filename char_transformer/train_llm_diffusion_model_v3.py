import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import logging
from dataclasses import dataclass, field
import sys
from datasets import load_dataset
import string
import math

"""
This character level transformers combine two technologies into one.
LLM for next token predection and a masked  text diffusion model
for futture tokens. Paper: TiDAR - Think in Diffusion, Talk in Autoregression, 2025
In benifit is about 4x speedup in text generation.
"""


# -----------------------------------------------------------------------------
# 1. Configuration & Rich Logging
# -----------------------------------------------------------------------------
@dataclass
class TiDARConfig:
    """
    Configuration for the TiDAR Hybrid Architecture.
    Version 3: 20K Iters + Cosine Decay Learning Rate Scheduler
    """
    # Dataset
    # Data set will be auto downloaded from HF
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    max_chars: int = 50_000_000

    # Model Architecture
    # vocab size and masked token id is claculated dynamically 
    vocab_size: int = None
    mask_token_id: int = None
    block_size: int = 128
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    # TiDAR Specific , from original paper.
    loss_alpha: float = 1.0
    draft_length: int = 16

    # Training & Checkpointing
    # updated as new in the third version 
    batch_size: int = 64
    max_iters: int = 20000          # <--- UPDATED: Increased to 20K
    learning_rate: float = 3e-4
    min_lr: float = 3e-5            # <--- NEW: 10% of peak learning rate
    warmup_iters: int = 1000        # <--- NEW: Ramp up for first 1000 steps
    lr_decay_iters: int = 20000     # <--- NEW: Match max_iters
    eval_interval: int = 500
    eval_iters: int = 20

    # System & Run Modes
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir: str = "checkpoints"
    resume: bool = True
    eval_only: bool = False

# Setup Rich Logging
logger = logging.getLogger("TiDAR_CharLevel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)


# -----------------------------------------------------------------------------
# 2. Dataset Processing (Wikitext-103 Filtered)
# -----------------------------------------------------------------------------
def get_data(config: TiDARConfig):
    logger.info(f"Fetching {config.dataset_name} ({config.dataset_config})...")
    dataset = load_dataset(config.dataset_name, config.dataset_config)

    logger.info("Concatenating text chunks...")
    text_chunks = []
    current_length = 0
    for row in dataset['train']:
        if row['text']:
            text_chunks.append(row['text'])
            current_length += len(row['text'])
            if current_length > config.max_chars:
                break

    raw_text = "".join(text_chunks)[:config.max_chars]

    logger.info("Filtering out non-English Unicode characters...")
    allowed_chars = set(string.printable)
    text = "".join(c for c in raw_text if c in allowed_chars)

    chars = sorted(list(set(text)))

    config.mask_token_id = len(chars)
    config.vocab_size = len(chars) + 1

    logger.info(f"Dataset loaded. Filtered Length: {len(text):,} characters (Raw was {len(raw_text):,}).")
    logger.info(f"Vocab size: {config.vocab_size} (Includes 1 special [MASK] token at ID {config.mask_token_id})")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    itos[config.mask_token_id] = "[MASK]"

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos.get(i, "") for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    logger.info(f"Train data split: {len(train_data):,} | Val data split: {len(val_data):,}")

    return train_data, val_data, encode, decode

def get_batch(data, config: TiDARConfig):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    return x.to(config.device)


# -----------------------------------------------------------------------------
# 3. Model Architecture (TiDAR Hybrid) LLM + Masked Diffusion in a single step.
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size, config: TiDARConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(attn_mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TiDARConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask):
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, attn_mask):
        x = x + self.sa(self.ln1(x), attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class TiDARTransformer(nn.Module):
    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size * 2, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.register_buffer('tidar_mask', self._build_hybrid_mask())
        self.apply(self._init_weights)

    def _build_hybrid_mask(self):
        T = self.config.block_size
        mask = torch.zeros(2 * T, 2 * T)
        # LLM auto regressive Zone (Causal)
        mask[:T, :T] = torch.tril(torch.ones(T, T))
        # Diffusion Zone (Bidirectional mapping to clean context)
        mask[T:, :T] = 1.0
        mask[T:, T:] = 1.0
        return mask

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        idx_ar = idx
        idx_diff = torch.full((B, T), self.config.mask_token_id, dtype=torch.long, device=self.config.device)
        hybrid_idx = torch.cat([idx_ar, idx_diff], dim=1)

        tok_emb = self.token_embedding_table(hybrid_idx)
        pos_emb = self.position_embedding_table(torch.arange(2 * T, device=self.config.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, self.tidar_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        logits_ar = logits[:, :T, :]
        logits_diff = logits[:, T:, :]
        return logits_ar, logits_diff


# -----------------------------------------------------------------------------
# 4. Utilities: Training, Checkpointing, & Generation Logs
# -----------------------------------------------------------------------------
def calculate_tidar_loss(logits_ar, logits_diff, targets, config: TiDARConfig):
    B, T = targets.shape
    V = config.vocab_size
    flat_logits_ar = logits_ar[:, :-1, :].reshape(-1, V)
    flat_targets_ar = targets[:, 1:].reshape(-1)
    loss_ar = F.cross_entropy(flat_logits_ar, flat_targets_ar)

    flat_logits_diff = logits_diff.reshape(-1, V)
    flat_targets_diff = targets.reshape(-1)
    loss_diff = F.cross_entropy(flat_logits_diff, flat_targets_diff)

    return (config.loss_alpha * loss_ar + loss_diff) / (1.0 + config.loss_alpha)


def save_checkpoint(model, optimizer, iter_num, config: TiDARConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iter_num': iter_num,
    }
    path = os.path.join(config.checkpoint_dir, 'tidar_ckpt_latest.pt')
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, config: TiDARConfig):
    path = os.path.join(config.checkpoint_dir, 'tidar_ckpt_latest.pt')
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        iter_num = checkpoint['iter_num']
        logger.info(f"Checkpoint loaded successfully. Resuming from iteration {iter_num}")
        return iter_num
    else:
        logger.warning(f"No checkpoint found at {path}. Starting from scratch.")
        return 0


@torch.no_grad()
def inspect_generation(model, encode, decode, val_data, config: TiDARConfig):
    """
    VERSION 3: True Speculative Generation Simulation.
    We append actual [MASK] tokens to a context string, force the diffusion
    engine to predict the future, and use the AR engine to verify it.
    """
    model.eval()

    # Define how much context we have vs how much we want to draft
    ctx_len = config.block_size - config.draft_length
    ix = torch.randint(len(val_data) - config.block_size, (1,)).item()

    # Grab a real sequence, but split it into "Context" and "Future"
    real_sequence = val_data[ix : ix + config.block_size].to(config.device)
    clean_ctx = real_sequence[:ctx_len]
    actual_future = real_sequence[ctx_len:] # For logging comparison only

    # =========================================================================
    # PHASE 1: THE DRAFTER (Thinking in Diffusion)
    # We append [MASK] tokens to the clean context and predict them all at once.
    # =========================================================================
    mask_padding = torch.full((config.draft_length,), config.mask_token_id, dtype=torch.long, device=config.device)
    idx_draft = torch.cat([clean_ctx, mask_padding]).unsqueeze(0)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, logits_diff = model(idx_draft)

    # Extract the diffusion engine's predictions for the masked slots
    diff_preds = torch.argmax(logits_diff[0, ctx_len:], dim=-1)

    # =========================================================================
    # PHASE 2: THE VERIFIER (Talking in Autoregression)
    # We append the drafted tokens to the clean context and let the AR engine
    # verify them sequentially step-by-step.
    # =========================================================================
    idx_verify = torch.cat([clean_ctx, diff_preds]).unsqueeze(0)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits_ar_verify, _ = model(idx_verify)

    # Extract the AR engine's predictions.
    # The AR prediction *for* position ctx_len is found at index ctx_len - 1.
    ar_preds = torch.argmax(logits_ar_verify[0, ctx_len - 1 : config.block_size - 1], dim=-1)

    # =========================================================================
    # PHASE 3: SPECULATIVE REJECTION SAMPLING
    # Compare the drafts against the AR verifier. The chain breaks on first failure.
    # =========================================================================
    combined = []
    for ar_tok, diff_tok in zip(ar_preds, diff_preds):
        if ar_tok.item() == diff_tok.item():
            combined.append(diff_tok.item())  # Accept draft token
        else:
            combined.append(ar_tok.item())    # Reject, fallback to the safe AR token
            break                             # Break the speculative chain!

    # Decode everything for standard logging
    print_ctx_len = min(40, ctx_len)
    context_str = decode(clean_ctx[-print_ctx_len:].tolist())
    actual_str = decode(actual_future.tolist())
    ar_str = decode(ar_preds.tolist())
    diff_str = decode(diff_preds.tolist())
    combined_str = decode(combined)

    print("\n" + "=" * 65)
    print("🔍 TiDAR TRUE SPECULATIVE GENERATION INSPECTION")
    print("=" * 65)
    print(f"Prompt Context (Last {print_ctx_len} chars): '{context_str.replace(chr(10), ' ')}'")
    print(f"Ground Truth Future (Hidden):             '{actual_str.replace(chr(10), ' ')}'")
    print("-" * 65)
    print(f"1. Diffusion Drafts (Parallel):           '{diff_str.replace(chr(10), ' ')}'")
    print(f"2. LLM AR Verification (Causal):          '{ar_str.replace(chr(10), ' ')}'")
    print(f"3. Final Combined Output:                 '{combined_str.replace(chr(10), ' ')}'")
    print("=" * 65 + "\n")

    model.train()


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, config: TiDARConfig):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            targets = get_batch(data, config)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits_ar, logits_diff = model(targets)
                loss = calculate_tidar_loss(logits_ar, logits_diff, targets, config)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, config: TiDARConfig):
    # warm up and learning rat added for version 3
    # 1. Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # 2. If it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr

    # 3. In between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def main():
    config = TiDARConfig()
    logger.info(f"Initializing TiDAR Hybrid architecture on device: {config.device.upper()}")

    train_data, val_data, encode, decode = get_data(config)

    model = TiDARTransformer(config)
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_iter = 0

    if config.resume or config.eval_only:
        start_iter = load_checkpoint(model, optimizer, config)

    if config.eval_only:
        logger.info("EVAL ONLY mode activated. Running evaluation...")
        losses = estimate_loss(model, train_data, val_data, config)
        logger.info(f"Val Loss: {losses['val']:.4f}")
        inspect_generation(model, encode, decode, val_data, config)
        return

    logger.info("Starting training loop with BF16 Mixed Precision & Cosine Decay...")
    for iter_num in range(start_iter, config.max_iters):

        # --- NEW: V3 improvements Dynamically update learning rate ---
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # ---------------------------------------------

        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            # Log the current learning rate as well so we can track the decay!
            logger.info(
                f"Step {iter_num:>4} | LR: {lr:.2e} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            inspect_generation(model, encode, decode, val_data, config)
            save_checkpoint(model, optimizer, iter_num, config)

        targets = get_batch(train_data, config)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits_ar, logits_diff = model(targets)
            loss = calculate_tidar_loss(logits_ar, logits_diff, targets, config)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    logger.info("Training complete.")

    logger.info("Training complete.")


if __name__ == '__main__':
    main()
