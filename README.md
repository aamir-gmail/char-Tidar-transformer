# TiDAR: Think in Diffusion, Talk in Autoregression (NVIDIA, 2025)

## I have kept the implementation small, so you can train and run even on a laptop

## This character level transformers combine two technologies into one.
* LLM for next token predection and a masked  text diffusion model for futture tokens
* Paper: TiDAR - Think in Diffusion, Talk in Autoregression, 2025
* In benifit is about 4x speedup in text generation.

## Train script for version three are self-contain with no external dependencies.

## Core Architecture: The Single-Model Concept
* **The Paradigm:** TiDAR rejects the traditional two-model speculative decoding setup. 
* **The Unified Design:** It executes parallel token drafting ("thinking") and high-fidelity sequential verification ("talking") inside a **single model** during a **single forward pass**.
* **The Efficiency:** It scales up to an **8B parameter size**, delivering a **4.7x to 5.9x throughput speedup** with effectively zero memory overhead and no extra draft model tracking.

## Inside the Single Forward Pass
* **Hybrid Attention Masking:** Split into two distinct functional zones processed simultaneously in one GPU operation:
  * *Autoregressive Zone:* A standard causal staircase mask verifies past text sequentially.
  * *Diffusion Zone:* A bidirectional mask maps out multiple future token slots all at once.
* **The Isolation Wall:** Autoregressive validation paths are strictly blocked from seeing the unverified draft slots to keep output quality safe from unverified text guesses.

## Mathematical Engine & Rejection Sampling
* **Marginal Distributions:** Standard LLMs rely on strict step-by-step joint distributions. TiDAR’s diffusion engine calculates the *marginal distribution* instead. This lets it predict a token 5 slots out without needing to lock down slots 1 to 4 first.
* **Rejection Sampling Integration:** The autoregressive engine runs an acceptance probability check (\(\alpha\)) over the diffusion drafts:
$$\alpha = \min\left(1, \frac{p_{\text{AR}}(x_i \mid x_{<i})}{p_{\text{Diff}}(x_i)}\right)$$
* **Lossless Output Guarantee:** If a guess fails the check, the model instantly flushes subsequent drafts and runs a mathematical recovery step. This ensures the final output exactly matches a slow, native autoregressive model.

## Anticipatory Pre-Drafting
* **No Compute Gaps:** Traditional speculative decoding stalls while the small model drafts the next set of words. 
* **Branching Speculation:** TiDAR uses idle GPU tensor slots during the current validation phase to compute future draft token branches for both acceptance and rejection paths simultaneously. 
* **Zero Net Latency:** When a validation check drops a token, the model instantly pivots to the pre-compiled fallback path with no drafting delay.

## Current Release Status
* **Weights Status:** TiDAR 1.5B and 8B are **not officially available for download**. They currently exist as NVIDIA academic research.
* **Current Alternative:** Open-source token-based diffusion can be tested today via **LLaDA**, which served as TiDAR's baseline comparison model.
