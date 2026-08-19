"""AR-to-dLM conversion, following Efficient-DLM (arXiv:2512.14067).

Three pieces, each traceable to the paper:

  1. Block-wise attention (Sec. 2): causal ACROSS blocks, bidirectional WITHIN
     each block, with each corrupted block conditioned on CLEAN context x_<b.
     The paper's central finding is that this preserves the pretrained AR
     weight distribution better than full bidirectional attention does, and it
     is what keeps KV caching usable across blocks.

  2. Position-dependent masking (Sec. 3.2): w_i(t) = exp[beta * (1 - t) * i],
     parameterized by half-life ratio lambda = ln2 / (beta * L'). Later tokens
     in a block are masked more often, matching the left-to-right order that
     confidence-based decoding actually produces at test time. Their sweep put
     lambda=0.1 ahead of 0.25 and 0.05; pure right-to-left collapsed.

  3. The MDM objective (Eq. 1):
         L = E_t E_x~q [ -(1/t) * sum_b log p(x_b | x~_b^t, x_<b) ]
     The 1/t weight is what makes this a proper diffusion bound rather than
     plain masked-LM training.

SCALE CAVEAT. The paper converts Qwen2.5 1.5B with 50B tokens and states that
recovery needs ~10B. This module is being run against a 50M model over 200M
tokens, which is a mechanism test -- does the machinery train and does loss
descend -- not a reproduction of the accuracy or throughput results.

MASK TOKEN. The 16,384 vocab is fully occupied (max id 16383), so there is no
free slot. We reuse <unk> (id 1), verified absent from both train.bin and
heldout.bin, so it carries no learned meaning. Reusing an existing id avoids
resizing the embedding, which would perturb exactly the weight distribution
the paper says must be preserved.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MASK_TOKEN_ID = 1  # <unk>; unused in the corpus, see module docstring.


def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Boolean [T, T] mask: True where attention is ALLOWED.

    Position j is visible to position i when j lies in a strictly earlier block
    (clean context) or in the same block (bidirectional within the block). That
    is Fig. 2(d) of the paper: block-wise attention with clean context.

    Note this is deliberately NOT the union of a causal mask -- within a block a
    token sees its right-hand neighbours, which is the whole point of the dLM.
    """
    idx = torch.arange(seq_len, device=device)
    block = idx // block_size
    return block.unsqueeze(1) >= block.unsqueeze(0)  # [T, T], query i, key j


def log_position_weights(block_size: int, t: torch.Tensor, beta: float) -> torch.Tensor:
    """log w_i(t) for w_i(t) = exp[beta * (1 - t) * i] (Eq. 2), log-normalized.

    `t` is [B] noise levels in (0, 1]. Returns [B, block_size]. At t -> 1 the
    tilt vanishes (uniform); at t -> 0 it is maximal, concentrating masking on
    late positions.

    Deliberately returns LOG weights. Normalizing in probability space makes
    early positions underflow to exactly 0 at strong tilt, and the Gumbel-top-k
    below then takes log(0) = -inf, which sorts to NaN and masks nothing.
    """
    i = torch.arange(block_size, device=t.device, dtype=t.dtype)
    logits = beta * (1.0 - t).unsqueeze(1) * i.unsqueeze(0)
    return logits - torch.logsumexp(logits, dim=1, keepdim=True)


def position_weights(block_size: int, t: torch.Tensor, beta: float) -> torch.Tensor:
    """Probability form of Eq. 2, for inspection and tests."""
    return log_position_weights(block_size, t, beta).exp()


def beta_from_lambda(lam: float, block_size: int) -> float:
    """Invert lambda = ln2 / (beta * L') to get beta. lam <= 0 means uniform."""
    if lam <= 0:
        return 0.0
    return math.log(2.0) / (lam * block_size)


def corrupt(input_ids: torch.Tensor, block_size: int, beta: float,
            generator: torch.Generator | None = None,
            t_min: float = 0.05
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply position-dependent masking blockwise.

    Returns (noisy_ids, mask_bool, t) where mask_bool marks the positions that
    were replaced by MASK_TOKEN_ID and t is the [B] noise level used, which the
    loss needs for its 1/t weighting.

    Sampling follows the paper: draw t ~ U(0,1], take k = ceil(t * L') tokens
    per block, and select which k via Gumbel-top-k over the position weights.
    """
    b, seq_len = input_ids.shape
    device = input_ids.device
    n_blocks = seq_len // block_size

    # The loss carries a 1/t factor, so t near zero makes one sequence dominate
    # the whole batch: at t=1e-3 a single row is weighted 1000x, which showed up
    # as loss spikes to ~175 with grad_norm ~1650 at microbatch 32. A floor of
    # 0.05 caps that leverage at 20x. The paper samples t ~ U(0,1] and averages
    # the variance away over a much larger batch than this project runs.
    t = torch.rand(b, device=device, generator=generator).clamp_min(t_min)
    k = torch.ceil(t * block_size).long().clamp(1, block_size)  # [B]

    # Gumbel-top-k over log-weights gives a without-replacement sample drawn
    # proportional to w, vectorized across batch and blocks in one shot.
    logw = log_position_weights(block_size, t, beta).unsqueeze(1).expand(
        b, n_blocks, block_size)
    # Gumbel(0,1) = -log(-log(u)). Both logs need guarding: u==0 breaks the
    # inner log, and u close to 1 drives -log(u) to 0 and breaks the outer one.
    # A single clamp on the inner term is not enough -- that bug produced -inf
    # keys, which sort to NaN and select nothing at all.
    u = torch.rand(b, n_blocks, block_size, device=device,
                   generator=generator).clamp(1e-9, 1.0 - 1e-7)
    g = -torch.log(-torch.log(u))
    keys = logw + g                                        # [B, n_blocks, L']

    # Select the k highest-key positions per block. Comparing each key against
    # its own row's k-th largest value keeps this a pure top-k: an earlier
    # version scattered ranks and accidentally selected by position instead of
    # by weight, which inverted the tilt (early tokens masked most).
    kth = keys.sort(dim=-1, descending=True).values.gather(
        -1, (k - 1).view(b, 1, 1).expand(b, n_blocks, 1))  # [B, n_blocks, 1]
    mask = keys >= kth                                     # [B, n_blocks, L']
    mask = mask.reshape(b, n_blocks * block_size)

    if mask.shape[1] < seq_len:  # tail shorter than a full block: leave clean
        mask = F.pad(mask, (0, seq_len - mask.shape[1]), value=False)

    noisy = input_ids.masked_fill(mask, MASK_TOKEN_ID)
    return noisy, mask, t


def mdm_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
    """Eq. 1: -(1/t) * sum over masked positions of log p(x | x~, x_<b).

    Only masked positions contribute -- unmasked ones are given context, so
    scoring them would be free credit. Normalized per sequence by its own
    masked count, then averaged, so a batch is not dominated by whichever
    sequence happened to draw a high t.
    """
    b, seq_len, vocab = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        targets.reshape(-1),
        reduction="none",
    ).view(b, seq_len)

    nll = nll * mask
    counts = mask.sum(dim=1).clamp_min(1)
    per_seq = nll.sum(dim=1) / counts
    return (per_seq / t).mean()


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int,
             block_size: int, confidence: float = 0.9,
             max_steps_per_block: int | None = None) -> torch.Tensor:
    """Confidence-threshold parallel decoding, one block at a time.

    Each block starts fully masked and is denoised over repeated forward passes;
    every masked position whose softmax confidence clears `confidence` is
    committed in that pass. If none clears, the single most confident position
    is committed so the loop cannot stall. Tokens per forward (TPF) > 1 is where
    a dLM's speed would come from.
    """
    model.eval()
    device = prompt_ids.device
    out = prompt_ids
    steps = max_steps_per_block or block_size
    # The block mask must be applied at decode time too. Without it the model
    # runs causal and the within-block bidirectionality it was trained with
    # disappears -- the generated text would come from a different model than
    # the one that was converted.
    total_len = prompt_ids.shape[1] + max_new_tokens
    mask = block_causal_mask(total_len, block_size, device)

    produced = 0
    while produced < max_new_tokens:
        width = min(block_size, max_new_tokens - produced)
        block = torch.full((out.shape[0], width), MASK_TOKEN_ID,
                           dtype=out.dtype, device=device)
        out = torch.cat([out, block], dim=1)
        start = out.shape[1] - width

        for _ in range(steps):
            todo = out[:, start:] == MASK_TOKEN_ID
            if not todo.any():
                break
            logits = model(out, attn_mask=mask).logits[:, start:, :]
            probs = logits.float().softmax(dim=-1)
            conf, pred = probs.max(dim=-1)

            commit = todo & (conf >= confidence)
            # Guarantee forward progress: if the threshold admits nothing for a
            # row, take that row's single best masked position.
            stalled = todo.any(dim=1) & ~commit.any(dim=1)
            if stalled.any():
                masked_conf = conf.masked_fill(~todo, -1.0)
                best = masked_conf.argmax(dim=1)
                commit[stalled, best[stalled]] = True

            out[:, start:] = torch.where(commit, pred, out[:, start:])

        produced += width

    return out
