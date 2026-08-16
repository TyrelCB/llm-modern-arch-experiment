import time, torch
from dataclasses import replace
from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.muon import build_optimizer

base = replace(ModernConfig(), dim=576, n_layers=11, n_heads=9, n_kv_heads=9, ffn_dim=1984)


def bench(siamese):
    cfg = replace(base, use_siamese_norm=siamese)
    m = ModernLM(cfg).cuda()
    opt = build_optimizer(m, learning_rate=3e-4, muon_learning_rate=0.005,
                          weight_decay=0.1, muon_weight_decay=None)
    c = torch.compile(m)
    ids = torch.randint(0, cfg.vocab_size, (16, 512), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = c(ids).logits
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, cfg.vocab_size).float(), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(12):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(30):
        step()
    torch.cuda.synchronize()
    return 30 * 16 * 512 / (time.perf_counter() - t)


pre = bench(False)
print(f"pre-LN   {pre:,.0f} tok/s", flush=True)
sia = bench(True)
print(f"siamese  {sia:,.0f} tok/s   ({(sia/pre-1)*100:+.1f}%)", flush=True)
