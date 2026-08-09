"""A resume keeps the optimizer settings the CLI asked for, not the checkpoint's.

torch's load_state_dict overwrites every non-tensor param-group key from the
checkpoint. For the combined Muon optimizer that includes `lr_scale`, so a
continued-pretraining run silently inherited the original run's Muon LR and
ignored --muon-learning-rate entirely -- a 5x reduction moved weight drift by
0.06pp because the flag never reached the optimizer.
"""
import torch

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.muon import build_optimizer
from modern_lm.train import load_checkpoint, save_checkpoint, TrainSettings


def _model():
    return ModernLM(ModernConfig(vocab_size=128, max_seq_len=32, dim=32,
                                 n_layers=2, n_heads=4, n_kv_heads=4, ffn_dim=64))


def _seed_optimizer_state(model, optimizer):
    """Populate momentum buffers without going through the compiled Muon step.

    zeropower_via_newtonschulz is torch.compile'd; calling it here would add
    dynamo cache entries and, once the rest of the suite has filled that cache,
    blow the recompile limit. The buffers are what the resume has to preserve,
    so write them directly.
    """
    # Only the Muon child (index 0); AdamW's state has its own required shape.
    muon = optimizer.optimizers[0]
    for group in muon.param_groups:
        for parameter in group["params"]:
            muon.state[parameter]["momentum_buffer"] = torch.randn_like(parameter)


def test_resume_keeps_the_requested_muon_lr(tmp_path):
    original = _model()
    saved_optimizer = build_optimizer(original, learning_rate=3e-4,
                                      muon_learning_rate=0.005,
                                      weight_decay=0.1, muon_weight_decay=None)
    # Real state to restore, so this is not a trivial no-op.
    _seed_optimizer_state(original, saved_optimizer)

    settings = TrainSettings(optimizer="muon", muon_learning_rate=0.005)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, original, saved_optimizer, original.config, settings,
                    {"tokens_seen": 1, "optimizer_step": 1})

    resumed = _model()
    optimizer = build_optimizer(resumed, learning_rate=3e-4,
                                muon_learning_rate=0.001,
                                weight_decay=0.1, muon_weight_decay=None)
    load_checkpoint(path, resumed, optimizer)

    muon_group = optimizer.param_groups[0]
    assert muon_group["lr_scale"] == 0.001, "resume clobbered the requested Muon LR"


def test_resume_still_restores_momentum(tmp_path):
    original = _model()
    optimizer = build_optimizer(original, learning_rate=3e-4, muon_learning_rate=0.005,
                                weight_decay=0.1, muon_weight_decay=None)
    _seed_optimizer_state(original, optimizer)

    settings = TrainSettings(optimizer="muon", muon_learning_rate=0.005)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, original, optimizer, original.config, settings,
                    {"tokens_seen": 1, "optimizer_step": 1})

    resumed = _model()
    fresh = build_optimizer(resumed, learning_rate=3e-4, muon_learning_rate=0.001,
                            weight_decay=0.1, muon_weight_decay=None)
    load_checkpoint(path, resumed, fresh)

    buffers = [state for state in fresh.optimizers[0].state.values()
               if "momentum_buffer" in state]
    assert buffers, "resume dropped the Muon momentum buffers"
