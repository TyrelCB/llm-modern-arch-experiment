"""Keep the spin-off architecture contract synchronized with executable code."""
from __future__ import annotations

from dataclasses import replace
import json
import re
from pathlib import Path

import torch

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM


ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "docs" / "architecture.json"
DECISIONS_PATH = ROOT / "docs" / "decisions.md"


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def test_manifest_canonical_switches_match_config_defaults():
    spec = load_spec()
    config = ModernConfig()

    assert spec["architecture_id"] == "dense-preln-v1"
    assert spec["interface"]["vocab_size"] == config.vocab_size
    assert spec["interface"]["max_sequence_length"] == config.max_seq_len
    assert spec["model"]["dropout"] == config.dropout
    assert spec["model"]["block"]["normalization"]["epsilon"] == config.norm_eps
    assert spec["model"]["block"]["attention"]["position"]["theta"] == config.rope_theta
    assert config.use_qk_norm
    assert not config.tie_embeddings
    assert not config.use_siamese_norm
    assert not config.use_mtp
    assert not config.use_moe


def test_manifest_profile_parameter_counts_match_models():
    spec = load_spec()

    for name, profile in spec["profiles"].items():
        config = replace(
            ModernConfig(),
            dim=profile["dim"],
            n_layers=profile["n_layers"],
            n_heads=profile["n_heads"],
            n_kv_heads=profile["n_kv_heads"],
            ffn_dim=profile["ffn_dim"],
        )
        # Meta tensors preserve shapes and names without allocating up to 600M
        # parameters merely to verify a documentation contract.
        with torch.device("meta"):
            model = ModernLM(config)

        stored = sum(parameter.numel() for parameter in model.parameters())
        body = sum(
            parameter.numel()
            for parameter_name, parameter in model.named_parameters()
            if "embedding" not in parameter_name and "lm_head" not in parameter_name
        )
        non_embedding = stored - model.embedding.weight.numel()

        assert stored == profile["stored_parameters"], name
        assert body == profile["body_parameters"], name
        assert non_embedding == profile["non_embedding_compute_bearing_parameters"], name


def test_manifest_decisions_and_source_paths_resolve():
    spec = load_spec()
    ledger = DECISIONS_PATH.read_text()
    anchors = set(re.findall(r'<a id="(d\d{3})"></a>', ledger))
    referenced = set(re.findall(r'D\d{3}', json.dumps(spec)))

    assert {decision.lower() for decision in referenced} <= anchors
    assert set(spec["decision_links"]) == referenced
    for relative_path in spec["implementation"].values():
        assert (ROOT / relative_path).is_file(), relative_path
