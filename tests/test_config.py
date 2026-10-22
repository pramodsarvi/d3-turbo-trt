from __future__ import annotations

from pathlib import Path

from d3trt.config import load_config


def test_config_contract(model_dir: Path):
    config = load_config(model_dir)
    assert config.num_layers == 2
    assert len(config.layer_specs) == 2
    assert config.full_attention_layers == (1,)
    assert config.linear_attention_layers == (0,)
    assert config.attention_mode == "prefix_bidir"
    assert config.max_options == 255
    assert config.codes[0]
    assert len(config.codes) == len(config.token_ids) == 255
    assert config.hidden_size > 0
    assert config.kv_bytes_per_state_token == len(config.full_attention_layers) * 2 * config.num_kv_heads * config.head_dim * 2
    assert config.gdn_state_bytes > 0
    assert config.state_cache_slots() >= 1


def test_state_key_is_content_addressed(model_dir: Path):
    config = load_config(model_dir)
    a = config.state_key([1, 2, 3, 4])
    b = config.state_key([1, 2, 3, 4])
    c = config.state_key([1, 2, 3, 5])
    d = config.state_key([1, 2, 3, 4], media_hash="pixels-a")
    e = config.state_key([1, 2, 3, 4], media_hash="pixels-b")
    assert a == b
    assert a != c
    assert a != d
    assert d != e
    assert len(a) == 64


def test_describe(model_dir: Path):
    payload = load_config(model_dir).describe()
    assert payload["attention_mode"] == "prefix_bidir"
    assert payload["layers"] == 2
    assert payload["vision"] is True
    assert payload["codes"] == 255
