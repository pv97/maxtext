# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU unit & cross-repo E2E parity tests for MaxText M1 + M3 weight conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from maxtext.integration.vllm.maxtext_manifest_generator import (
    generate_maxtext_to_hf_manifest,
    generate_maxtext_to_hf_manifest_dict,
)
from maxtext.integration.vllm.weight_converter import WeightConverter
from maxtext.integration.vllm.weight_mapping import (
    SCHEMA_VERSION,
    execute_manifest,
    from_hf,
    from_hf_impl,
    invert_manifest,
    to_hf,
    to_hf_impl,
    validate_manifest,
)
import numpy as np
import pytest


def _m(rules: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
  return {
      "schema_version": SCHEMA_VERSION,
      "expert_axis_grouped": False,
      "stripes": {},
      "tied_keys": [],
      "dropped_keys": [],
      "rules": rules,
      **kw,
  }


def _round_trip_jax(manifest: dict[str, Any], src: dict[str, jax.Array]) -> dict[str, jax.Array]:
  validate_manifest(manifest)
  dst = jax.jit(lambda w: to_hf_impl(manifest, w))(src)
  inv = invert_manifest(manifest)
  validate_manifest(inv)
  rec = jax.jit(lambda w: from_hf_impl(inv, w))(dst)
  assert set(rec) == set(src)
  for k, v in src.items():
    assert rec[k].shape == v.shape and rec[k].dtype == v.dtype
    np.testing.assert_array_equal(np.asarray(rec[k]), np.asarray(v))
  # Verify jax.eval_shape works with zero FLOPs
  abs_src = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in src.items()}
  abs_dst = jax.eval_shape(lambda w: to_hf_impl(manifest, w), abs_src)
  for k, v in dst.items():
    assert abs_dst[k].shape == v.shape and abs_dst[k].dtype == v.dtype
  return dst


# ---------------------------------------------------------------------------
# M1: Pure-JAX 12-op executor, inverter, and validator
# ---------------------------------------------------------------------------


def test_m1_all_12_ops_jit_and_eval_shape_round_trip() -> None:
  rng = np.random.default_rng(0)
  bf16 = lambda *s: jnp.asarray(rng.standard_normal(s), dtype=jnp.bfloat16)  # noqa: E731

  # stack / unstack
  es = [bf16(4, 6) for _ in range(3)]
  _round_trip_jax(
      _m([{"source_keys": ["e0", "e1", "e2"], "target_keys": ["s"], "ops": [{"stack": {"axis": 0}}]}]),
      {"e0": es[0], "e1": es[1], "e2": es[2]},
  )
  # concat / split
  _round_trip_jax(
      _m([{"source_keys": ["a", "b"], "target_keys": ["c"], "ops": [{"concat": {"axis": 0, "sizes": [4, 6]}}]}]),
      {"a": bf16(4, 8), "b": bf16(6, 8)},
  )
  # reshape + transpose
  _round_trip_jax(
      _m(
          [
              {
                  "source_keys": ["x"],
                  "target_keys": ["y"],
                  "ops": [
                      {"reshape": {"src_shape": [2, 3, 4], "dst_shape": [6, 4]}},
                      {"transpose": {"axes": [1, 0]}},
                  ],
              }
          ]
      ),
      {"x": bf16(2, 3, 4)},
  )
  # pad / slice (and unpad alias)
  p = bf16(5, 4)
  dst_p = _round_trip_jax(
      _m([{"source_keys": ["p"], "target_keys": ["q"], "ops": [{"pad": {"axis": 0, "sizes": [1, 2]}}]}]),
      {"p": p},
  )
  unpadded = execute_manifest(
      _m([{"source_keys": ["q"], "target_keys": ["p"], "ops": [{"unpad": {"axis": 0, "sizes": [1, 2]}}]}]),
      dst_p,
  )
  np.testing.assert_array_equal(np.asarray(unpadded["p"]), np.asarray(p))

  # repeat / mean_pool (and unrepeat alias)
  r = bf16(2, 4)
  dst_r = _round_trip_jax(
      _m([{"source_keys": ["r"], "target_keys": ["rr"], "ops": [{"repeat": {"axis": 0, "times": 3}}]}]),
      {"r": r},
  )
  unrep = execute_manifest(
      _m([{"source_keys": ["rr"], "target_keys": ["r"], "ops": [{"unrepeat": {"axis": 0, "times": 3}}]}]),
      dst_r,
  )
  np.testing.assert_array_equal(np.asarray(unrep["r"]), np.asarray(r))

  # scale + cast
  s32 = jnp.asarray(rng.standard_normal((4, 4)), dtype=jnp.float32)
  _round_trip_jax(
      _m([{"source_keys": ["s"], "target_keys": ["t"], "ops": [{"scale": {"factor": 4.0}}]}]),
      {"s": s32},
  )
  _round_trip_jax(
      _m(
          [
              {
                  "source_keys": ["c"],
                  "target_keys": ["d"],
                  "ops": [{"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}],
              }
          ]
      ),
      {"c": bf16(4, 4)},
  )

  # JSON str wrappers enforce str
  with pytest.raises(TypeError, match="to_hf expects src_rule to be str"):
    to_hf(_m([]), {})  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="from_hf expects dst_rule to be str"):
    from_hf(_m([]), {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# M3: MaxText -> HF manifest generator (Qwen2.5 dense + Qwen3.5-MoE)
# ---------------------------------------------------------------------------


def _build_maxtext_qwen2_state(
    *,
    vocab_size: int = 64,
    padded_vocab_size: int = 64,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    num_layers: int = 1,
    scan_layers: bool = False,
    tie_word_embeddings: bool = True,
    seed: int = 42,
) -> tuple[Any, dict[str, jax.Array]]:
  rng = np.random.default_rng(seed)
  r = lambda *s: jnp.asarray(rng.standard_normal(s), dtype=jnp.bfloat16)  # noqa: E731
  hd = hidden_size // num_heads
  cfg = SimpleNamespace(
      decoder_block="qwen2",
      num_decoder_layers=num_layers,
      param_scan_axis=0,
      quantization="",
  )
  state: dict[str, jax.Array] = {
      "params.token_embedder.embedding": r(padded_vocab_size, hidden_size),
      "params.decoder.decoder_norm.scale": r(hidden_size),
  }
  if not tie_word_embeddings:
    state["params.decoder.logits_dense.kernel"] = r(hidden_size, padded_vocab_size)
  if scan_layers:
    p = "params.decoder.layers"
    state.update(
        {
            f"{p}.pre_self_attention_layer_norm.scale": r(num_layers, hidden_size),
            f"{p}.self_attention.query.kernel": r(num_layers, hidden_size, num_heads, hd),
            f"{p}.self_attention.query.bias": r(num_layers, num_heads, hd),
            f"{p}.self_attention.key.kernel": r(num_layers, hidden_size, num_kv_heads, hd),
            f"{p}.self_attention.key.bias": r(num_layers, num_kv_heads, hd),
            f"{p}.self_attention.value.kernel": r(num_layers, hidden_size, num_kv_heads, hd),
            f"{p}.self_attention.value.bias": r(num_layers, num_kv_heads, hd),
            f"{p}.self_attention.out.kernel": r(num_layers, num_heads, hd, hidden_size),
            f"{p}.post_self_attention_layer_norm.scale": r(num_layers, hidden_size),
            f"{p}.mlp.wi_0.kernel": r(num_layers, hidden_size, intermediate_size),
            f"{p}.mlp.wi_1.kernel": r(num_layers, hidden_size, intermediate_size),
            f"{p}.mlp.wo.kernel": r(num_layers, intermediate_size, hidden_size),
        }
    )
  else:
    for i in range(num_layers):
      p = f"params.decoder.layers_{i}"
      state.update(
          {
              f"{p}.pre_self_attention_layer_norm.scale": r(hidden_size),
              f"{p}.self_attention.query.kernel": r(hidden_size, num_heads, hd),
              f"{p}.self_attention.query.bias": r(num_heads, hd),
              f"{p}.self_attention.key.kernel": r(hidden_size, num_kv_heads, hd),
              f"{p}.self_attention.key.bias": r(num_kv_heads, hd),
              f"{p}.self_attention.value.kernel": r(hidden_size, num_kv_heads, hd),
              f"{p}.self_attention.value.bias": r(num_kv_heads, hd),
              f"{p}.self_attention.out.kernel": r(num_heads, hd, hidden_size),
              f"{p}.post_self_attention_layer_norm.scale": r(hidden_size),
              f"{p}.mlp.wi_0.kernel": r(hidden_size, intermediate_size),
              f"{p}.mlp.wi_1.kernel": r(hidden_size, intermediate_size),
              f"{p}.mlp.wo.kernel": r(intermediate_size, hidden_size),
          }
      )
  return cfg, state


def _build_maxtext_qwen35_moe_state(
    *,
    vocab_size: int = 64,
    hidden_size: int = 64,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    num_k_heads: int = 2,
    num_v_heads: int = 4,
    d_k: int = 8,
    d_v: int = 8,
    num_experts: int = 4,
    moe_intermediate_size: int = 192,
    shared_intermediate_size: int = 64,
    scan_layers: bool = False,
    seed: int = 123,
) -> tuple[Any, dict[str, jax.Array]]:
  rng = np.random.default_rng(seed)
  r = lambda *s, dt=jnp.bfloat16: jnp.asarray(rng.standard_normal(s), dtype=dt)  # noqa: E731
  v_per_k = num_v_heads // num_k_heads
  qkvz_dim = num_k_heads * (2 * d_k + 2 * v_per_k * d_v)
  ba_dim = num_k_heads * 2 * v_per_k
  conv_dim = 2 * num_k_heads * d_k + num_v_heads * d_v
  cfg = SimpleNamespace(
      decoder_block="qwen3_5",
      num_decoder_layers=2,
      inhomogeneous_layer_cycle_interval=2,
      param_scan_axis=0,
      quantization="",
      gdn_conv_num_key_heads=num_k_heads,
      gdn_conv_num_value_heads=num_v_heads,
      gdn_key_head_dim=d_k,
      gdn_value_head_dim=d_v,
  )
  state: dict[str, jax.Array] = {
      "params.token_embedder.embedding": r(vocab_size, hidden_size),
      "params.decoder.decoder_norm.scale": r(hidden_size),
      "params.decoder.logits_dense.kernel": r(hidden_size, vocab_size),
  }

  def _add_layer(prefix: str, is_global: bool, extra_dims: tuple[int, ...]) -> None:
    ed = extra_dims
    exp_ed = (num_experts, *ed)
    state.update(
        {
            f"{prefix}.input_layernorm.scale": r(*ed, hidden_size),
            f"{prefix}.post_attention_layernorm.scale": r(*ed, hidden_size),
            f"{prefix}.mlp.routed_experts.gate.kernel": r(*ed, hidden_size, num_experts),
            f"{prefix}.mlp.routed_experts.wi_0": r(*exp_ed, hidden_size, moe_intermediate_size),
            f"{prefix}.mlp.routed_experts.wi_1": r(*exp_ed, hidden_size, moe_intermediate_size),
            f"{prefix}.mlp.routed_experts.wo": r(*exp_ed, moe_intermediate_size, hidden_size),
            f"{prefix}.mlp.shared_expert_gate.kernel": r(*ed, hidden_size, 1),
            f"{prefix}.mlp.shared_expert.wi_0.kernel": r(*ed, hidden_size, shared_intermediate_size),
            f"{prefix}.mlp.shared_expert.wi_1.kernel": r(*ed, hidden_size, shared_intermediate_size),
            f"{prefix}.mlp.shared_expert.wo.kernel": r(*ed, shared_intermediate_size, hidden_size),
        }
    )
    if is_global:
      state.update(
          {
              f"{prefix}.attention.attention.query.kernel": r(*ed, hidden_size, 2 * num_heads, head_dim),
              f"{prefix}.attention.attention.key.kernel": r(*ed, hidden_size, num_kv_heads, head_dim),
              f"{prefix}.attention.attention.value.kernel": r(*ed, hidden_size, num_kv_heads, head_dim),
              f"{prefix}.attention.attention.out.kernel": r(*ed, num_heads, head_dim, hidden_size),
              f"{prefix}.attention.attention.query_norm.scale": r(*ed, head_dim),
              f"{prefix}.attention.attention.key_norm.scale": r(*ed, head_dim),
          }
      )
    else:
      state.update(
          {
              f"{prefix}.attention.conv1d.kernel": r(*ed, 4, 1, conv_dim),
              f"{prefix}.attention.in_proj_qkvz.kernel": r(*ed, hidden_size, qkvz_dim),
              f"{prefix}.attention.in_proj_ba.kernel": r(*ed, hidden_size, ba_dim),
              f"{prefix}.attention.A_log": r(*ed, num_v_heads, dt=jnp.float32),
              f"{prefix}.attention.dt_bias": r(*ed, num_v_heads),
              f"{prefix}.attention.norm.rms_norm.scale": r(*ed, d_v),
              f"{prefix}.attention.out_proj.kernel": r(*ed, num_v_heads * d_v, hidden_size),
          }
      )

  if scan_layers:
    _add_layer("params.decoder.layers.local_layers", is_global=False, extra_dims=(1, 1))
    _add_layer("params.decoder.layers.global_layer", is_global=True, extra_dims=(1,))
  else:
    _add_layer("params.decoder.layers_0", is_global=False, extra_dims=())
    _add_layer("params.decoder.layers_1", is_global=True, extra_dims=())
  return cfg, state


@pytest.mark.parametrize("scan_layers", [False, True])
def test_m3_qwen2_dense_to_hf_round_trip(scan_layers: bool) -> None:
  cfg, state = _build_maxtext_qwen2_state(num_layers=2, scan_layers=scan_layers, tie_word_embeddings=False)
  manifest = generate_maxtext_to_hf_manifest_dict(cfg, state)
  hf = _round_trip_jax(manifest, state)
  assert hf["model.layers.0.self_attn.q_proj.weight"].shape == (64, 64)
  assert hf["model.layers.1.mlp.gate_proj.weight"].shape == (128, 64)
  assert hf["lm_head.weight"].shape == (64, 64)


@pytest.mark.parametrize("scan_layers,expert_axis_grouped", [(False, True), (False, False), (True, True)])
def test_m3_qwen35_moe_to_hf_round_trip(scan_layers: bool, expert_axis_grouped: bool) -> None:
  cfg, state = _build_maxtext_qwen35_moe_state(scan_layers=scan_layers)
  manifest = generate_maxtext_to_hf_manifest_dict(cfg, state, expert_axis_grouped=expert_axis_grouped)
  assert manifest["expert_axis_grouped"] is expert_axis_grouped
  hf = _round_trip_jax(manifest, state)
  assert hf["model.layers.0.linear_attn.conv1d.weight"].shape == (64, 1, 4)
  assert hf["model.layers.0.linear_attn.in_proj_qkv.weight"].shape == (64, 64)
  assert hf["model.layers.0.linear_attn.in_proj_z.weight"].shape == (32, 64)
  assert hf["model.layers.0.linear_attn.in_proj_b.weight"].shape == (4, 64)
  assert hf["model.layers.0.linear_attn.in_proj_a.weight"].shape == (4, 64)
  assert hf["model.layers.1.self_attn.q_proj.weight"].shape == (128, 64)
  if expert_axis_grouped:
    assert hf["model.layers.0.mlp.experts.gate_proj.weight"].shape == (4, 192, 64)
  else:
    assert hf["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (192, 64)


def test_m3_fail_closed_guards() -> None:
  cfg, state = _build_maxtext_qwen2_state()
  with pytest.raises(NotImplementedError, match="Quantization"):
    generate_maxtext_to_hf_manifest(SimpleNamespace(decoder_block="qwen2", quantization="fp8"), state)
  with pytest.raises(NotImplementedError, match="decoder_block"):
    generate_maxtext_to_hf_manifest(SimpleNamespace(decoder_block="unsupported_arch"), state)
  bad_state = {**state, "params.unknown.weight": jnp.zeros((4,), dtype=jnp.bfloat16)}
  with pytest.raises(NotImplementedError, match="Unclaimed"):
    generate_maxtext_to_hf_manifest(cfg, bad_state)


# ---------------------------------------------------------------------------
# Scenario 1 Cross-Repo E2E Parity: MaxText (JAX) -> vllm-torchtpu (PyTorch)
# ---------------------------------------------------------------------------


def _maybe_import_torchtpu():
  torchtpu_root = Path(__file__).resolve().parents[4] / "torchtpu"
  fakes_path = torchtpu_root / "tests" / "rl" / "_fakes.py"
  if not fakes_path.exists():
    return None
  src_dir = str(torchtpu_root / "src")
  if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
  import torch
  from vllm_torchtpu.rl import from_hf as torch_from_hf, generate_manifest as torch_gen_manifest

  spec = importlib.util.spec_from_file_location("_torchtpu_rl_fakes", fakes_path)
  assert spec is not None and spec.loader is not None
  fakes_mod = importlib.util.module_from_spec(spec)
  sys.modules["_torchtpu_rl_fakes"] = fakes_mod
  spec.loader.exec_module(fakes_mod)
  return torch, fakes_mod.Qwen2ForCausalLM, fakes_mod.Qwen3_5MoeForCausalLM, torch_gen_manifest, torch_from_hf


def test_scenario1_e2e_cross_repo_maxtext_to_torchtpu_parity() -> None:
  """MaxText WeightConverter(dst_rule=vllm_torchtpu_manifest) matches TorchTPU bit-for-bit."""
  deps = _maybe_import_torchtpu()
  if deps is None:
    pytest.skip("Sibling vllm-torchtpu repo not present")
  torch, Qwen2ForCausalLM, Qwen3_5MoeForCausalLM, torch_gen_manifest, torch_from_hf = deps

  # 1. Qwen2.5 dense (tp=2, GQA + tied embeddings)
  mt_cfg, mt_state = _build_maxtext_qwen2_state(num_layers=1, scan_layers=True, tie_word_embeddings=True)
  tpu_model = Qwen2ForCausalLM(tp_size=2, tie_word_embeddings=True)
  dst_rule_json = torch_gen_manifest(tpu_model, tp_size=2)

  converter = WeightConverter(config=mt_cfg, dst_rule=dst_rule_json)
  jax_converted = converter.convert(mt_state)

  # Reference path: MaxText to_hf -> PyTorch from_hf on vllm-torchtpu
  src_rule_json = generate_maxtext_to_hf_manifest(mt_cfg, mt_state, expert_axis_grouped=False)
  jax_hf = to_hf(src_rule_json, mt_state)
  torch_hf = {
      k: torch.from_numpy(np.asarray(v.astype(jnp.float32))).to(
          torch.bfloat16 if v.dtype == jnp.bfloat16 else torch.float32
      )
      for k, v in jax_hf.items()
  }
  torch_converted = torch_from_hf(dst_rule_json, torch_hf)

  assert set(jax_converted) == set(torch_converted)
  for k, t_val in torch_converted.items():
    j_np = np.asarray(jax_converted[k].astype(jnp.float32))
    t_np = t_val.to(torch.float32).numpy()
    np.testing.assert_array_equal(j_np, t_np, err_msg=f"Qwen2 parity mismatch on {k}")

  # 2. Qwen3.5-MoE (inhomogeneous scan, tp=2, ep=2, canonical 3D expert_axis_grouped=True)
  moe_cfg, moe_state = _build_maxtext_qwen35_moe_state(scan_layers=True)
  tpu_moe = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=2, moe_tp_size=1)
  moe_dst_json = torch_gen_manifest(tpu_moe, tp_size=2, dp_size=1, ep_size=2, expert_axis_grouped=True)

  moe_converter = WeightConverter(config=moe_cfg, dst_rule=moe_dst_json)
  jax_moe_converted = moe_converter.convert(moe_state)

  moe_src_json = generate_maxtext_to_hf_manifest(moe_cfg, moe_state, expert_axis_grouped=True)
  jax_moe_hf = to_hf(moe_src_json, moe_state)
  torch_moe_hf = {
      k: torch.from_numpy(np.asarray(v.astype(jnp.float32))).to(
          torch.bfloat16 if v.dtype == jnp.bfloat16 else torch.float32
      )
      for k, v in jax_moe_hf.items()
  }
  torch_moe_converted = torch_from_hf(moe_dst_json, torch_moe_hf)

  assert set(jax_moe_converted) == set(torch_moe_converted)
  for k, t_val in torch_moe_converted.items():
    j_np = np.asarray(jax_moe_converted[k].astype(jnp.float32))
    t_np = t_val.to(torch.float32).numpy()
    np.testing.assert_array_equal(j_np, t_np, err_msg=f"Qwen3.5-MoE parity mismatch on {k}")
