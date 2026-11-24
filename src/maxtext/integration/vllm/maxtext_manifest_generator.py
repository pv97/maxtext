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
"""MaxText -> canonical HuggingFace ``weight_mapping`` manifest generator (M3).

Generates a validated 12-op ``weight_mapping`` JSON manifest (``src_rule``)
mapping flattened MaxText JAX pytree weights (``sep="."``) to canonical
HuggingFace keys for:
- Qwen2.5 dense (scanned and unscanned decoder layers)
- Qwen3.5-MoE (unscanned and inhomogeneous local/global/remainder scanned layers)

Fails closed with ``NotImplementedError`` on quantized parameters, unsupported
decoder blocks, or unclaimed parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp
from maxtext.integration.vllm.weight_mapping import (
    SCHEMA_VERSION,
    OpSpec,
    validate_manifest,
)

_ALLOWED_DTYPES = frozenset({jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)})
_SUPPORTED_DECODER_BLOCKS = frozenset({"qwen2", "qwen3", "qwen3_5", "qwen3_next"})
_SKIPPED_ROOTS = ("vision_encoder", "visual", "audio_encoder", "mtp", "mtp_block")
_LAYER_IDX_RE = re.compile(r"^layers_(\d+)\.(.+)$")
_SLOT_LAYER_RE = re.compile(r"^layers\.layer_(\d+)\.(.+)$")
_SUBPATH_ALIASES: tuple[tuple[str, str], ...] = (
    ("linear_attention.norm.rms_norm.", "attention.norm.rms_norm."),
    ("linear_attention.", "attention."),
    ("moe_block.shared_experts.", "mlp.shared_expert."),
    ("moe_block.shared_expert.", "mlp.shared_expert."),
    ("moe_block.shared_expert_gate.", "mlp.shared_expert_gate."),
    ("moe_block.", "mlp.routed_experts."),
)


def _normalize_subpath(subpath: str) -> str:
  for old, new in _SUBPATH_ALIASES:
    if subpath.startswith(old):
      return new + subpath[len(old) :]
  return subpath


def _rule(source_keys: list[str], target_keys: list[str], ops: list[OpSpec]) -> dict[str, Any]:
  return {"source_keys": source_keys, "target_keys": target_keys, "ops": ops}


def flatten_maxtext_state(state: Any) -> dict[str, Any]:
  """Flatten a MaxText pytree / NNX state into dotted ``str -> array`` dict."""
  if isinstance(state, Mapping) and all(isinstance(k, str) and "." in k for k in state):
    raw = dict(state)
  else:
    pure = state.to_pure_dict() if hasattr(state, "to_pure_dict") else state
    raw = {str(k): v for k, v in traverse_util.flatten_dict(pure, sep=".").items()}
  out: dict[str, Any] = {}
  for k, v in raw.items():
    arr = getattr(v, "value", v)
    if not (hasattr(arr, "shape") and hasattr(arr, "dtype")):
      continue
    clean_k = k[:-6] if k.endswith(".value") else k
    parts = clean_k.split(".")
    if any(p in _SKIPPED_ROOTS for p in parts[:2]):
      continue
    out[clean_k] = arr
  return out


def _decoder_block_name(config: Any) -> str:
  raw = getattr(config, "decoder_block", "")
  val = raw.value if hasattr(raw, "value") else raw
  return str(val).lower()


def _check_supported(config: Any, flat_state: Mapping[str, Any]) -> None:
  block = _decoder_block_name(config)
  if block and block not in _SUPPORTED_DECODER_BLOCKS:
    raise NotImplementedError(
        f"MaxText manifest generator does not support decoder_block={block!r} "
        f"(supported: {sorted(_SUPPORTED_DECODER_BLOCKS)})."
    )
  quant = str(getattr(config, "quantization", "") or "").lower()
  if quant not in ("", "none"):
    raise NotImplementedError(f"Quantization ({quant!r}) is not supported.")
  for k, v in flat_state.items():
    dt = jnp.dtype(v.dtype)
    if dt not in _ALLOWED_DTYPES:
      raise NotImplementedError(f"Unsupported dtype {dt} on MaxText param {k!r}.")
    leaf = k.rsplit(".", 1)[-1].lower()
    if leaf.endswith(("_scale", "_scale_inv", "_scales")):
      raise NotImplementedError(f"Quantization scale param {k!r} is not supported.")


def _kernel_to_hf_ops(shape: tuple[int, ...]) -> list[OpSpec]:
  """Convert MaxText kernel ``[in..., out...]`` to HF 2D ``[out, in]``."""
  if len(shape) == 2:
    return [{"transpose": {"axes": [1, 0]}}]
  if len(shape) == 3:
    # Q/K/V: [D, H, D_h] -> [D, H*D_h] -> [H*D_h, D]
    # Out:   [H, D_h, D] -> [H*D_h, D] -> [D, H*D_h]
    # Note: caller disambiguates Q/K/V vs Out via _attn_kernel_ops.
    return [
        {"reshape": {"src_shape": list(shape), "dst_shape": [shape[0], shape[1] * shape[2]]}},
        {"transpose": {"axes": [1, 0]}},
    ]
  raise ValueError(f"Unexpected kernel rank {len(shape)} for shape {shape}")


def _emit_single_layer_rules(
    src_key: str,
    subpath: str,
    shape: tuple[int, ...],
    layer_idx: int,
    hf_prefix: str,
    config: Any,
    *,
    expert_axis_grouped: bool,
) -> list[dict[str, Any]]:
  """Emit MaxText -> HF rule(s) for one unrolled layer parameter."""
  subpath = _normalize_subpath(subpath)
  lp = f"{hf_prefix}.layers.{layer_idx}"

  # Norms (identity)
  norm_map = {
      "pre_self_attention_layer_norm.scale": f"{lp}.input_layernorm.weight",
      "post_self_attention_layer_norm.scale": f"{lp}.post_attention_layernorm.weight",
      "input_layernorm.scale": f"{lp}.input_layernorm.weight",
      "post_attention_layernorm.scale": f"{lp}.post_attention_layernorm.weight",
      "self_attention.query_norm.scale": f"{lp}.self_attn.q_norm.weight",
      "self_attention.key_norm.scale": f"{lp}.self_attn.k_norm.weight",
      "attention.attention.query_norm.scale": f"{lp}.self_attn.q_norm.weight",
      "attention.attention.key_norm.scale": f"{lp}.self_attn.k_norm.weight",
      "attention.norm.rms_norm.scale": f"{lp}.linear_attn.norm.weight",
      "attention.A_log": f"{lp}.linear_attn.A_log",
      "attention.dt_bias": f"{lp}.linear_attn.dt_bias",
  }
  if subpath in norm_map:
    return [_rule([src_key], [norm_map[subpath]], [])]

  # Full Attention Q/K/V kernels: [D, H, D_h] -> [H*D_h, D]
  qkv_map = {
      "self_attention.query.kernel": f"{lp}.self_attn.q_proj.weight",
      "self_attention.key.kernel": f"{lp}.self_attn.k_proj.weight",
      "self_attention.value.kernel": f"{lp}.self_attn.v_proj.weight",
      "attention.attention.query.kernel": f"{lp}.self_attn.q_proj.weight",
      "attention.attention.key.kernel": f"{lp}.self_attn.k_proj.weight",
      "attention.attention.value.kernel": f"{lp}.self_attn.v_proj.weight",
  }
  if subpath in qkv_map:
    return [_rule([src_key], [qkv_map[subpath]], _kernel_to_hf_ops(shape))]

  # Full Attention Q/K/V biases: [H, D_h] -> [H*D_h]
  qkv_bias_map = {
      "self_attention.query.bias": f"{lp}.self_attn.q_proj.bias",
      "self_attention.key.bias": f"{lp}.self_attn.k_proj.bias",
      "self_attention.value.bias": f"{lp}.self_attn.v_proj.bias",
  }
  if subpath in qkv_bias_map:
    ops: list[OpSpec] = (
        [{"reshape": {"src_shape": list(shape), "dst_shape": [shape[0] * shape[1]]}}]
        if len(shape) == 2
        else []
    )
    return [_rule([src_key], [qkv_bias_map[subpath]], ops)]

  # Full Attention Out kernel: [H, D_h, D] -> [H*D_h, D] -> [D, H*D_h]
  if subpath in ("self_attention.out.kernel", "attention.attention.out.kernel"):
    ops = (
        [
            {"reshape": {"src_shape": list(shape), "dst_shape": [shape[0] * shape[1], shape[2]]}},
            {"transpose": {"axes": [1, 0]}},
        ]
        if len(shape) == 3
        else [{"transpose": {"axes": [1, 0]}}]
    )
    return [_rule([src_key], [f"{lp}.self_attn.o_proj.weight"], ops)]

  # GDN conv1d: [K, 1, C] -> [C, 1, K]
  if subpath == "attention.conv1d.kernel":
    return [_rule([src_key], [f"{lp}.linear_attn.conv1d.weight"], [{"transpose": {"axes": [2, 1, 0]}}])]

  # GDN in_proj_ba: [D, H_k * 2 * V_per_K] -> (in_proj_b [H_v, D], in_proj_a [H_v, D])
  if subpath == "attention.in_proj_ba.kernel":
    h_k = int(
        getattr(
            config,
            "gdn_num_key_heads",
            getattr(config, "gdn_conv_num_key_heads", getattr(config, "linear_num_key_heads", 0)),
        )
    )
    h_v = int(
        getattr(
            config,
            "gdn_num_value_heads",
            getattr(config, "gdn_conv_num_value_heads", getattr(config, "linear_num_value_heads", 0)),
        )
    )
    v_per_k = h_v // h_k
    d_in = shape[0]
    ops = [
        {"transpose": {"axes": [1, 0]}},
        {"reshape": {"src_shape": [h_k * 2 * v_per_k, d_in], "dst_shape": [h_k, 2, v_per_k, d_in]}},
        {"transpose": {"axes": [1, 0, 2, 3]}},
        {"reshape": {"src_shape": [2, h_k, v_per_k, d_in], "dst_shape": [2, h_v, d_in]}},
        {"unstack": {"axis": 0}},
    ]
    return [
        _rule(
            [src_key],
            [f"{lp}.linear_attn.in_proj_b.weight", f"{lp}.linear_attn.in_proj_a.weight"],
            ops,
        )
    ]

  # GDN in_proj_qkvz: [D, H_k * (2*D_k + 2*V_per_K*D_v)] -> (in_proj_qkv, in_proj_z)
  if subpath == "attention.in_proj_qkvz.kernel":
    h_k = int(
        getattr(
            config,
            "gdn_num_key_heads",
            getattr(config, "gdn_conv_num_key_heads", getattr(config, "linear_num_key_heads", 0)),
        )
    )
    h_v = int(
        getattr(
            config,
            "gdn_num_value_heads",
            getattr(config, "gdn_conv_num_value_heads", getattr(config, "linear_num_value_heads", 0)),
        )
    )
    d_k = int(getattr(config, "gdn_key_head_dim", getattr(config, "linear_key_head_dim", 0)))
    d_v = int(getattr(config, "gdn_value_head_dim", getattr(config, "linear_value_head_dim", 0)))
    v_per_k = h_v // h_k
    d_in = shape[0]
    blk = 2 * d_k + 2 * v_per_k * d_v
    q_t, k_t, v_t, z_t = (f"__tmp__:{lp}:qkvz_{c}" for c in ("q", "k", "v", "z"))
    q2, k2, v2 = (f"__tmp__:{lp}:qkvz_{c}2" for c in ("q", "k", "v"))
    return [
        _rule(
            [src_key],
            [q_t, k_t, v_t, z_t],
            [
                {"transpose": {"axes": [1, 0]}},
                {"reshape": {"src_shape": [h_k * blk, d_in], "dst_shape": [h_k, blk, d_in]}},
                {"split": {"axis": 1, "sizes": [d_k, d_k, v_per_k * d_v, v_per_k * d_v]}},
            ],
        ),
        _rule([q_t], [q2], [{"reshape": {"src_shape": [h_k, d_k, d_in], "dst_shape": [h_k * d_k, d_in]}}]),
        _rule([k_t], [k2], [{"reshape": {"src_shape": [h_k, d_k, d_in], "dst_shape": [h_k * d_k, d_in]}}]),
        _rule([v_t], [v2], [{"reshape": {"src_shape": [h_k, v_per_k * d_v, d_in], "dst_shape": [h_v * d_v, d_in]}}]),
        _rule(
            [q2, k2, v2],
            [f"{lp}.linear_attn.in_proj_qkv.weight"],
            [{"concat": {"axis": 0, "sizes": [h_k * d_k, h_k * d_k, h_v * d_v]}}],
        ),
        _rule(
            [z_t],
            [f"{lp}.linear_attn.in_proj_z.weight"],
            [{"reshape": {"src_shape": [h_k, v_per_k * d_v, d_in], "dst_shape": [h_v * d_v, d_in]}}],
        ),
    ]

  # 2D linear projections (Dense MLP, GDN out_proj, MoE gate, shared_expert)
  linear_2d_map = {
      "attention.out_proj.kernel": f"{lp}.linear_attn.out_proj.weight",
      "mlp.wi_0.kernel": f"{lp}.mlp.gate_proj.weight",
      "mlp.wi_1.kernel": f"{lp}.mlp.up_proj.weight",
      "mlp.wo.kernel": f"{lp}.mlp.down_proj.weight",
      "mlp.routed_experts.gate.kernel": f"{lp}.mlp.gate.weight",
      "moe_block.gate.kernel": f"{lp}.mlp.gate.weight",
      "mlp.shared_expert_gate.kernel": f"{lp}.mlp.shared_expert_gate.weight",
      "mlp.shared_expert.wi_0.kernel": f"{lp}.mlp.shared_expert.gate_proj.weight",
      "mlp.shared_expert.wi_1.kernel": f"{lp}.mlp.shared_expert.up_proj.weight",
      "mlp.shared_expert.wo.kernel": f"{lp}.mlp.shared_expert.down_proj.weight",
  }
  if subpath in linear_2d_map:
    return [_rule([src_key], [linear_2d_map[subpath]], [{"transpose": {"axes": [1, 0]}}])]

  # Pre-fused Routed MoE gate+up: [E, D, 2*F] -> (gate_proj [E, F, D], up_proj [E, F, D])
  if subpath in ("mlp.routed_experts.wi", "moe_block.wi"):
    num_e, _, double_f = shape
    half_f = double_f // 2
    if expert_axis_grouped:
      return [
          _rule(
              [src_key],
              [f"{lp}.mlp.experts.gate_proj.weight", f"{lp}.mlp.experts.up_proj.weight"],
              [
                  {"transpose": {"axes": [0, 2, 1]}},
                  {"split": {"axis": 1, "sizes": [half_f, half_f]}},
              ],
          )
      ]
    g_tmp, u_tmp = f"__tmp__:{lp}:wi_g", f"__tmp__:{lp}:wi_u"
    return [
        _rule(
            [src_key],
            [g_tmp, u_tmp],
            [
                {"transpose": {"axes": [0, 2, 1]}},
                {"split": {"axis": 1, "sizes": [half_f, half_f]}},
            ],
        ),
        _rule([g_tmp], [f"{lp}.mlp.experts.{e}.gate_proj.weight" for e in range(num_e)], [{"unstack": {"axis": 0}}]),
        _rule([u_tmp], [f"{lp}.mlp.experts.{e}.up_proj.weight" for e in range(num_e)], [{"unstack": {"axis": 0}}]),
    ]

  # Routed MoE experts: [E, in, out] -> [E, out, in] (3D grouped) or E x [out, in] (2D)
  expert_map = {
      "mlp.routed_experts.wi_0": "gate_proj.weight",
      "mlp.routed_experts.wi_1": "up_proj.weight",
      "mlp.routed_experts.wo": "down_proj.weight",
      "moe_block.wi_0": "gate_proj.weight",
      "moe_block.wi_1": "up_proj.weight",
      "moe_block.wo": "down_proj.weight",
  }
  if subpath in expert_map:
    proj_suffix = expert_map[subpath]
    num_e = shape[0]
    if expert_axis_grouped:
      return [_rule([src_key], [f"{lp}.mlp.experts.{proj_suffix}"], [{"transpose": {"axes": [0, 2, 1]}}])]
    return [
        _rule(
            [src_key],
            [f"{lp}.mlp.experts.{e}.{proj_suffix}" for e in range(num_e)],
            [{"transpose": {"axes": [0, 2, 1]}}, {"unstack": {"axis": 0}}],
        )
    ]

  raise NotImplementedError(f"Unsupported MaxText layer subpath {subpath!r} (key={src_key!r}).")


def generate_maxtext_to_hf_manifest_dict(
    config: Any,
    state: Any,
    *,
    hf_prefix: str = "model",
    hf_vocab_size: int | None = None,
    expert_axis_grouped: bool = True,
) -> dict[str, Any]:
  """Generate a validated MaxText -> HF ``weight_mapping`` manifest dict (M3)."""
  flat_state = flatten_maxtext_state(state)
  _check_supported(config, flat_state)

  # Detect optional "params." root prefix.
  has_params_root = any(k.startswith("params.") for k in flat_state)
  root = "params." if has_params_root else ""
  scan_axis = int(getattr(config, "param_scan_axis", 0))
  num_layers = int(getattr(config, "num_decoder_layers", 0))
  cycle = int(getattr(config, "inhomogeneous_layer_cycle_interval", 1) or 1)

  rules: list[dict[str, Any]] = []
  has_moe = False
  claimed: set[str] = set()

  def _unroll_and_emit(
      src_key: str,
      subpath: str,
      shape: tuple[int, ...],
      layer_indices: list[int],
      unroll_axes: list[int],
  ) -> None:
    nonlocal has_moe
    if "routed_experts" in subpath or "moe_block" in subpath:
      has_moe = True
    cur_keys = [src_key]
    cur_shape = list(shape)
    if len(unroll_axes) == 2:
      # Inhomogeneous local_layers: flatten (B, C-1) scan axes into B*(C-1), then unstack.
      ax0, ax1 = unroll_axes
      if ax1 != ax0 + 1:
        perm = [i for i in range(len(cur_shape)) if i not in (ax0, ax1)]
        perm = perm[:ax0] + [ax0, ax1] + perm[ax0:]
        pre_ops: list[OpSpec] = [{"transpose": {"axes": perm}}]
        cur_shape = [cur_shape[i] for i in perm]
      else:
        pre_ops = []
      n_total = cur_shape[ax0] * cur_shape[ax0 + 1]
      flat_shape = cur_shape[:ax0] + [n_total] + cur_shape[ax0 + 2 :]
      pre_ops.extend(
          [
              {"reshape": {"src_shape": list(cur_shape), "dst_shape": flat_shape}},
              {"unstack": {"axis": ax0}},
          ]
      )
      per_layer_shape = tuple(flat_shape[:ax0] + flat_shape[ax0 + 1 :])
      tmp_keys = [f"__tmp__:{src_key}:l{li}" for li in layer_indices]
      rules.append(_rule([src_key], tmp_keys, pre_ops))
      cur_keys = tmp_keys
    elif len(unroll_axes) == 1:
      ax = unroll_axes[0]
      per_layer_shape = tuple(cur_shape[:ax] + cur_shape[ax + 1 :])
      tmp_keys = [f"__tmp__:{src_key}:l{li}" for li in layer_indices]
      rules.append(_rule([src_key], tmp_keys, [{"unstack": {"axis": ax}}]))
      cur_keys = tmp_keys
    else:
      per_layer_shape = shape

    for tk, li in zip(cur_keys, layer_indices, strict=True):
      rules.extend(
          _emit_single_layer_rules(
              tk, subpath, per_layer_shape, li, hf_prefix, config, expert_axis_grouped=expert_axis_grouped
          )
      )

  for key, val in flat_state.items():
    rel = key
    while rel.startswith(("params.", "base.")):
      rel = rel.split(".", 1)[1]
    shape = tuple(int(d) for d in val.shape)

    if rel == "token_embedder.embedding":
      claimed.add(key)
      ops: list[OpSpec] = []
      if hf_vocab_size is not None and shape[0] > hf_vocab_size:
        ops.append({"slice": {"axis": 0, "sizes": [0, shape[0] - hf_vocab_size]}})
      rules.append(_rule([key], [f"{hf_prefix}.embed_tokens.weight"], ops))
      continue

    if rel == "decoder.decoder_norm.scale":
      claimed.add(key)
      rules.append(_rule([key], [f"{hf_prefix}.norm.weight"], []))
      continue

    if rel == "decoder.logits_dense.kernel":
      claimed.add(key)
      ops = [{"transpose": {"axes": [1, 0]}}]
      if hf_vocab_size is not None and shape[1] > hf_vocab_size:
        ops.append({"slice": {"axis": 0, "sizes": [0, shape[1] - hf_vocab_size]}})
      rules.append(_rule([key], ["lm_head.weight"], ops))
      continue

    if not rel.startswith("decoder."):
      continue
    dec_rel = rel[len("decoder.") :]

    # 1. Unscanned layer: decoder.layers_{i}.<subpath>
    m = _LAYER_IDX_RE.match(dec_rel)
    if m:
      claimed.add(key)
      _unroll_and_emit(key, m.group(2), shape, [int(m.group(1))], [])
      continue

    # 2. Inhomogeneous scanned Qwen3.5: local_layers / global_layer / layers_remainder
    num_blocks = num_layers // cycle if cycle > 1 else 0
    if dec_rel.startswith("layers.local_layers."):
      claimed.add(key)
      sub = dec_rel[len("layers.local_layers.") :]
      is_exp = _normalize_subpath(sub).startswith(("mlp.routed_experts.wi", "mlp.routed_experts.wo"))
      ax0 = 1 if is_exp else scan_axis
      indices = [b * cycle + p for b in range(num_blocks) for p in range(cycle - 1)]
      _unroll_and_emit(key, sub, shape, indices, [ax0, ax0 + 1])
      continue

    if dec_rel.startswith("layers.global_layer."):
      claimed.add(key)
      sub = dec_rel[len("layers.global_layer.") :]
      is_exp = _normalize_subpath(sub).startswith(("mlp.routed_experts.wi", "mlp.routed_experts.wo"))
      ax = 1 if is_exp else scan_axis
      indices = [b * cycle + (cycle - 1) for b in range(num_blocks)]
      _unroll_and_emit(key, sub, shape, indices, [ax])
      continue

    if dec_rel.startswith("layers_remainder.local_layers."):
      claimed.add(key)
      sub = dec_rel[len("layers_remainder.local_layers.") :]
      is_exp = _normalize_subpath(sub).startswith(("mlp.routed_experts.wi", "mlp.routed_experts.wo"))
      ax = 1 if is_exp else scan_axis
      rem_start = num_blocks * cycle
      indices = list(range(rem_start, num_layers))
      _unroll_and_emit(key, sub, shape, indices, [ax])
      continue

    # 2b. Slot-scanned inhomogeneous layers: decoder.layers.layer_{slot}.<subpath>
    m_slot = _SLOT_LAYER_RE.match(dec_rel)
    if m_slot:
      claimed.add(key)
      slot = int(m_slot.group(1))
      sub = m_slot.group(2)
      is_exp = _normalize_subpath(sub).startswith(("mlp.routed_experts.wi", "mlp.routed_experts.wo"))
      ax = 1 if is_exp else scan_axis
      indices = list(range(slot, num_layers, cycle))
      _unroll_and_emit(key, sub, shape, indices, [ax] if len(indices) > 1 or shape[ax] == len(indices) else [])
      continue

    # 3. Homogeneous scanned layers: decoder.layers.<subpath>
    if dec_rel.startswith("layers."):
      claimed.add(key)
      sub = dec_rel[len("layers.") :]
      is_exp = _normalize_subpath(sub).startswith(("mlp.routed_experts.wi", "mlp.routed_experts.wo"))
      ax = 1 if is_exp else scan_axis
      _unroll_and_emit(key, sub, shape, list(range(num_layers)), [ax])
      continue

  unclaimed = sorted(set(flat_state.keys()) - claimed)
  if unclaimed:
    raise NotImplementedError(f"Unclaimed MaxText parameter(s): {unclaimed}")

  return validate_manifest(
      {
          "schema_version": SCHEMA_VERSION,
          "expert_axis_grouped": has_moe and expert_axis_grouped,
          "stripes": {},
          "tied_keys": [],
          "dropped_keys": [],
          "rules": rules,
      }
  )


def generate_maxtext_to_hf_manifest(
    config: Any,
    state: Any,
    *,
    hf_prefix: str = "model",
    hf_vocab_size: int | None = None,
    expert_axis_grouped: bool = True,
) -> str:
  """Generate a serialized JSON MaxText -> HF ``weight_mapping`` manifest (M3)."""
  return json.dumps(
      generate_maxtext_to_hf_manifest_dict(
          config,
          state,
          hf_prefix=hf_prefix,
          hf_vocab_size=hf_vocab_size,
          expert_axis_grouped=expert_axis_grouped,
      )
  )
