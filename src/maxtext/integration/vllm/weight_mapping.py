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
"""Pure-JAX 12-op executor, validator, and inverter for ``weight_mapping`` (M1).

Implements the Distributed RL Weight Conversion Protocol in JAX (JIT-traceable
and ``jax.eval_shape``-compatible) with Option 1 rule chaining (``__tmp__:*``
intermediates) so MaxText can execute both its own ``to_hf`` manifests and
serving-engine ``from_hf`` manifests (e.g. ``vllm-torchtpu``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import json
import math
from typing import Any, TypeGuard

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax.numpy as jnp

SCHEMA_VERSION = "weight_mapping"


class OpName(str, Enum):
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  SPLIT = "split"
  CONCAT = "concat"
  STACK = "stack"
  UNSTACK = "unstack"
  SLICE = "slice"
  UNPAD = "unpad"
  PAD = "pad"
  REPEAT = "repeat"
  MEAN_POOL = "mean_pool"
  UNREPEAT = "unrepeat"
  SCALE = "scale"
  CAST = "cast"


OpSpec = dict[str, dict[str, Any]]
RuleSpec = str | Mapping[str, Any]

_DTYPE_MAP: dict[str, Any] = {
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float16": jnp.float16,
    "fp16": jnp.float16,
    "float32": jnp.float32,
    "fp32": jnp.float32,
    "float64": jnp.float64,
    "fp64": jnp.float64,
    "int32": jnp.int32,
    "int64": jnp.int64,
    "int8": jnp.int8,
    "uint8": jnp.uint8,
    "bool": jnp.bool_,
}
for _fp8_name in ("float8_e4m3fn", "float8_e5m2"):
  if hasattr(jnp, _fp8_name):
    _DTYPE_MAP[_fp8_name] = getattr(jnp, _fp8_name)

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "expert_axis_grouped",
        "stripes",
        "tied_keys",
        "dropped_keys",
        "rules",
    }
)
_RULE_KEYS = frozenset({"source_keys", "target_keys", "ops", "sharding"})


def _is_int(x: Any) -> bool:
  return isinstance(x, int) and not isinstance(x, bool)


def _is_seq(x: Any) -> TypeGuard[Sequence[Any]]:
  return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def _check_keys(obj: Mapping[str, Any], expected: set[str], ctx: str) -> None:
  keys = set(obj.keys())
  if keys != expected:
    raise ValueError(f"{ctx}: expected keys {sorted(expected)}, got {sorted(keys)}")


def _check_pos_ints(seq: Any, ctx: str) -> list[int]:
  if not _is_seq(seq) or len(seq) == 0 or not all(_is_int(v) and v > 0 for v in seq):
    raise ValueError(f"{ctx}: expected non-empty sequence of positive ints")
  return [int(v) for v in seq]


def _validate_op(op: Any, ctx: str) -> None:
  if not isinstance(op, Mapping) or len(op) != 1:
    raise ValueError(f"{ctx}: each op must be a single-key mapping {{op: kwargs}}")
  op_name, kw = next(iter(op.items()))
  if not isinstance(kw, Mapping):
    raise ValueError(f"{ctx}: kwargs for {op_name!r} must be a mapping")

  if op_name == OpName.RESHAPE:
    _check_keys(kw, {"src_shape", "dst_shape"}, f"{ctx}.reshape")
    for field in ("src_shape", "dst_shape"):
      shape = kw[field]
      if not _is_seq(shape) or not all(_is_int(v) and v > 0 for v in shape):
        raise ValueError(f"{ctx}.reshape.{field}: expected positive int dims")
    if math.prod(kw["src_shape"]) != math.prod(kw["dst_shape"]):
      raise ValueError(f"{ctx}.reshape: src_shape and dst_shape element counts differ")
  elif op_name == OpName.TRANSPOSE:
    _check_keys(kw, {"axes"}, f"{ctx}.transpose")
    axes = kw["axes"]
    if not _is_seq(axes) or not all(_is_int(a) for a in axes):
      raise ValueError(f"{ctx}.transpose.axes: expected int sequence")
    if sorted(int(a) for a in axes) != list(range(len(axes))):
      raise ValueError(f"{ctx}.transpose.axes: must be a permutation of 0..N-1")
  elif op_name in (OpName.SPLIT, OpName.CONCAT):
    _check_keys(kw, {"axis", "sizes"}, f"{ctx}.{op_name}")
    if not _is_int(kw["axis"]):
      raise ValueError(f"{ctx}.{op_name}.axis: expected int")
    _check_pos_ints(kw["sizes"], f"{ctx}.{op_name}.sizes")
  elif op_name in (OpName.STACK, OpName.UNSTACK):
    _check_keys(kw, {"axis"}, f"{ctx}.{op_name}")
    if not _is_int(kw["axis"]):
      raise ValueError(f"{ctx}.{op_name}.axis: expected int")
  elif op_name in (OpName.SLICE, OpName.UNPAD, OpName.PAD):
    _check_keys(kw, {"axis", "sizes"}, f"{ctx}.{op_name}")
    if not _is_int(kw["axis"]):
      raise ValueError(f"{ctx}.{op_name}.axis: expected int")
    sizes = kw["sizes"]
    if not _is_seq(sizes) or len(sizes) != 2 or not all(_is_int(v) and v >= 0 for v in sizes):
      raise ValueError(f"{ctx}.{op_name}.sizes: expected [left, right] >= 0")
  elif op_name in (OpName.REPEAT, OpName.MEAN_POOL, OpName.UNREPEAT):
    _check_keys(kw, {"axis", "times"}, f"{ctx}.{op_name}")
    if not _is_int(kw["axis"]) or not (_is_int(kw["times"]) and kw["times"] > 0):
      raise ValueError(f"{ctx}.{op_name}: expected int axis and times > 0")
  elif op_name == OpName.SCALE:
    _check_keys(kw, {"factor"}, f"{ctx}.scale")
    f = kw["factor"]
    if not isinstance(f, (int, float)) or isinstance(f, bool) or not math.isfinite(f) or f == 0:
      raise ValueError(f"{ctx}.scale.factor: expected finite non-zero number")
  elif op_name == OpName.CAST:
    _check_keys(kw, {"src_dtype", "dst_dtype"}, f"{ctx}.cast")
    for field in ("src_dtype", "dst_dtype"):
      if not isinstance(kw[field], str) or kw[field] not in _DTYPE_MAP:
        raise ValueError(f"{ctx}.cast.{field}: unknown dtype {kw[field]!r}")
  else:
    raise ValueError(f"{ctx}: unknown op {op_name!r}")


def validate_manifest(manifest: RuleSpec) -> dict[str, Any]:
  """Validate a ``weight_mapping`` manifest and return its normalized dict."""
  if isinstance(manifest, str):
    try:
      data = json.loads(manifest)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid manifest JSON: {exc}") from exc
  elif isinstance(manifest, Mapping):
    data = dict(manifest)
  else:
    raise TypeError(f"Expected str or Mapping, got {type(manifest).__name__}")

  if not isinstance(data, dict):
    raise ValueError("Manifest JSON must decode to an object")
  extra = set(data.keys()) - _TOP_KEYS
  if extra:
    raise ValueError(f"Unknown top-level manifest keys: {sorted(extra)}")
  if data.get("schema_version") != SCHEMA_VERSION:
    raise ValueError(
        f"schema_version must be {SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
    )

  eag = data.get("expert_axis_grouped", False)
  if not isinstance(eag, bool):
    raise ValueError("expert_axis_grouped must be a bool")

  stripes_raw = data.get("stripes", {})
  if not isinstance(stripes_raw, Mapping):
    raise ValueError("stripes must be a mapping")
  stripes: dict[str, int] = {}
  for k, v in stripes_raw.items():
    if not isinstance(k, str) or not (_is_int(v) and v > 0):
      raise ValueError(f"stripes[{k!r}] must be a positive int, got {v!r}")
    stripes[k] = int(v)

  tied_raw = data.get("tied_keys", [])
  if not _is_seq(tied_raw):
    raise ValueError("tied_keys must be a sequence")
  tied_keys: list[list[str]] = []
  for pair in tied_raw:
    if not _is_seq(pair) or len(pair) != 2 or not all(isinstance(s, str) for s in pair):
      raise ValueError(f"tied_keys entries must be [src, dst] pairs, got {pair!r}")
    tied_keys.append([pair[0], pair[1]])

  dropped_raw = data.get("dropped_keys", [])
  if not _is_seq(dropped_raw) or not all(isinstance(s, str) for s in dropped_raw):
    raise ValueError("dropped_keys must be a sequence of strings")

  rules_raw = data.get("rules")
  if not _is_seq(rules_raw):
    raise ValueError("rules is required and must be a sequence")
  rules: list[dict[str, Any]] = []
  for idx, rule in enumerate(rules_raw):
    ctx = f"rules[{idx}]"
    if not isinstance(rule, Mapping):
      raise ValueError(f"{ctx}: must be a mapping")
    extra_r = set(rule.keys()) - _RULE_KEYS
    if extra_r:
      raise ValueError(f"{ctx}: unknown keys {sorted(extra_r)}")
    for req in ("source_keys", "target_keys"):
      vals = rule.get(req)
      if not _is_seq(vals) or not vals or not all(isinstance(s, str) for s in vals):
        raise ValueError(f"{ctx}.{req}: must be a non-empty sequence of strings")
    if "ops" not in rule or not _is_seq(rule["ops"]):
      raise ValueError(f"{ctx}.ops: is required and must be a sequence")
    for op_i, op in enumerate(rule["ops"]):
      _validate_op(op, f"{ctx}.ops[{op_i}]")
    entry: dict[str, Any] = {
        "source_keys": list(rule["source_keys"]),
        "target_keys": list(rule["target_keys"]),
        "ops": [dict(op) for op in rule["ops"]],
    }
    if "sharding" in rule:
      sh = rule["sharding"]
      if not _is_seq(sh) or not all(a is None or isinstance(a, str) for a in sh):
        raise ValueError(f"{ctx}.sharding: must be a sequence of str | None")
      entry["sharding"] = list(sh)
    rules.append(entry)

  return {
      "schema_version": SCHEMA_VERSION,
      "expert_axis_grouped": eag,
      "stripes": stripes,
      "tied_keys": tied_keys,
      "dropped_keys": list(dropped_raw),
      "rules": rules,
  }


def _norm_axis(axis: int, ndim: int, op: str) -> int:
  norm = axis + ndim if axis < 0 else axis
  if not (0 <= norm < ndim):
    raise ValueError(f"{op}: axis {axis} out of range for ndim={ndim}")
  return norm


def _apply_unary(op_name: str, kw: Mapping[str, Any], t: jax.Array) -> jax.Array:
  if op_name == OpName.RESHAPE:
    if list(t.shape) != list(kw["src_shape"]):
      raise ValueError(f"reshape: shape {list(t.shape)} != {list(kw['src_shape'])}")
    return jnp.reshape(t, tuple(kw["dst_shape"]))
  if op_name == OpName.TRANSPOSE:
    axes = list(kw["axes"])
    if len(axes) != t.ndim:
      raise ValueError(f"transpose: len(axes)={len(axes)} != ndim={t.ndim}")
    return jnp.transpose(t, axes)
  if op_name in (OpName.SLICE, OpName.UNPAD):
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    left, right = kw["sizes"]
    limit = int(t.shape[axis]) - right
    if limit <= left:
      raise ValueError(f"{op_name}: invalid [{left}, {right}] for dim={t.shape[axis]}")
    return jax.lax.slice_in_dim(t, left, limit, axis=axis)
  if op_name == OpName.PAD:
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    left, right = kw["sizes"]
    if left == 0 and right == 0:
      return t
    pad_width = [(0, 0) if i != axis else (left, right) for i in range(t.ndim)]
    return jnp.pad(t, pad_width)
  if op_name == OpName.REPEAT:
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    return jnp.repeat(t, int(kw["times"]), axis=axis)
  if op_name in (OpName.MEAN_POOL, OpName.UNREPEAT):
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    times = int(kw["times"])
    if int(t.shape[axis]) % times != 0:
      raise ValueError(f"{op_name}: dim={t.shape[axis]} not divisible by {times}")
    if op_name == OpName.UNREPEAT:
      return jax.lax.slice_in_dim(t, 0, int(t.shape[axis]), stride=times, axis=axis)
    new_shape = [*t.shape[:axis], int(t.shape[axis]) // times, times, *t.shape[axis + 1 :]]
    compute_dt = jnp.float64 if t.dtype == jnp.float64 else jnp.float32
    return jnp.mean(jnp.reshape(t, new_shape).astype(compute_dt), axis=axis + 1).astype(t.dtype)
  if op_name == OpName.SCALE:
    compute_dt = jnp.float64 if t.dtype == jnp.float64 else jnp.float32
    return (t.astype(compute_dt) * float(kw["factor"])).astype(t.dtype)
  if op_name == OpName.CAST:
    src_dt, dst_dt = _DTYPE_MAP[kw["src_dtype"]], _DTYPE_MAP[kw["dst_dtype"]]
    if t.dtype != jnp.dtype(src_dt):
      raise ValueError(f"cast: array dtype {t.dtype} != src_dtype {src_dt}")
    return t.astype(dst_dt)
  raise ValueError(f"Unknown op {op_name!r}")


def _apply_op(op_name: str, kw: Mapping[str, Any], tensors: list[jax.Array]) -> list[jax.Array]:
  if op_name == OpName.CONCAT:
    axis = _norm_axis(kw["axis"], tensors[0].ndim, op_name)
    if [int(t.shape[axis]) for t in tensors] != list(kw["sizes"]):
      raise ValueError(
          f"concat: array sizes {[int(t.shape[axis]) for t in tensors]} != {list(kw['sizes'])}"
      )
    return [jnp.concatenate(tensors, axis=axis)]
  if op_name == OpName.STACK:
    axis = _norm_axis(kw["axis"], tensors[0].ndim + 1, op_name)
    return [jnp.stack(tensors, axis=axis)]
  if op_name == OpName.SPLIT:
    if len(tensors) != 1:
      raise ValueError(f"split: expected 1 array, got {len(tensors)}")
    t = tensors[0]
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    sizes = [int(s) for s in kw["sizes"]]
    if sum(sizes) != int(t.shape[axis]):
      raise ValueError(f"split: sum({sizes}) != shape[{axis}]={t.shape[axis]}")
    offsets = []
    acc = 0
    for s in sizes[:-1]:
      acc += s
      offsets.append(acc)
    return list(jnp.split(t, offsets, axis=axis))
  if op_name == OpName.UNSTACK:
    if len(tensors) != 1:
      raise ValueError(f"unstack: expected 1 array, got {len(tensors)}")
    t = tensors[0]
    axis = _norm_axis(kw["axis"], t.ndim, op_name)
    return list(jnp.unstack(t, axis=axis))
  return [_apply_unary(op_name, kw, t) for t in tensors]


def _invert_op(op: Mapping[str, Any]) -> OpSpec:
  op_name, kw = next(iter(op.items()))
  if op_name == OpName.RESHAPE:
    return {"reshape": {"src_shape": list(kw["dst_shape"]), "dst_shape": list(kw["src_shape"])}}
  if op_name == OpName.TRANSPOSE:
    axes = list(kw["axes"])
    inv = [0] * len(axes)
    for i, a in enumerate(axes):
      inv[a] = i
    return {"transpose": {"axes": inv}}
  if op_name == OpName.CONCAT:
    return {"split": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
  if op_name == OpName.SPLIT:
    return {"concat": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
  if op_name == OpName.STACK:
    return {"unstack": {"axis": kw["axis"]}}
  if op_name == OpName.UNSTACK:
    return {"stack": {"axis": kw["axis"]}}
  if op_name == OpName.PAD:
    return {"slice": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
  if op_name in (OpName.SLICE, OpName.UNPAD):
    return {"pad": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
  if op_name == OpName.REPEAT:
    return {"mean_pool": {"axis": kw["axis"], "times": kw["times"]}}
  if op_name in (OpName.MEAN_POOL, OpName.UNREPEAT):
    return {"repeat": {"axis": kw["axis"], "times": kw["times"]}}
  if op_name == OpName.SCALE:
    return {"scale": {"factor": 1.0 / float(kw["factor"])}}
  if op_name == OpName.CAST:
    return {"cast": {"src_dtype": kw["dst_dtype"], "dst_dtype": kw["src_dtype"]}}
  raise ValueError(f"Unknown op {op_name!r}")


def invert_manifest(manifest: RuleSpec) -> dict[str, Any]:
  """Invert a ``weight_mapping`` manifest (reversing chained rules and ops)."""
  spec = validate_manifest(manifest)
  inv_rules = [
      {
          "source_keys": list(r["target_keys"]),
          "target_keys": list(r["source_keys"]),
          "ops": [_invert_op(op) for op in reversed(r["ops"])],
      }
      for r in reversed(spec["rules"])
  ]
  return {
      "schema_version": SCHEMA_VERSION,
      "expert_axis_grouped": spec["expert_axis_grouped"],
      "stripes": dict(spec["stripes"]),
      "tied_keys": [],
      "dropped_keys": list(spec["dropped_keys"]),
      "rules": inv_rules,
  }


def execute_manifest(
    manifest: RuleSpec,
    src_weights: Mapping[str, jax.Array],
    mesh: Mesh | None = None,
) -> dict[str, jax.Array]:
  """Execute a ``weight_mapping`` manifest over JAX arrays (Option 1 chaining)."""
  spec = validate_manifest(manifest)
  dropped = set(spec["dropped_keys"])
  out: dict[str, jax.Array] = {}
  intermediates: set[str] = set()

  for idx, rule in enumerate(spec["rules"]):
    if any(k in dropped for k in rule["source_keys"]):
      for k in rule["source_keys"]:
        if k not in src_weights and k in out:
          intermediates.add(k)
      dropped.update(rule["target_keys"])
      continue

    tensors: list[jax.Array] = []
    for k in rule["source_keys"]:
      if k in src_weights:
        tensors.append(src_weights[k])
      elif k in out:
        tensors.append(out[k])
        intermediates.add(k)
      else:
        raise KeyError(f"rules[{idx}]: missing source weight key {k!r}")

    for op in rule["ops"]:
      op_name, kw = next(iter(op.items()))
      tensors = _apply_op(op_name, kw, tensors)

    if len(tensors) != len(rule["target_keys"]):
      raise ValueError(
          f"rules[{idx}]: produced {len(tensors)} array(s) for "
          f"{len(rule['target_keys'])} target key(s)"
      )
    sharding = rule.get("sharding")
    for tk, arr in zip(rule["target_keys"], tensors, strict=True):
      if mesh is not None and sharding is not None:
        arr = jax.lax.with_sharding_constraint(
            arr, NamedSharding(mesh, PartitionSpec(*sharding))
        )
      out[tk] = arr

  for k in intermediates:
    out.pop(k, None)
  for k in [k for k in out if k.startswith("__tmp__:")]:
    out.pop(k, None)

  for src_k, dst_k in spec["tied_keys"]:
    if src_k in out and dst_k not in out and dst_k not in dropped:
      out[dst_k] = out[src_k]
  return out


def to_hf_impl(
    src_rule: RuleSpec,
    src_weights: Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
  """JIT-traceable MaxText -> HF conversion using a ``weight_mapping`` rule."""
  return execute_manifest(src_rule, src_weights, mesh=None)


def to_hf(
    src_rule: str,
    src_weights: Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
  """Convert MaxText weights to HF format using a serialized JSON ``src_rule``."""
  if not isinstance(src_rule, str):
    raise TypeError(f"to_hf expects src_rule to be str, got {type(src_rule).__name__}")
  return to_hf_impl(src_rule, src_weights)


def from_hf_impl(
    dst_rule: RuleSpec,
    hf_weights: Mapping[str, jax.Array],
    mesh: Mesh | None = None,
) -> dict[str, jax.Array]:
  """JIT-traceable HF -> serving conversion using a ``weight_mapping`` rule."""
  return execute_manifest(dst_rule, hf_weights, mesh=mesh)


def from_hf(
    dst_rule: str,
    hf_weights: Mapping[str, jax.Array],
    mesh: Mesh | None = None,
) -> dict[str, jax.Array]:
  """Convert HF weights to serving format using a serialized JSON ``dst_rule``."""
  if not isinstance(dst_rule, str):
    raise TypeError(f"from_hf expects dst_rule to be str, got {type(dst_rule).__name__}")
  return from_hf_impl(dst_rule, hf_weights, mesh=mesh)
