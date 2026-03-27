from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from fla.models.utils import Cache


LayerState = Dict[str, Any]
AttnState = Tuple[torch.Tensor, torch.Tensor]


class LaCTCache(Cache):
    """
    Minimal cache for LaCT generation.

    Each layer stores:
    - ``attn_state``: flattened attention KV cache with shape [batch, seq, hidden]
    - ``lact_state``: TTT decode state
    """

    is_compileable = False

    def __init__(self, seen_tokens: int = 0) -> None:
        super().__init__()
        self.states: List[LayerState] = []
        self._seen_tokens = int(seen_tokens)

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            yield state

    def __getitem__(self, layer_idx: int) -> LayerState:
        if layer_idx >= len(self.states):
            raise KeyError(
                f"Cache only has {len(self.states)} layers, attempted to access layer {layer_idx}"
            )
        return self.states[layer_idx]

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def _ensure_layer(self, layer_idx: int) -> LayerState:
        while len(self.states) <= layer_idx:
            self.states.append({"attn_state": None, "lact_state": None})
        return self.states[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def advance_seq_length(self, offset: int) -> int:
        if offset < 0:
            raise ValueError(f"`offset` must be non-negative, got {offset}")
        self._seen_tokens += int(offset)
        return self._seen_tokens

    def get_lact_state(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        layer_state = self._ensure_layer(layer_idx)
        return layer_state.get("lact_state")

    def set_lact_state(self, layer_idx: int, state: Dict[str, Any]) -> Dict[str, Any]:
        layer_state = self._ensure_layer(layer_idx)
        layer_state["lact_state"] = state
        return state

    def update_attn_state(
        self,
        layer_idx: int,
        attn_state: AttnState,
        offset: int = 1,
        window_size: Optional[int] = None,
    ) -> LayerState:
        return self.update(
            attn_state=attn_state,
            layer_idx=layer_idx,
            offset=offset,
            cache_kwargs={"window_size": window_size},
        )

    def update(
        self,
        attn_state: Optional[AttnState] = None,
        lact_state: Optional[Dict[str, Any]] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LayerState:
        cache_kwargs = {} if cache_kwargs is None else cache_kwargs
        layer_state = self._ensure_layer(layer_idx)

        if attn_state is not None:
            if not isinstance(attn_state, tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a (key, value) tuple")
            key_state, value_state = attn_state
            window_size = cache_kwargs.get("window_size")
            existing = layer_state.get("attn_state")
            if existing is None:
                next_key, next_value = key_state, value_state
            else:
                prev_key, prev_value = existing
                next_key = torch.cat([prev_key, key_state], dim=1)
                next_value = torch.cat([prev_value, value_state], dim=1)
            if window_size is not None and next_key.shape[1] > window_size:
                next_key = next_key[:, -window_size:, :].contiguous()
                next_value = next_value[:, -window_size:, :].contiguous()
            layer_state["attn_state"] = (next_key, next_value)

        if lact_state is not None:
            layer_state["lact_state"] = lact_state

        return layer_state

    def to_legacy_cache(self) -> Tuple[LayerState, ...]:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Any, ...]] = None,
        seen_tokens: int = 0,
    ) -> "LaCTCache":
        cache = cls(seen_tokens=seen_tokens)
        if past_key_values is None:
            return cache
        if isinstance(past_key_values, cls):
            return past_key_values
        if not isinstance(past_key_values, (list, tuple)):
            raise ValueError(
                f"Unsupported legacy cache type for LaCTCache: {type(past_key_values)}"
            )
        for entry in past_key_values:
            if entry is None:
                cache.states.append({"attn_state": None, "lact_state": None})
            elif isinstance(entry, dict):
                cache.states.append(
                    {
                        "attn_state": entry.get("attn_state"),
                        "lact_state": entry.get("lact_state"),
                    }
                )
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                cache.states.append({"attn_state": tuple(entry), "lact_state": None})
            else:
                raise ValueError(f"Unsupported legacy cache entry type: {type(entry)}")
        return cache
