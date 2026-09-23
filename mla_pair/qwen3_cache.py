"""Reference pair cache compatible with Qwen3's DynamicCache length API.

All persistent content and RoPE elements live in pair_layers. This is not a
fused GPU cache.
"""

from __future__ import annotations

from torch import Tensor
from transformers.cache_utils import DynamicCache

from .reference import PairCache, PairCompiler, append_token, cache_scalars


class PairDynamicCache(DynamicCache):
    def __init__(self, config=None):
        try:
            super().__init__(config=config)
        except TypeError:  # Transformers 4.x local correctness tests
            super().__init__()
        self.pair_layers: dict[int, PairCache] = {}

    def append_pair(self, layer_idx: int, content: Tensor, rope: Tensor,
                    compiler: PairCompiler) -> PairCache:
        previous = self.pair_layers.get(layer_idx, PairCache())
        updated = append_token(previous, content, rope, compiler)
        self.pair_layers[layer_idx] = updated
        return updated

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.pair_layers.get(layer_idx)
        return 0 if state is None else state.length

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + query_length, 0

    def persistent_scalars(self) -> int:
        return sum(cache_scalars(layer) * (
            layer.pair_content if layer.pair_content is not None
            else layer.pending_content).shape[0]
            for layer in self.pair_layers.values())

    def persistent_bytes(self) -> int:
        tensors = (
            tensor
            for layer in self.pair_layers.values()
            for tensor in (layer.pair_content, layer.pair_rope,
                           layer.pending_content, layer.pending_rope)
            if tensor is not None
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
