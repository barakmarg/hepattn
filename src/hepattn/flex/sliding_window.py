from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)

    return mask_mod


def sliding_window_mask_wrapped(window_size: int, q_len: Tensor) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        diagonal = (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
        wrap = ((q_idx - kv_idx + q_len[0]) <= window_size // 2) | ((kv_idx - q_idx + q_len[0]) <= window_size // 2)
        return diagonal | wrap

    return mask_mod


def global_token_window_mask(window_size: int, n_global_tokens: int, seq_len: int) -> _mask_mod_signature:
    """Sliding window attention where the last n_global_tokens tokens are globally visible.

    Every regular token attends to: its local window + all global tokens.
    Every global token attends to: all tokens in the sequence.
    """
    n_regular = seq_len - n_global_tokens

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        in_window = (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
        kv_is_global = kv_idx >= n_regular
        q_is_global = q_idx >= n_regular
        return in_window | kv_is_global | q_is_global

    return mask_mod
