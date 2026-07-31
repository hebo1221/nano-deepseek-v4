from __future__ import annotations

import torch

from . import __version__
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM


def main() -> None:
    torch.manual_seed(0)
    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    with torch.inference_mode():
        full = model(input_ids, output_router_logits=True)
        prefix = model(input_ids[:, :5], use_cache=True)
        if prefix.past_key_values is None:
            raise RuntimeError("demo expected an inference cache")
        prefix_tokens = prefix.past_key_values.get_seq_length()
        suffix = model(
            input_ids[:, 5:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        if suffix.past_key_values is None:
            raise RuntimeError("demo expected an updated inference cache")
        cached_logits = torch.cat([prefix.logits, suffix.logits], dim=1)
        cache_matches = torch.allclose(cached_logits, full.logits, atol=1e-4, rtol=1e-4)
        if not cache_matches:
            raise RuntimeError("cached inference does not match the full forward pass")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    schedule = " -> ".join(layer.replace("_attention", "") for layer in config.layer_types or [])
    mlp_schedule = " -> ".join(config.mlp_layer_types or [])

    print(f"nano-deepseek-v4 {__version__}")
    print(f"parameters: {parameters:,}")
    print(f"attention: {schedule}")
    print(f"mlp: {mlp_schedule}")
    print(f"logits: {tuple(full.logits.shape)}")
    print(f"cache tokens: {prefix_tokens} -> {suffix.past_key_values.get_seq_length()}")
    print(f"cached/full match: {cache_matches}")
    print(f"router layers: {len(full.router_logits or [])}")


if __name__ == "__main__":
    main()
