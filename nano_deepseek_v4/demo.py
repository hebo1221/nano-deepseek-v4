from __future__ import annotations

import torch

from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM


def main() -> None:
    torch.manual_seed(0)
    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids, labels=input_ids, output_router_logits=True)
    print(f"logits: {tuple(output.logits.shape)}")
    print(f"loss: {output.loss.item():.4f}")
    if output.mtp_loss is not None:
        print(f"mtp_loss: {output.mtp_loss.item():.4f}")
    print(f"router layers: {len(output.router_logits or [])}")


if __name__ == "__main__":
    main()
