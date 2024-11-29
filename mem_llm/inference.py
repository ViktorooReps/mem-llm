import torch

from mem_llm.custom_tqdm import HumanizedTqdm
from mem_llm.interface import Generator, ModelOutput, LayerCache
from mem_llm.tokenizer import CharTokenizer


@torch.no_grad()
def generate(
        seed: str,
        model: Generator,
        tokenizer: CharTokenizer,
        *,
        device: str,
        max_length: int,
        amp_enabled: bool = True,
        progress_bar: bool = True,
        top_k: int = 1,
        temperature: float = 1.0,
) -> list[str]:
    model.eval()
    model = model.to(device)

    start = len(seed)
    end = start + max_length

    saved_tokens = torch.tensor((top_k, 0), dtype=torch.long, device='cpu')

    cache: list[LayerCache] | None = None
    tokens = tokenizer.encode(seed).unsqueeze(0).to(device)  # (1, len(seed)) at first iteration, (top_k, 1) on rest
    cumulative_log_probs = torch.zeros((top_k, 1), dtype=torch.float, device=device)

    for _ in HumanizedTqdm(
            range(start, end),
            total=end - start,
            desc="Generating",
            disable=not progress_bar
    ):
        with torch.amp.autocast(enabled=amp_enabled, device_type=device):
            outputs: ModelOutput = model(tokens=tokens, past_cache=cache)

        cache = outputs.cache

        log_probs = torch.log_softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
        top_k_log_probs, top_k_tokens = torch.topk(log_probs, k=top_k, dim=-1, sorted=True)

        # shape: (top_k, top_k)
        cumulative_log_probs += top_k_log_probs

        # Flatten cumulative log probabilities to consider all (top_k, top_k) combinations
        flattened_log_probs = cumulative_log_probs.view(-1)  # shape: (top_k * top_k)

        # Get the top_k combinations with the highest cumulative log probabilities
        top_k_flat_log_probs, top_k_flat_indices = torch.topk(flattened_log_probs, k=top_k, sorted=True)
        tokens = top_k_tokens.view(-1)[top_k_flat_indices].view(top_k, 1)

        # Update saved tokens

        row_indices = top_k_flat_indices // top_k  # Corresponding row in the previous top_k

        saved_tokens = saved_tokens[row_indices.cpu(), :]
        saved_tokens = torch.concat((saved_tokens, tokens.cpu()), dim=-1)

    return [tokenizer.decode(option) for option in saved_tokens]
