"""H4: LoRA adapters on one part of SAM 2 (memory attention or image encoder); all base weights stay frozen.

python lora.py    builds SAM 2.1, adds LoRA to each part with matched parameter counts, checks it.
"""
import torch
import torch.nn as nn

TARGETS = {  # attention projections only, as SAMed does for SAM's image encoder
    "memory_attention": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "image_encoder": ("qkv", "proj"),
}


class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)). B starts at zero, so the model is unchanged until training."""

    def __init__(self, base, r, alpha=None):
        super().__init__()
        self.base = base
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.A = nn.Parameter(torch.empty(r, base.in_features, **kw))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, **kw))
        self.scale = (alpha or r) / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def _targets(model, part):
    names = TARGETS[part]
    return [(m, n) for m in getattr(model, part).modules() for n, c in m.named_children()
            if isinstance(c, nn.Linear) and n in names]


def per_rank(model, part):
    """LoRA parameters added to `part` per unit of rank."""
    return sum(getattr(m, n).in_features + getattr(m, n).out_features for m, n in _targets(model, part))


def matched_rank(model, part, n_params):
    """Rank that gives `part` about n_params LoRA parameters, for a fair memory-vs-encoder comparison."""
    return max(1, round(n_params / per_rank(model, part)))


def add_lora(model, part, r):
    """Freeze everything, wrap the target layers of `part`, return the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad = False
    for m, n in _targets(model, part):
        setattr(m, n, LoRALinear(getattr(m, n), r))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import argparse

    from sam2.build_sam import build_sam2_hf

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank for the memory attention")
    args = ap.parse_args()
    torch.manual_seed(0)

    mem = build_sam2_hf(args.model, device="cpu")
    n_mem = add_lora(mem, "memory_attention", args.rank)
    enc = build_sam2_hf(args.model, device="cpu")
    r_enc = matched_rank(enc, "image_encoder", n_mem)
    n_enc = add_lora(enc, "image_encoder", r_enc)
    total = sum(p.numel() for p in mem.parameters())
    print(f"memory attention: rank {args.rank}, {n_mem:,} trainable ({100 * n_mem / total:.2f}% of {total:,})")
    print(f"image encoder:    rank {r_enc}, {n_enc:,} trainable ({100 * n_enc / total:.2f}%)")

    # checks: unchanged at start, only LoRA weights get gradients
    ref = build_sam2_hf(args.model, device="cpu")
    curr, memory = torch.randn(4096, 1, 256), torch.randn(4096, 1, 64)
    pos, mpos = torch.randn(4096, 1, 256), torch.randn(4096, 1, 64)
    with torch.no_grad():
        same = torch.allclose(ref.memory_attention(curr, memory, pos, mpos), mem.memory_attention(curr, memory, pos, mpos))
    assert same, "LoRA changed the memory attention before training"
    mem.memory_attention(curr, memory, pos, mpos).sum().backward()
    grads = {n for n, p in mem.named_parameters() if p.grad is not None}
    assert grads and all(n.endswith((".A", ".B")) for n in grads), "a frozen weight got a gradient"
    print(f"checks passed: output unchanged at start, gradients only in {len(grads)} LoRA tensors")
