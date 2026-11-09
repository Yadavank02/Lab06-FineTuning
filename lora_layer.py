"""Piece 2: LoRA from scratch, verified.

Fulfills Lab 06 Piece 2:
1. Implements LoRALinear(base, r, alpha) using only tensor ops, nn.Linear, and nn.Parameter.
2. Demonstrates output equivalence to base layer at initialization (B=0).
3. Proves trainable parameter count equals r * (in_features + out_features).
4. Demonstrates merged weight layer W_merged = W + (alpha/r)*B*A matches two-path output.
5. Verifies numerical equivalence against Hugging Face's peft library.
6. Computes and validates model-wide LoRA parameters on q_proj and v_proj for ranks 1, 4, 16.
"""
import math
import copy
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import lab06_starter as S


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear layer and adds the trainable low-rank update (alpha / r) * B @ A."""
    
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(2 * r)
        self.scale = self.alpha / float(self.r)

        # Freeze the underlying pretrained base layer
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Trainable low-rank decomposition matrices
        # A maps from in_features -> r; initialized with Kaiming uniform
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # B maps from r -> out_features; initialized to strictly zeros
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear output: x @ W^T + b
        base_out = self.base(x)
        # Low-rank path: (x @ A^T) @ B^T scaled by (alpha / r)
        lora_out = (x @ self.A.t()) @ self.B.t() * self.scale
        return base_out + lora_out

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Return a single consolidated nn.Linear layer holding W + (alpha / r) * B @ A."""
        merged_linear = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=(self.base.bias is not None),
            dtype=self.base.weight.dtype,
            device=self.base.weight.device
        )
        # Compute fused weight matrix
        delta_W = (self.B @ self.A) * self.scale
        merged_linear.weight.copy_(self.base.weight + delta_W)
        if self.base.bias is not None:
            merged_linear.bias.copy_(self.base.bias)
        return merged_linear


def verify_lora_layer_mechanics():
    print("--- 1. Testing LoRALinear Layer Mechanics ---")
    in_dim, out_dim, rank, alpha = 576, 192, 8, 16.0
    base_linear = nn.Linear(in_dim, out_dim)
    lora_linear = LoRALinear(base_linear, r=rank, alpha=alpha)

    # 1. Check trainable parameters
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    expected_params = rank * (in_dim + out_dim)
    assert trainable_params == expected_params, f"Param count mismatch: {trainable_params} vs {expected_params}"
    print(f"Trainable parameters check passed: {trainable_params:,} == {expected_params:,} [r * (in + out)]")

    # 2. Check output at initialization (B=0, so lora_out must be 0)
    x = torch.randn(4, 16, in_dim)
    with torch.no_grad():
        out_lora_init = lora_linear(x)
        out_base_init = base_linear(x)
    assert torch.allclose(out_lora_init, out_base_init, atol=1e-6), "Init check failed: LoRA does not equal base at init!"
    print("Initialization check passed: LoRALinear output matches base layer exactly (B=0).")

    # 3. Check merging after giving non-zero values to B
    nn.init.normal_(lora_linear.B, std=0.02)
    merged_linear = lora_linear.merged()
    with torch.no_grad():
        out_two_path = lora_linear(x)
        out_merged = merged_linear(x)
    diff = (out_two_path - out_merged).abs().max().item()
    assert torch.allclose(out_two_path, out_merged, atol=1e-5), f"Merged check failed! Max diff: {diff}"
    print(f"Merging check passed: Two-path output matches merged output (max diff: {diff:.2e}).\n")


def verify_against_peft():
    print("--- 2. Verifying Equivalence against Hugging Face PEFT ---")
    in_dim, out_dim, rank, alpha = 576, 576, 4, 8.0
    base_linear = nn.Linear(in_dim, out_dim, bias=False)

    # 1. Custom LoRALinear
    my_lora = LoRALinear(copy.deepcopy(base_linear), r=rank, alpha=alpha)

    # 2. PEFT LoRA via get_peft_model
    class SimpleModule(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x)

    dummy_model = SimpleModule(copy.deepcopy(base_linear))
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["linear"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(dummy_model, cfg)

    # Set identical non-zero weights for verification
    torch.manual_seed(42)
    test_A = torch.randn(rank, in_dim) * 0.05
    test_B = torch.randn(out_dim, rank) * 0.05

    # Copy into custom LoRA
    my_lora.A.data.copy_(test_A)
    my_lora.B.data.copy_(test_B)

    # Copy into PEFT module
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            module.lora_A["default"].weight.data.copy_(test_A)
            module.lora_B["default"].weight.data.copy_(test_B)

    x = torch.randn(2, 8, in_dim)
    with torch.no_grad():
        out_my = my_lora(x)
        out_peft = peft_model(x)

    diff = (out_my - out_peft).abs().max().item()
    assert torch.allclose(out_my, out_peft, atol=1e-5), f"PEFT equivalence failed! Diff: {diff}"
    print(f"PEFT verification passed: Custom LoRALinear matches PEFT LoraLayer identically (diff: {diff:.2e}).\n")


def compute_model_lora_parameters():
    print("--- 3. Analytical & Measured LoRA Parameter Counts across SmolLM2-135M ---")
    model, _ = S.load_model()
    total_model_params = sum(p.numel() for p in model.parameters())
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # In SmolLM2-135M:
    # q_proj: in = 576, out = 576
    # v_proj: in = 576, out = num_key_value_heads * head_dim = 3 * 64 = 192
    q_in, q_out = hidden_size, hidden_size
    v_in, v_out = hidden_size, (model.config.num_key_value_heads * (hidden_size // model.config.num_attention_heads))

    print(f"Model Summary: {total_model_params:,} total parameters, {num_layers} layers")
    print(f"Target Modules per layer: q_proj ({q_in} -> {q_out}), v_proj ({v_in} -> {v_out})")
    print(f"Theoretical formula per layer: r * ({q_in + q_out} + {v_in + v_out}) = r * 1,920")
    print(f"Theoretical formula model-wide: 30 * r * 1,920 = 57,600 * r\n")

    print(f"{'Rank (r)':<10}{'Formula Count':<18}{'PEFT Measured':<18}{'% of Model':<15}")
    print("-" * 61)

    for r in [1, 4, 16]:
        formula_count = num_layers * r * ((q_in + q_out) + (v_in + v_out))
        peft_model = S.apply_lora(copy.deepcopy(model), r=r, alpha=2 * r, target_modules=("q_proj", "v_proj"))
        measured_trainable, _ = S.count_params(peft_model)
        pct = (measured_trainable / total_model_params) * 100.0

        assert formula_count == measured_trainable, f"Mismatch at r={r}: {formula_count} vs {measured_trainable}"
        print(f"r = {r:<6}{formula_count:<18,}{measured_trainable:<18,}{pct:.4f}%")

    print("\nAll theoretical formulas match measured PEFT parameter counts exactly.")


if __name__ == "__main__":
    print("=================================================================")
    print("PIECE 2: LoRA From Scratch, Mathematical Verification")
    print("=================================================================")
    verify_lora_layer_mechanics()
    verify_against_peft()
    compute_model_lora_parameters()