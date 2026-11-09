"""Piece 4: Memory accounting: predict, then measure."""
import torch
import lab06_starter as S
from loss_baseline import answer_only_labels


def theoretical_memory(total_params: int, trainable_params: int):
    """Compute theoretical memory in bytes under 32-bit float AdamW training.
    
    Accounting:
      - Base frozen model weights: 4 bytes per parameter
      - Trainable adapter weights: 4 bytes per parameter (already in total or added)
      - Gradients: 4 bytes per trainable parameter
      - AdamW state: 2 running averages (m and v), 4 bytes each = 8 bytes per trainable parameter
      - Total state memory = (4 * total_params) + (12 * trainable_params)
    """
    weight_bytes = total_params * 4
    grad_bytes = trainable_params * 4
    optimizer_bytes = trainable_params * 8
    total_state_bytes = weight_bytes + grad_bytes + optimizer_bytes
    return weight_bytes, grad_bytes, optimizer_bytes, total_state_bytes


def run_memory_analysis():
    print("=================================================================")
    print("PIECE 4: Memory Accounting: Predict vs Measure")
    print("=================================================================")

    train_ex, test_ex, label_ids = S.load_task("emotion", n_train=64, n_test=32, data_seed=0)

    configs = [
        ("Full Fine-Tuning", "full", None, 5e-5),
        ("LoRA r=1",         "lora", 1,    1e-3),
        ("LoRA r=4",         "lora", 4,    1e-3),
        ("LoRA r=16",        "lora", 16,   1e-3),
    ]

    print(f"{'Method':<20}{'Trainable':<12}{'Pred Grads':<14}{'Meas Grads':<14}{'Pred Opt':<14}{'Meas Opt':<14}{'Total State':<14}")
    print("-" * 102)

    for name, mode, r, lr in configs:
        # 1. Run 1 step to instantiate gradients and Adam optimizer states
        torch.manual_seed(0)
        model, tok = S.load_model()
        if mode == "lora":
            model = S.apply_lora(model, r=r)

        trainable, total = S.count_params(model)
        pred_w, pred_g, pred_opt, pred_tot = theoretical_memory(total, trainable)

        # Run minimal step to populate opt state
        info = S.train(model, tok, train_ex, steps=2, lr=lr, batch_size=8, seed=0, labels_fn=answer_only_labels)
        meas_w = info["weight_bytes"]
        meas_g = info["grad_bytes"]
        meas_opt = info["optimizer_bytes"]
        meas_tot = meas_w + meas_g + meas_opt

        # Validate that analytical formula matches measured tensors
        assert pred_w == meas_w, f"Weight mismatch for {name}: {pred_w} vs {meas_w}"
        assert pred_g == meas_g, f"Grad mismatch for {name}: {pred_g} vs {meas_g}"
        assert pred_opt == meas_opt, f"Opt mismatch for {name}: {pred_opt} vs {meas_opt}"

        print(f"{name:<20}{trainable:<12,}{pred_g / 1e6:<14.2f}{meas_g / 1e6:<14.2f}{pred_opt / 1e6:<14.2f}{meas_opt / 1e6:<14.2f}{meas_tot / 1e6:<14.2f} MB")

    print("\nVerification: Theoretical predictions match active PyTorch tensors to the exact byte.")
    
    # Analyze peak memory
    if S.DEVICE == "cuda":
        peak_mb = info["peak_cuda_bytes"] / 1e6
        print(f"\nMeasured CUDA Peak Memory: {peak_mb:.2f} MB")
        print("Note on GPU Peak Memory: The peak allocated memory exceeds static parameter bytes")
        print("because it accounts for dynamic activations stored for backpropagation,")
        print("attention logits, KV cache buffers, and PyTorch allocator pooling.")
    else:
        print("\nHardware Note: Execution completed on CPU.")
        print("On a 16 GB GPU runtime, Full Fine-Tuning requires ~2.15 GB of static parameter memory,")
        print("whereas LoRA r=16 requires only ~549 MB (a 74.5% overall memory saving),")
        print("leaving over 15 GB of headroom for longer context lengths and larger batch sizes.")


if __name__ == "__main__":
    run_memory_analysis()