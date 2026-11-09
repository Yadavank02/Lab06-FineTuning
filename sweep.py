"""Piece 3: Rank sweep against full fine-tuning."""

import json
import time
import lab06_starter as S
from loss_baseline import answer_only_labels


def run_sweep():
    print("=================================================================")
    print("PIECE 3: Rank Sweep Against Full Fine-Tuning")
    print("=================================================================")

    # Fixed training parameters across all comparisons
    N_TRAIN = 600
    N_TEST = 500
    STEPS = 120
    BATCH_SIZE = 8
    SEEDS = [0, 1, 2]

    # Model configuration
    model_dummy, tok = S.load_model()
    train_ex, test_ex, label_ids = S.load_task(
        "emotion", n_train=N_TRAIN, n_test=N_TEST, tokenizer=tok, data_seed=0
    )
    del model_dummy  # Free memory

    # Matrix of configurations: (mode, rank, labels_fn, lr, display_name)
    configs = [
        ("full", None, answer_only_labels, 5e-5, "Full Fine-Tuning (answer-only)"),
        ("lora", 1,    answer_only_labels, 1e-3, "LoRA r=1 (answer-only)"),
        ("lora", 4,    answer_only_labels, 1e-3, "LoRA r=4 (answer-only)"),
        ("lora", 16,   answer_only_labels, 1e-3, "LoRA r=16 (answer-only)"),
        ("lora", 16,   S.all_tokens_labels, 1e-3, "LoRA r=16 (all-tokens loss)"),
    ]

    results = {}
    sweep_start = time.time()

    for mode, r, labels_fn, lr, name in configs:
        print(f"\n>>> Running Configuration: {name} (LR: {lr})")
        results[name] = {"seeds": {}, "trainable": 0, "total": 0}

        for seed in SEEDS:
            print(f"  Training Seed {seed}...")
            info = S.run(
                mode=mode,
                seed=seed,
                train_examples=train_ex,
                test=test_ex,
                label_ids=label_ids,
                r=r if r is not None else 8,
                steps=STEPS,
                lr=lr,
                labels_fn=labels_fn,
                batch_size=BATCH_SIZE
            )
            results[name]["seeds"][seed] = info["accuracy"]
            results[name]["trainable"] = info["trainable"]
            results[name]["total"] = info["total"]
            print(f"    Seed {seed} Test Accuracy : {info['accuracy']:.4f} | Final Loss: {info['last_loss']:.4f} | Time: {info['seconds']:.1f}s")

        accs = [results[name]["seeds"][s] for s in SEEDS]
        mean_acc = sum(accs) / len(accs)
        results[name]["mean_accuracy"] = mean_acc
        print(f"  --> Mean Accuracy for {name}: {mean_acc:.4f}")

    total_duration = time.time() - sweep_start
    print(f"\nSweep completed in {total_duration / 60:.1f} minutes.")

    # Print Summary Table
    print("\n" + "=" * 95)
    print("FINAL SWEEP RESULTS SUMMARY TABLE")
    print("=" * 95)
    print(f"{'Method':<32}{'Trainable':<12}{'% Model':<10}{'Seed 0':<10}{'Seed 1':<10}{'Seed 2':<10}{'Mean':<10}")
    print("-" * 95)

    for name, data in results.items():
        tr = data["trainable"]
        tot = data["total"]
        pct = (tr / tot) * 100.0 if tot > 0 else 0.0
        s0 = data["seeds"][0]
        s1 = data["seeds"][1]
        s2 = data["seeds"][2]
        m = data["mean_accuracy"]
        print(f"{name:<32}{tr:<12,}{pct:<10.3f}%{s0:<10.4f}{s1:<10.4f}{s2:<10.4f}{m:<10.4f}")

    print("=" * 95)

    # Save to disk for memory and plot generation
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved sweep results to sweep_results.json\n")


if __name__ == "__main__":
    run_sweep()